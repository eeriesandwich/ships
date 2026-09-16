"""
Extract TCE (time charter equivalent) day-rate tables from CMB.TECH /
Euronav SEC earnings-release HTML exhibits.

Design: pull every period/vessel-class/rate-type value the table actually
shows (quarterly AND full-year columns, wherever present), tag each row
with which source filing it came from, and defer picking "which quarter
counts" to the Power Query layer -- same philosophy as the rest of this
project (pull everything, filter downstream).

Usage:
    python extract_dayrates.py <folder_of_htm_files> <output_csv>
"""
import sys
import re
import csv
from pathlib import Path
from bs4 import BeautifulSoup

MONTHS_QUARTER_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
}


def clean_text(el):
    text = el.get_text(separator=" ", strip=True).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_period_label(label):
    """'Fourth quarter 2015' -> (Quarter, 4, 2015); 'Full year 2015' -> (Annual, None, 2015);
    'Quarter-to-Date Q4 2025' -> (QuarterToDate, 4, 2025) -- kept distinct from a completed
    quarter since it's a partial-period estimate, not an actual result."""
    label_low = label.lower()
    year_match = re.search(r"(19|20)\d{2}", label)
    year = int(year_match.group()) if year_match else None

    is_qtd = "quarter-to-date" in label_low or "quarter to date" in label_low or "qtd" in label_low

    if "full year" in label_low or "year ended" in label_low or "annual" in label_low:
        return "Annual", None, year

    quarter = None
    for word, q in MONTHS_QUARTER_WORDS.items():
        if word in label_low:
            quarter = q
            break
    if quarter is None:
        q_match = re.search(r"\bq([1-4])\b", label_low)
        if q_match:
            quarter = int(q_match.group(1))

    if quarter is not None:
        return ("QuarterToDate" if is_qtd else "Quarter"), quarter, year

    return "Unknown", None, year


def clean_number(text):
    text = text.replace(",", "").replace("–", "-").strip()
    if text in ("", "-", "n.a.", "n/a", "N/A"):
        return None
    text = text.rstrip("*")
    try:
        return float(text)
    except ValueError:
        return None


def find_tce_table(soup):
    """Find the table following text mentioning 'time charter equivalent'."""
    anchor = soup.find(string=re.compile(r"time charter equivalent", re.IGNORECASE))
    if anchor is None:
        return None
    # anchor might be a NavigableString; walk up to a tag, then find next table
    el = anchor
    while el is not None and not hasattr(el, "find_next"):
        el = el.parent
    if el is None:
        return None
    return el.find_next("table")


def extract_table_rows(table, source_file, filing_date):
    rows = table.find_all("tr")
    if not rows:
        return []

    # Some filings put a blank spacer row before the real header row (alternating
    # spacer/value column layout). Scan forward to find the first row whose
    # non-label cells actually contain text -- that's the real header.
    header_idx = None
    period_labels = []
    for idx, tr in enumerate(rows):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        labels = [clean_text(c) for c in cells[1:]]
        if any(labels):
            header_idx = idx
            period_labels = labels
            break

    if header_idx is None:
        return []

    records = []
    current_vessel_class = None

    for tr in rows[header_idx + 1:]:
        cells = tr.find_all("td")
        if not cells:
            continue
        first_cell = cells[0]
        first_text = clean_text(first_cell)
        value_cells = cells[1:]
        value_texts = [clean_text(c) for c in value_cells]

        is_category_row = (
            first_cell.get("colspan") is not None
            and all(v in ("", "\xa0") for v in value_texts)
        )
        if is_category_row:
            if first_text:
                current_vessel_class = first_text
            continue

        if not first_text:
            continue

        # align value cells to period_labels positionally (best effort)
        for i, period_label in enumerate(period_labels):
            if i >= len(value_texts):
                break
            raw_val = value_texts[i]
            value = clean_number(raw_val)
            if value is None:
                continue
            period_type, quarter, year = parse_period_label(period_label)
            records.append({
                "SourceFile": source_file,
                "FilingDate": filing_date,
                "VesselClass": current_vessel_class,
                "RateType": first_text.rstrip("*").strip(),
                "PeriodLabel": period_label,
                "PeriodType": period_type,
                "Quarter": quarter,
                "Year": year,
                "ValueUSDPerDay": value,
            })
    return records


def process_file(path):
    filename = path.name
    # our download script saved files as "YYYY-MM-DD_originalname.htm"
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})_", filename)
    filing_date = date_match.group(1) if date_match else "unknown"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    table = find_tce_table(soup)
    if table is None:
        print(f"  NO TCE TABLE FOUND: {filename}")
        return []

    records = extract_table_rows(table, filename, filing_date)
    if not records:
        print(f"  TABLE FOUND BUT NO ROWS PARSED: {filename}")
    return records


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_dayrates.py <folder_of_htm_files> <output_csv>")
        sys.exit(1)

    in_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    all_records = []
    files = sorted(in_dir.glob("*.htm")) + sorted(in_dir.glob("*.html"))
    print(f"Found {len(files)} files")

    for path in files:
        records = process_file(path)
        all_records.extend(records)

    if not all_records:
        print("No records extracted at all -- check the parser against a sample file.")
        sys.exit(1)

    fieldnames = ["SourceFile", "FilingDate", "VesselClass", "RateType",
                  "PeriodLabel", "PeriodType", "Quarter", "Year", "ValueUSDPerDay"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Extracted {len(all_records)} rows from {len(files)} files")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
