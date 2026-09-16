"""
Extract TCE (time charter equivalent) day-rate tables from CMB.TECH /
Euronav SEC earnings-release HTML exhibits.

Design: pull every period/vessel-class/rate-type value the table actually
shows (quarterly AND full-year columns, wherever present), tag each row
with which source filing it came from, and defer picking "which quarter
counts" to the Power Query layer (pull everything, filter downstream).

Table detection is deliberately defensive. The phrase "time charter
equivalent" appears in prose as well as above the rate table, and the
table that physically follows a prose mention is often something else
entirely: a board-of-directors roster, a fleet list with vessel counts
and carrying values, a contact block, a page-break spacer. A candidate
table is therefore only accepted if it both mentions rate vocabulary and
produces values that fall in a plausible USD-per-day range.

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

# A genuine rate table always states its unit or names the rate itself.
# Matching on vessel classes alone is not enough: fleet lists and operating
# expense breakdowns name the same vessel classes.
RATE_UNIT_KEYWORDS = (
    "PER DAY", "/DAY", "TCE", "TIME CHARTER EQUIVALENT",
    "SPOT RATE", "TIME CHARTER RATE",
)

# Vocabulary belonging to tables that are definitely not day rates, even
# when they mention vessel classes and sit near a TCE mention.
NON_RATE_KEYWORDS = (
    "IN THOUSANDS", "OPERATING EXPENSES", "CARRYING VALUE",
    "NUMBER OF VESSELS", "NUMBERS OF VESSELS", "DATE OF EXPIRY",
)

# Day rates for these vessel classes have historically run from roughly
# USD 5,000 to USD 200,000. The band below is deliberately wider than that
# so genuine outliers survive, while vessel counts (1-70), ages (30-75) and
# carrying values in thousands (600,000+) fall outside it.
MIN_PLAUSIBLE_RATE = 1_000
MAX_PLAUSIBLE_RATE = 500_000
MIN_RATE_SHARE = 0.5

# How many tables after each anchor mention to consider, and overall cap.
TABLES_PER_ANCHOR = 10
MAX_CANDIDATES = 40

# "20.975" in a table where every other cell reads "23,081" is a European
# thousands separator, not a decimal. Day rates are never quoted to three
# decimal places, so this pattern is unambiguous in this dataset.
EURO_THOUSANDS = re.compile(r"-?\d{1,3}(?:\.\d{3})+")


def clean_text(el):
    text = el.get_text(separator=" ", strip=True).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_period_label(label):
    """'Fourth quarter 2015' -> (Quarter, 4, 2015); 'Full year 2015' -> (Annual, None, 2015);
    'Quarter-to-Date Q4 2025' -> (QuarterToDate, 4, 2025), kept distinct from a completed
    quarter since it's a partial-period estimate rather than an actual result."""
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
    raw = text.replace("\xa0", " ").strip().rstrip("*").strip()
    if raw in ("", "-", "--", "n.a.", "n/a", "N/A"):
        return None

    if EURO_THOUSANDS.fullmatch(raw):
        return float(raw.replace(".", ""))

    normalised = raw.replace(",", "").replace("–", "-")
    try:
        return float(normalised)
    except ValueError:
        pass

    # Cells that pack extra context after the figure, e.g. "43096 (85% fixed)".
    lead_match = re.match(r"\s*(-?\d+(?:\.\d+)?)", normalised)
    if lead_match:
        try:
            return float(lead_match.group(1))
        except ValueError:
            return None
    return None


def find_candidate_tables(soup):
    """Collect tables that follow any mention of 'time charter equivalent'.

    Every mention is considered, not just the first, because the phrase turns
    up in prose and footnotes as well as immediately above the rate table.
    Order is preserved and duplicates are dropped, so the caller can try the
    nearest candidates first.
    """
    anchors = soup.find_all(string=re.compile(r"time charter equivalent", re.IGNORECASE))
    candidates = []
    seen = set()

    for anchor in anchors:
        el = anchor
        while el is not None and not hasattr(el, "find_next"):
            el = el.parent
        if el is None:
            continue
        node = el
        for _ in range(TABLES_PER_ANCHOR):
            node = node.find_next("table")
            if node is None:
                break
            if id(node) in seen:
                continue
            seen.add(id(node))
            candidates.append(node)
            if len(candidates) >= MAX_CANDIDATES:
                return candidates
    return candidates


def mentions_rates(table):
    text = clean_text(table).upper()
    if any(kw in text for kw in NON_RATE_KEYWORDS):
        return False
    return any(kw in text for kw in RATE_UNIT_KEYWORDS)


def looks_like_rate_table(records):
    """Accept a table only if most of its values sit in a believable
    USD-per-day range. This is what separates a rate table from a fleet
    table (counts and carrying values) or a personnel roster (ages)."""
    if not records:
        return False
    values = [r["ValueUSDPerDay"] for r in records]
    in_band = sum(1 for v in values if MIN_PLAUSIBLE_RATE <= v <= MAX_PLAUSIBLE_RATE)
    return in_band / len(values) >= MIN_RATE_SHARE


def extract_table_rows(table, source_file, filing_date):
    rows = table.find_all("tr")
    if not rows:
        return []

    # Some filings put a blank spacer row before the real header row
    # (alternating spacer/value column layout). Scan forward to find the
    # first row whose non-label cells actually contain text.
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
        value_texts = [clean_text(c) for c in cells[1:]]

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

        for i, period_label in enumerate(period_labels):
            if i >= len(value_texts):
                break
            value = clean_number(value_texts[i])
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

    candidates = find_candidate_tables(soup)
    if not candidates:
        print(f"  NO TCE TABLE FOUND: {filename}")
        return []

    for table in candidates:
        if not mentions_rates(table):
            continue
        records = extract_table_rows(table, filename, filing_date)
        if looks_like_rate_table(records):
            return records

    print(f"  NO RATE TABLE MATCHED: {filename}")
    return []


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_dayrates.py <folder_of_htm_files> <output_csv>")
        sys.exit(1)

    in_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    all_records = []
    files = sorted(in_dir.glob("*.htm")) + sorted(in_dir.glob("*.html"))
    print(f"Found {len(files)} files")

    parsed_files = 0
    for path in files:
        records = process_file(path)
        if records:
            parsed_files += 1
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

    print(f"Extracted {len(all_records)} rows from {parsed_files} of {len(files)} files")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
