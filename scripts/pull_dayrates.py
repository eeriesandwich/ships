"""
Download CMB.TECH / Euronav 6-K earnings press releases from SEC EDGAR,
filtered to filings that actually contain a TCE (time charter equivalent)
day-rate table, using EDGAR's full-text search API.

"""
import time
import json
import requests
from pathlib import Path

USER_AGENT = "Joan Casaramona portfolio-project joancasaramonaa@gmail.com"  # EDIT THIS
CIK = "1604481"  # Euronav NV / CMB.TECH NV
OUT_DIR = Path("../raw/dayrates")
SEARCH_TERM = "time charter equivalent"

HEADERS = {"User-Agent": USER_AGENT}


def fetch_all_hits():
    """Page through EDGAR full-text search results (10 per page)."""
    hits = []
    start = 0
    while True:
        url = (
            "https://efts.sec.gov/LATEST/search-index"
            f"?q=%22{SEARCH_TERM.replace(' ', '+')}%22"
            f"&forms=6-K&ciks={CIK.zfill(10)}&from={start}"
        )
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()

        if start == 0:
            total = data["hits"]["total"]["value"]
            print(f"Total hits: {total}")

        page_hits = data["hits"]["hits"]
        if not page_hits:
            break
        hits.extend(page_hits)
        start += len(page_hits)
        time.sleep(0.3)  # be polite

        if start >= data["hits"]["total"]["value"]:
            break

    return hits


def download_hits(hits):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for hit in hits:
        # _id format is typically "<accession-with-dashes>:<filename>"
        hit_id = hit.get("_id", "")
        if ":" not in hit_id:
            print(f"Skipping unexpected hit id format: {hit_id!r}")
            continue
        accession_dashed, filename = hit_id.split(":", 1)
        accession_nodash = accession_dashed.replace("-", "")

        source = hit.get("_source", {})
        file_date = source.get("file_date", "unknown-date")

        url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{accession_nodash}/{filename}"
        out_path = OUT_DIR / f"{file_date}_{filename}"

        if out_path.exists():
            print(f"Already have {out_path.name}, skipping")
            continue

        print(f"Downloading {file_date}  {url}")
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            print(f"  FAILED ({resp.status_code}): {url}")
            continue
        out_path.write_bytes(resp.content)
        time.sleep(0.3)  # be polite -- do not hammer SEC's servers

    print(f"Done. Files saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    hits = fetch_all_hits()
    print(f"Fetched {len(hits)} hit records")
    if hits:
        print("Sample hit record (first one), to sanity-check the shape:")
        print(json.dumps(hits[0], indent=2)[:1500])
    download_hits(hits)
