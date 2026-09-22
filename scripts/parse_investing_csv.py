"""Normalize investing.com historical-data CSV exports into the repo's freight schema.

investing.com's historical-data pages (Baltic Dry, Baltic Dirty Tanker, Baltic
Capesize) offer a date-range picker and a Download button, which is a cleaner
and more reproducible source than capturing the TradingView chart widget's
WebSocket frames. The export needs normalizing: rows come newest-first, numbers
carry comma thousands separators, dates come in several formats depending on
locale settings, and volume uses K/M/B suffixes.

The Baltic indices publish one fix per day with no intraday trading, so the
export's Open/High/Low columns are all equal to Price. The script checks that
and, when it holds, writes Date,Close only rather than implying an OHLC series
that does not exist.

The range picker usually caps how much history one download returns, so pull in
chunks and pass them all at once. Overlapping chunks are fine: rows are
de-duplicated by date, keeping the last value seen.

Usage:
    python scripts/parse_investing_csv.py data/freight/BDI.csv \
        --source-url https://www.investing.com/indices/baltic-dry-chart \
        --symbol "Baltic Dry Index" \
        vendor/freight_captures/bdi_*.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATE_FORMATS = ["%b %d, %Y", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"]
SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_number(v):
    """'3,327.00' -> 3327.0 ; '1.2K' -> 1200.0 ; '-' / '' -> None."""
    if pd.isna(v):
        return None
    s = str(v).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "N/A"):
        return None
    m = re.fullmatch(r"(-?\d*\.?\d+)([KMB])", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1)) * SUFFIX[m.group(2).upper()]
    try:
        return float(s)
    except ValueError:
        return None


def parse_dates(series: pd.Series) -> pd.Series:
    """Try each known export format; fall back to pandas inference."""
    for fmt in DATE_FORMATS:
        out = pd.to_datetime(series, format=fmt, errors="coerce")
        if out.notna().mean() > 0.95:
            return out
    return pd.to_datetime(series, errors="coerce", dayfirst=False)


def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]

    date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
    if date_col is None:
        raise ValueError(f"{path}: no Date column, found {list(df.columns)}")

    out = pd.DataFrame({"Date": parse_dates(df[date_col])})
    for src, dst in [("Price", "Close"), ("Open", "Open"),
                     ("High", "High"), ("Low", "Low")]:
        if src in df.columns:
            out[dst] = df[src].map(parse_number)
    if "Close" not in out.columns and "Close" in df.columns:
        out["Close"] = df["Close"].map(parse_number)

    bad = out["Date"].isna().sum()
    if bad:
        print(f"  {path.name}: dropped {bad} rows with unparseable dates")
    return out.dropna(subset=["Date", "Close"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", help="e.g. data/freight/BDI.csv")
    p.add_argument("inputs", nargs="+", help="downloaded CSV chunks")
    p.add_argument("--source-url", required=True)
    p.add_argument("--symbol", required=True, help="series name as the page shows it")
    p.add_argument("--keep-ohlc", action="store_true",
                   help="keep Open/High/Low even when they duplicate Close")
    args = p.parse_args()

    frames, sources = [], []
    for raw in args.inputs:
        path = Path(raw)
        print(f"reading {path}")
        frames.append(load_one(path))
        sources.append({"file": path.name, "sha256": _sha256(path)})

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = (df.sort_values("Date")
            .drop_duplicates(subset="Date", keep="last")
            .reset_index(drop=True))
    if before != len(df):
        print(f"de-duplicated {before - len(df)} overlapping rows across chunks")

    ohlc_present = all(c in df.columns for c in ["Open", "High", "Low"])
    degenerate = False
    if ohlc_present:
        degenerate = bool(
            df[["Open", "High", "Low"]].eq(df["Close"], axis=0).all().all()
        )
        if degenerate:
            print("Open/High/Low are identical to Close on every row "
                  "(one published fix per day, no intraday).")
            if not args.keep_ohlc:
                df = df[["Date", "Close"]]
        else:
            n = int((~df[["Open", "High", "Low"]].eq(df["Close"], axis=0)
                     .all(axis=1)).sum())
            print(f"WARNING: {n} rows have OHLC differing from Close. Keeping all columns.")

    gaps = df["Date"].diff().dt.days.dropna()
    median_gap = float(gaps.median()) if len(gaps) else float("nan")
    grain = ("daily" if median_gap <= 1.5
             else "weekly" if median_gap <= 8
             else "monthly" if median_gap <= 35
             else "coarser than monthly")
    print(f"median gap between rows: {median_gap} days -> {grain}")
    if grain != "daily":
        print("WARNING: expected daily bars. Check the Time Frame selector on the page.")

    big = gaps[gaps > 10]
    if len(big):
        idx = big.index
        print(f"{len(big)} gaps longer than 10 days, e.g.:")
        for i in list(idx)[:5]:
            print(f"  {df['Date'].iloc[i-1].date()} -> {df['Date'].iloc[i].date()}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df.to_csv(out_path, index=False)

    manifest_path = out_path.parent / "_manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing[out_path.name] = {
        "symbol": args.symbol,
        "source_url": args.source_url,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "investing.com historical-data CSV export (Download button)",
        "source_files": sources,
        "rows": int(len(df)),
        "first_date": df["Date"].iloc[0],
        "last_date": df["Date"].iloc[-1],
        "median_gap_days": median_gap,
        "grain": grain,
        "ohlc_degenerate": degenerate,
        "columns": list(df.columns),
        "sha256": _sha256(out_path),
    }
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    print(f"\nwrote {len(df)} rows to {out_path} "
          f"({df['Date'].iloc[0]} to {df['Date'].iloc[-1]})")
    print(f"manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
