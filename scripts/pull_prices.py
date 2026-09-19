"""Pull daily OHLCV price history from Yahoo Finance with provenance stamping.

Writes one CSV per ticker plus a manifest recording when each file was pulled,
which yfinance version produced it, and integrity checks on the adjustment.

Why the pull date matters: Yahoo recalculates Adj Close on every download,
because each new dividend re-adjusts the whole history behind it. A file is a
snapshot as of its pull date, and a fresh pull will produce slightly different
adjusted values for every historical row. The CSVs in the repo are therefore
the reproducible artifact, not the pull command.

Usage from the notebook:
    %run ../scripts/pull_prices.py
    pull_prices(TANKER_TICKERS, start="2005-01-03",
                out_dir=Path("../raw/tanker"), lineage="Tanker")
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# Adj Close first: it is the series every return measure must use.
# Open/High/Low are split-adjusted but NOT dividend-adjusted, so they are not on
# the same basis as Adj Close. Keep them only for raw price-level charts.
COLUMN_ORDER = [
    "Date", "Adj Close", "Close", "Open", "High", "Low",
    "Volume", "Ticker", "Lineage", "PullDate",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns MultiIndex columns (field, ticker).

    Writing that frame straight to CSV emits a second header row holding the
    ticker under every column, which Power Query imports as a junk data row.
    Collapse to the field level.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _adjustment_checks(df: pd.DataFrame) -> dict:
    """Integrity checks on the dividend/split back-adjustment.

    ratio_last should be ~1.0: nothing left to adjust for at the latest date.
    The ratio should only ever rise going forward in time; a real fall would
    indicate a split handled incorrectly. Tolerance is relative, because the
    ratio is computed from float32-precision Yahoo values and shows noise at
    roughly 1e-6.
    """
    if not {"Adj Close", "Close"}.issubset(df.columns):
        return {"adjustment_checked": False}

    valid = df[df["Close"].notna() & (df["Close"] != 0) & df["Adj Close"].notna()]
    if len(valid) < 2:
        return {"adjustment_checked": False}

    r = (valid["Adj Close"] / valid["Close"]).to_numpy()
    rel_step = (r[1:] - r[:-1]) / r[:-1]
    return {
        "adjustment_checked": True,
        "adj_ratio_first": round(float(r[0]), 6),
        "adj_ratio_last": round(float(r[-1]), 6),
        "cumulative_adjustment_factor": round(float(r[-1] / r[0]), 4),
        "negative_adjustment_steps": int((rel_step < -1e-5).sum()),
    }


def _liquidity_checks(df: pd.DataFrame) -> dict:
    """Staleness indicators. A high zero-return share means the quote is being
    carried forward on days with no trading, which dilutes any seasonal signal.
    """
    if "Adj Close" not in df.columns:
        return {}
    ret = df["Adj Close"].pct_change().dropna()
    out = {}
    if len(ret):
        out["zero_return_pct"] = round(100 * float((ret.abs() < 1e-12).mean()), 2)
    if "Volume" in df.columns and df["Volume"].notna().any():
        out["median_volume"] = int(df["Volume"].median())
        out["zero_volume_days"] = int((df["Volume"] == 0).sum())
    return out


def pull_prices(tickers: dict, start: str, out_dir: Path, lineage: str, end=None):
    """Pull each ticker to its own CSV and update the directory manifest.

    tickers: {ticker: human-readable description}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pulled_at = datetime.now(timezone.utc)
    pull_date = pulled_at.strftime("%Y-%m-%d")
    end = end or date.today().isoformat()
    manifest_path = out_dir / "_manifest.json"

    entries = []
    for ticker, name in tickers.items():
        print(f"Pulling {ticker} ({name})...")

        # auto_adjust=False keeps BOTH Close and Adj Close. Do not change:
        # raw Close for price-level charts, Adj Close for every return measure.
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=False, actions=False, progress=False)

        if df is None or df.empty:
            print(f"  WARNING: no data returned for {ticker}")
            entries.append({"ticker": ticker, "description": name,
                            "lineage": lineage, "status": "empty",
                            "pulled_at_utc": pulled_at.isoformat()})
            continue

        df = _flatten_columns(df).reset_index()
        df["Ticker"] = ticker
        df["Lineage"] = lineage
        df["PullDate"] = pull_date

        checks = _adjustment_checks(df)
        liq = _liquidity_checks(df)

        df = df[[c for c in COLUMN_ORDER if c in df.columns]]

        out_path = out_dir / f"{ticker}.csv"
        df.to_csv(out_path, index=False)

        entry = {
            "ticker": ticker,
            "description": name,
            "lineage": lineage,
            "file": out_path.name,
            "pulled_at_utc": pulled_at.isoformat(),
            "pull_date": pull_date,
            "requested_start": str(start),
            "requested_end": str(end),
            "auto_adjust": False,
            "yfinance_version": getattr(yf, "__version__", "unknown"),
            "pandas_version": pd.__version__,
            "rows": int(len(df)),
            "first_date": str(pd.to_datetime(df["Date"]).min().date()),
            "last_date": str(pd.to_datetime(df["Date"]).max().date()),
            "columns": list(df.columns),
            "sha256": _sha256(out_path),
            "status": "ok",
        }
        entry.update(checks)
        entry.update(liq)
        entries.append(entry)

        warn = ""
        if checks.get("adjustment_checked"):
            if abs(checks["adj_ratio_last"] - 1.0) > 0.01:
                warn += "  [WARN last Adj/Close ratio not ~1.0]"
            if checks["negative_adjustment_steps"]:
                warn += "  [WARN adjustment ratio falls]"
        if liq.get("zero_return_pct", 0) > 10:
            warn += f"  [WARN {liq['zero_return_pct']}% zero-return days: stale quotes]"

        print(f"  saved {entry['rows']} rows to {out_path} "
              f"({entry['first_date']} to {entry['last_date']}), "
              f"cumulative adjustment {checks.get('cumulative_adjustment_factor','n/a')}x"
              f"{warn}")

    # Merge so pulling one lineage does not erase the other's record.
    existing = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    by_ticker = {e.get("ticker"): e for e in existing}
    for e in entries:
        by_ticker[e["ticker"]] = e
    manifest_path.write_text(
        json.dumps(sorted(by_ticker.values(), key=lambda e: e["ticker"]), indent=2),
        encoding="utf-8",
    )
    print(f"manifest updated: {manifest_path}")
    return entries
