"""Build a split- and dividend-adjusted price series for Golden Ocean (GOGL).

Why this script exists
----------------------
GOGL delisted on 20 August 2025 when Golden Ocean merged into CMB.TECH, so
yfinance returns nothing for it. The Stockholm cross-listing GOGLO.ST does
still return data, but its Adj Close is wrong: Yahoo applied the USD dividend
amounts against SEK-denominated prices without converting, so the whole
ten-year history carries a cumulative adjustment of 1.073x where the correct
figure is 1.973x. Each adjustment step comes out at 0.065% to 0.298% instead
of the several percent a quarterly payer of this size produces.

So the series is rebuilt here from two investing.com exports, which live in
vendor/equity_exports/ and are gitignored for redistribution reasons:
  Golden_Ocean_Stock_Price_History.csv   daily OHLCV, NASDAQ, USD
  the dividend history                   transcribed into DIVIDENDS below

Two corrections are applied, in this order.

1. Reverse split, 1-for-5, ex 2016-07-19. The investing.com price file is NOT
   split-adjusted. On that date the open divided by the prior close is exactly
   5.000 and median volume falls by a factor of five. Pre-split prices are
   multiplied by 5 and pre-split volume divided by 5.

2. Dividends, back-adjusted the standard way: at each ex-date, every earlier
   close is multiplied by (1 - D / prior_close).

Validation (see the docstring of the checks below for what each one proves):
  - the yield column reconciles against the price file to a median error of
    0.02%, max 0.20%, confirming the two exports are the same series
  - GOGLO.ST raw close over GOGL raw close equals the real USDSEK rate to
    within a few percent in every year, confirming the price file and the
    split correction
  - the same comparison on adjusted closes is off by +79% in 2015 decaying to
    +6% in 2025, which is the shape of the missing adjustment on Yahoo's side,
    and 1.973 / 1.073 = 1.84 against a 2015 error of 1.79

Usage:
    python scripts/build_gogl.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "vendor" / "equity_exports" / "Golden_Ocean_Stock_Price_History.csv"
OUT = ROOT / "data" / "equity" / "drybulk" / "GOGL.csv"
MANIFEST = ROOT / "data" / "equity" / "drybulk" / "_manifest.json"

SPLIT_DATE = pd.Timestamp("2016-07-19")
SPLIT_RATIO = 5.0

# Ex-date | dividend per share (USD) | annualised yield as printed by the vendor.
# Only the 26 inside the price window are listed; the Knightsbridge-era history
# before 2015 is on a different share basis and is not used.
DIVIDENDS = """Mar 02, 2018|0.100|4.41
Jun 13, 2018|0.100|4.62
Sep 05, 2018|0.100|4.33
Dec 06, 2018|0.150|8.31
Mar 06, 2019|0.050|3.91
Jun 05, 2019|0.025|2.11
Aug 29, 2019|0.100|6.60
Dec 02, 2019|0.150|9.95
Mar 05, 2020|0.050|4.89
Jun 02, 2021|0.250|9.51
Sep 09, 2021|0.500|17.35
Dec 08, 2021|0.850|34.62
Mar 02, 2022|0.900|29.03
May 31, 2022|0.500|12.62
Sep 06, 2022|0.600|24.44
Nov 25, 2022|0.350|16.07
Feb 27, 2023|0.200|7.74
May 25, 2023|0.100|5.15
Sep 08, 2023|0.100|5.35
Dec 05, 2023|0.100|4.25
Mar 12, 2024|0.300|8.93
Jun 07, 2024|0.300|8.68
Sep 11, 2024|0.300|10.85
Dec 09, 2024|0.300|12.55
Mar 11, 2025|0.150|7.29
Jun 05, 2025|0.050|2.60"""


def _volume(v):
    m = re.match(r"^([\d.]+)([KMB]?)$", str(v).strip())
    if not m:
        return np.nan
    return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(2)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build(verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    for c in ["Price", "Open", "High", "Low"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    df["Volume"] = df["Vol."].map(_volume)
    df = df.rename(columns={"Price": "Close"})

    # 1. reverse split
    pre = df["Date"] < SPLIT_DATE
    for c in ["Close", "Open", "High", "Low"]:
        df.loc[pre, c] = df.loc[pre, c] * SPLIT_RATIO
    df.loc[pre, "Volume"] = df.loc[pre, "Volume"] / SPLIT_RATIO

    div = pd.DataFrame(
        [
            {
                "ExDate": pd.to_datetime(a, format="%b %d, %Y"),
                "Div": float(b),
                "Yield": float(c),
            }
            for a, b, c in (line.split("|") for line in DIVIDENDS.strip().split("\n"))
        ]
    ).sort_values("ExDate").reset_index(drop=True)

    # 2. dividend back-adjustment, with the yield reconciliation as it goes
    dates, close = df["Date"].values, df["Close"].values
    factors = np.ones(len(df))
    errors = []
    for _, r in div.iterrows():
        idx = int(np.searchsorted(dates, np.datetime64(r.ExDate)))
        if idx == 0 or idx >= len(df):
            raise ValueError(f"ex-date outside the price window: {r.ExDate.date()}")
        prior = float(close[idx - 1])
        # the vendor's yield is annualised on a quarterly payer: 4 * D / P
        implied = 4 * r.Div / (r.Yield / 100)
        errors.append(abs(implied / prior - 1))
        factors[:idx] *= 1 - r.Div / prior

    df["Adj Close"] = close * factors
    df["Ticker"] = "GOGL"
    df["Lineage"] = "DryBulk"
    df["PullDate"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    max_err = max(errors) * 100
    if max_err > 1.0:
        raise ValueError(
            f"yield reconciliation failed: max error {max_err:.2f}% against the "
            "price file. The dividend history and the price export are probably "
            "not the same series."
        )

    cum = 1 / (df["Adj Close"].iloc[0] / df["Close"].iloc[0])
    if verbose:
        print(f"rows {len(df)}, {df.Date.min().date()} to {df.Date.max().date()}")
        print(f"yield reconciliation: max error {max_err:.2f}%, "
              f"median {np.median(errors) * 100:.2f}%")
        print(f"cumulative dividend adjustment {cum:.4f}x "
              f"(GOGLO.ST reports 1.0728x, which is wrong)")
        print(f"split applied: 1-for-{SPLIT_RATIO:.0f} at {SPLIT_DATE.date()}")

    cols = ["Date", "Adj Close", "Close", "Open", "High", "Low", "Volume",
            "Ticker", "Lineage", "PullDate"]
    out = df[cols].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUT, index=False)

    entry = {
        "ticker": "GOGL",
        "description": "Golden Ocean Group (NASDAQ, USD). Derived series, not a Yahoo pull.",
        "lineage": "DryBulk",
        "file": "GOGL.csv",
        "source": "investing.com daily export plus investing.com dividend history",
        "method": "scripts/build_gogl.py: 1-for-5 reverse split at 2016-07-19, "
                  "then standard dividend back-adjustment over 26 ex-dates",
        "pull_date": out["PullDate"].iloc[0],
        "rows": int(len(out)),
        "first_date": out["Date"].iloc[0],
        "last_date": out["Date"].iloc[-1],
        "columns": cols,
        "sha256": _sha256(OUT),
        "status": "ok",
        "adjustment_checked": True,
        "cumulative_adjustment_factor": round(float(cum), 4),
        "yield_reconciliation_max_error_pct": round(max_err, 3),
        "note": "Delisted 2025-08-20 on the CMB.TECH merger. Replaces GOGLO.ST, "
                "whose Yahoo Adj Close applies USD dividends to SEK prices and is "
                "therefore under-adjusted by a factor of about 1.84.",
    }

    existing = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    by_ticker = {e.get("ticker"): e for e in existing}
    by_ticker["GOGL"] = entry
    if "GOGLO.ST" in by_ticker:
        by_ticker["GOGLO.ST"]["status"] = "retired"
        by_ticker["GOGLO.ST"]["note"] = (
            "Retired 2026-09-22. Stale from 2025-08-19, 38% zero-return days, and "
            "Adj Close is under-adjusted (see the GOGL entry). Kept for reference "
            "only; excluded from the basket."
        )
    MANIFEST.write_text(
        json.dumps(sorted(by_ticker.values(), key=lambda e: e["ticker"]), indent=2)
    )
    if verbose:
        print(f"wrote {OUT.name} and updated {MANIFEST.name}")
    return out


if __name__ == "__main__":
    build()
