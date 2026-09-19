"""Compare candidate tickers on data quality before committing to one.

Reports the share of zero-return days (a carried-forward quote on a day with no
trading) and median volume, overall and by year. A thin secondary listing can
look complete while being unusable: Golden Ocean's Stockholm line (GOGLO.ST)
runs 56-82% zero-return days across 2016-2019 with a median volume of zero.

Run from the repo root in an environment that can reach Yahoo:
    python scripts/check_ticker_quality.py GOGL GOGL.OL GOGLO.ST --start 2015-01-01
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def quality(ticker: str, start: str, end: str | None = None) -> pd.DataFrame | None:
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=False, actions=False, progress=False)
    if df is None or df.empty:
        print(f"{ticker}: no data returned")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    ret = price.pct_change()
    zero = ret.abs() < 1e-12

    print(f"\n{ticker}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  zero-return days: {100 * zero.mean():.1f}%   "
          f"median volume: {int(df['Volume'].median()):,}   "
          f"annualised vol: {100 * ret.std() * np.sqrt(252):.1f}%")

    by_year = pd.DataFrame({
        "days": price.groupby(df.index.year).size(),
        "zero_ret_pct": (100 * zero.groupby(df.index.year).mean()).round(1),
        "median_volume": df["Volume"].groupby(df.index.year).median().astype(int),
        "zero_volume_days": (df["Volume"] == 0).groupby(df.index.year).sum(),
    })
    print(by_year.to_string())
    return by_year


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tickers", nargs="+")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()
    for t in args.tickers:
        quality(t, args.start, args.end)
    print("\nPick the series with the lowest zero-return share. Anything above "
          "roughly 10% in a year should not carry a seasonality test.")


if __name__ == "__main__":
    main()
