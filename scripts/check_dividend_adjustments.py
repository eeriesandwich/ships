#!/usr/bin/env python3
"""
Compare every dividend with the step its Adj Close encodes.

check_corporate_actions.py tests a listed set of large events. That missed the
2020.OL defect being systematic, because it was found on the two largest
payouts only. This script measures all of them.

For each vendor dividend D with ex-session i:

    share   = D / Close[i-1]                      what was paid, as a share of price
    encoded = 1 - 1 / step                        what the adjustment encodes
    step    = (Adj/Close)[i+1] / (Adj/Close)[i-2] over the sessions around the
                                                  ex-date, so a vendor date one
                                                  session out still counts

encoded / share is about 1 when the adjustment is right and about 0.1 when the
vendor divided by an FX rate near 10. Reported per ticker as the median ratio
and the count of dividends below MISFIT_LOW.

  python check_dividend_adjustments.py              all loaded tickers
  python check_dividend_adjustments.py --self-test  prove it detects the defect
  python check_dividend_adjustments.py --root PATH  explicit data/equity

Dividends are fetched live, so amounts drift about 0.1% between days.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "data" / "equity"
FOLDERS = ("tanker", "drybulk")

MIN_SHARE = 0.005      # below this a payout hides in price rounding
MISFIT_LOW = 0.5       # encoded / share under this: under-adjusted
MISFIT_HIGH = 2.0      # over this: over-adjusted


def measure(px: pd.DataFrame, divs: pd.Series) -> pd.DataFrame:
    """One row per dividend that falls inside the file and clears MIN_SHARE."""
    px = px.sort_values("Date").reset_index(drop=True)
    ratio = (px["Adj Close"] / px["Close"]).to_numpy()
    close = px["Close"].to_numpy()
    rows = []
    for ex, amount in divs.items():
        pos = px.index[px["Date"] >= ex]
        if len(pos) == 0:
            continue
        i = int(pos[0])
        if i < 2 or i + 1 >= len(px):
            continue
        share = amount / close[i - 1]
        if share < MIN_SHARE:
            continue
        step = ratio[i + 1] / ratio[i - 2]
        encoded = 1 - 1 / step
        rows.append((px["Date"][i].date(), amount, share, encoded, encoded / share))
    return pd.DataFrame(rows, columns=["ex", "amount", "share", "encoded", "ratio"])


def summarise(m: pd.DataFrame) -> tuple[int, float, int, int]:
    if m.empty:
        return 0, float("nan"), 0, 0
    return (len(m), float(m["ratio"].median()),
            int((m["ratio"] < MISFIT_LOW).sum()),
            int((m["ratio"] > MISFIT_HIGH).sum()))


def load_divs(ticker: str) -> pd.Series:
    import yfinance as yf
    d = yf.Ticker(ticker).dividends
    if d is None or d.empty:
        return pd.Series(dtype=float)
    d = d.copy()
    d.index = pd.to_datetime(d.index, utc=True).tz_localize(None).normalize()
    return d[d > 0].sort_index()


def self_test() -> int:
    """Build a price file with known dividends, adjust it three ways, and
    confirm the ratio reads about 1, about 0.1 and about 3."""
    dates = pd.bdate_range("2024-01-01", periods=60)
    close = pd.Series(100.0, index=range(60))
    divs = pd.Series([2.0, 3.0, 2.5],
                     index=[dates[15], dates[30], dates[45]])

    def adjusted(scale: float) -> pd.Series:
        factor = pd.Series(1.0, index=range(60))
        for ex, a in divs.items():
            i = int(np.where(dates == ex)[0][0])
            factor.iloc[:i] *= 1 - scale * a / 100.0
        return close * factor

    ok = True
    for name, scale, lo, hi in [("correct adjustment", 1.0, 0.9, 1.1),
                                ("vendor divides by 10", 0.1, 0.05, 0.15),
                                ("vendor over-adjusts 3x", 3.0, 2.6, 3.4)]:
        px = pd.DataFrame({"Date": dates, "Close": close.to_numpy(),
                           "Adj Close": adjusted(scale).to_numpy()})
        m = measure(px, divs)
        med = float(m["ratio"].median())
        passed = len(m) == 3 and lo <= med <= hi
        ok = ok and passed
        print(f"  {'ok  ' if passed else 'BAD '} {name:26} median ratio {med:.3f}"
              f" (want {lo}-{hi})")
    tiny = measure(px.assign(**{"Adj Close": px["Close"]}),
                   pd.Series([0.1], index=[dates[20]]))
    passed = tiny.empty
    ok = ok and passed
    print(f"  {'ok  ' if passed else 'BAD '} payout below MIN_SHARE is skipped")
    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else ROOT
    print(f"{'ticker':10}{'lineage':14}{'divs':>6}{'median':>9}{'under':>7}{'over':>6}  verdict")
    worst = 0
    for folder in FOLDERS:
        for f in sorted((root / folder).glob("*.csv")):
            ticker = f.stem
            px = pd.read_csv(f)
            px["Date"] = pd.to_datetime(px["Date"], errors="coerce",
                                        utc=True).dt.tz_localize(None)
            px = px.dropna(subset=["Date", "Close", "Adj Close"])
            lineage = str(px["Lineage"].iloc[-1]) if "Lineage" in px else ""
            m = measure(px, load_divs(ticker))
            n, med, under, over = summarise(m)
            if n == 0:
                verdict = "no testable dividends"
            elif under > n * 0.25 or over > n * 0.25:
                verdict = "MISFIT"
                worst = 1
            else:
                verdict = "ok"
            print(f"{ticker:10}{lineage[:13]:14}{n:>6}{med:>9.2f}{under:>7}{over:>6}  {verdict}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
