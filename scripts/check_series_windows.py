#!/usr/bin/env python3
"""
Audit data/series_windows.csv against the price files.

The model's window filter treats a missing row and an unbounded row the same
way: both pass everything through. So a ticker whose data stops early, with no
row in series_windows.csv, is an unchecked assumption rather than a decision.
This script makes each ticker one of:

  RUNS TO PULL    last date is within STALE_DAYS of the file's PullDate
  WINDOWED        a row with a ValidTo exists and the data ends on or after it
                  (data past ValidTo is the point: the model's filter cuts it)
  ACKNOWLEDGED    a row with no ValidFrom or ValidTo records, in its Reason,
                  that the vendor simply stopped and nothing is wrong
  UNEXPLAINED     data stops early and nothing in the file says why
  MISMATCH        a ValidTo is later than the data ends by more than
                  STALE_DAYS, so the window claims rows the file does not hold

Exit status is 1 if anything is UNEXPLAINED or MISMATCH.

  python check_series_windows.py                  all loaded tickers
  python check_series_windows.py --self-test      prove it detects each case
  python check_series_windows.py --root PATH      explicit data/equity
  python check_series_windows.py --windows PATH   explicit series_windows.csv
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "data" / "equity"
FOLDERS = ("tanker", "drybulk")

STALE_DAYS = 7      # a vendor lag of a few sessions is normal, a month is not


def audit(root: Path, windows: pd.DataFrame) -> list[tuple]:
    rows = []
    for folder in FOLDERS:
        for f in sorted((root / folder).glob("*.csv")):
            ticker = f.stem
            px = pd.read_csv(f, usecols=["Date", "PullDate"])
            px["Date"] = pd.to_datetime(px["Date"], errors="coerce")
            last = px["Date"].max()
            pulled = pd.to_datetime(px["PullDate"].dropna().iloc[-1])
            gap = (pulled - last).days
            w = windows[windows["Ticker"] == ticker]
            bounded = w[w["ValidTo"].notna()]
            if len(bounded):
                end = bounded["ValidTo"].iloc[0]
                if (end - last).days > STALE_DAYS:
                    verdict, why = "MISMATCH", f"ValidTo {end:%Y-%m-%d} is {(end - last).days}d after the data ends"
                else:
                    cut = int((px["Date"] > end).sum())
                    verdict = "WINDOWED"
                    why = f"ValidTo {end:%Y-%m-%d}, cuts {cut} rows; {bounded['Reason'].iloc[0][:50]}"
            elif len(w):
                verdict, why = "ACKNOWLEDGED", w["Reason"].iloc[0][:70]
            elif gap <= STALE_DAYS:
                verdict, why = "RUNS TO PULL", f"{gap}d before pull"
            else:
                verdict, why = "UNEXPLAINED", f"data ends {gap}d before pull and no row says why"
            rows.append((ticker, folder, last.date(), pulled.date(), gap, verdict, why))
    return rows


def load_windows(path: Path) -> pd.DataFrame:
    w = pd.read_csv(path)
    for c in ("ValidFrom", "ValidTo"):
        w[c] = pd.to_datetime(w[c], errors="coerce")
    return w


def report(rows: list[tuple]) -> int:
    print(f"{'ticker':10}{'folder':9}{'last date':12}{'pulled':12}{'gap':>5}  {'verdict':13} why")
    bad = 0
    for t, fo, last, pulled, gap, v, why in rows:
        print(f"{t:10}{fo:9}{last!s:12}{pulled!s:12}{gap:>4}d  {v:13} {why}")
        bad += v in ("UNEXPLAINED", "MISMATCH")
    print(f"\n{len(rows)} tickers, {bad} needing attention")
    return 1 if bad else 0


def self_test() -> int:
    """Six planted tickers covering every verdict, checked against known answers."""
    root = Path(tempfile.mkdtemp(prefix="windows-selftest-"))
    (root / "tanker").mkdir()
    (root / "drybulk").mkdir()
    pull = "2026-09-20"

    def write(folder, ticker, last):
        d = pd.bdate_range(end=last, periods=10)
        pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "PullDate": pull}
                     ).to_csv(root / folder / f"{ticker}.csv", index=False)

    write("tanker", "OK", "2026-09-18")            # runs to pull, no row
    write("tanker", "STOPPED", "2025-03-14")       # stops early, no row
    write("tanker", "CUT", "2025-08-19")           # window matches the data
    write("drybulk", "ACK", "2024-06-28")          # stops early, acknowledged
    write("drybulk", "DRIFT", "2025-03-14")        # window claims rows the data lacks
    write("drybulk", "CUTS", "2026-09-18")         # data runs past the window; must not flag
    win = root / "w.csv"
    win.write_text(
        "Ticker,ValidFrom,ValidTo,Reason\n"
        "CUT,,2025-08-19,merger completed\n"
        "ACK,,,vendor stopped quoting; nothing wrong\n"
        "DRIFT,,2025-09-01,window later than the data ends\n"
        "CUTS,,2025-09-25,business changed\n")
    got = {r[0]: r[5] for r in audit(root, load_windows(win))}
    want = {"OK": "RUNS TO PULL", "STOPPED": "UNEXPLAINED", "CUT": "WINDOWED",
            "ACK": "ACKNOWLEDGED", "DRIFT": "MISMATCH", "CUTS": "WINDOWED"}
    ok = True
    for t, w in want.items():
        good = got.get(t) == w
        ok = ok and good
        print(f"  {'ok  ' if good else 'BAD '} {t:8} expected {w:13} got {got.get(t)}")
    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else ROOT
    wpath = (Path(sys.argv[sys.argv.index("--windows") + 1]) if "--windows" in sys.argv
             else root.parent / "series_windows.csv")
    return report(audit(root, load_windows(wpath)))


if __name__ == "__main__":
    raise SystemExit(main())
