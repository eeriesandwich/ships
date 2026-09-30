#!/usr/bin/env python3
"""
Prove the corporate-action diagnostic detects what it claims, using synthetic
files where the defect is present or absent by construction.

This tests the CHECKER, not the project's data. A checker that cannot fail is
indistinguishable from clean data, and that is the failure this catches.

Four cases:
  1. split, correctly adjusted      -> expect PASS
  2. split, NOT adjusted            -> expect FAIL
  3. distribution, adjusted         -> expect PASS
  4. distribution, NOT adjusted     -> expect FAIL

Synthetic files go to a temporary directory that is removed afterwards.
Nothing in data/ is read or written.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# The diagnostic sits beside this file, wherever the repo happens to live.
HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_corporate_actions.py"

EXPECTED = {
    "DHT 2012-07-17":     "PASS",   # split, adjusted
    "STNG 2019-01-18":    "FAIL",   # split, not adjusted
    "DSX 2011-01-19":     "FAIL",   # distribution, adjusted but far too small
    "DSX 2021-11-29":     "PASS",   # adjusted correctly, ex-date one session late
    "2020.OL 2026-04-28": "FAIL",   # not adjusted; ACTIONS date is the
                                #   vendor ex-date, the price falls on the 29th
}

rng = np.random.default_rng(0)


def series(start, n):
    dates = pd.bdate_range(start, periods=n)
    px = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return dates, px


def write(root, folder, ticker, dates, close, adj):
    pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Adj Close": adj,
        "Close": close,
        "Open": close, "High": close, "Low": close,
        "Volume": 100000,
        "Ticker": ticker,
        "Lineage": folder,
        "PullDate": "2026-09-28",
    }).to_csv(root / folder / f"{ticker}.csv", index=False)


def build(root):
    for sub in ("tanker", "drybulk"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    def at(dates, iso):
        return list(dates.strftime("%Y-%m-%d")).index(iso)

    # 1. DHT, 1-for-12 reverse split 2012-07-17, CORRECTLY adjusted.
    #    No jump at the split date: the vendor scaled the history.
    d, px = series("2012-06-01", 60)
    write(root, "tanker", "DHT", d, px, px)

    # 2. STNG, 1-for-10 reverse split 2019-01-18, NOT adjusted.
    #    Pre-split prices left at one tenth, so Close jumps 10x on the date.
    d, px = series("2018-12-03", 60)
    k = at(d, "2019-01-18")
    raw = px.copy()
    raw[:k] = raw[:k] / 10
    write(root, "tanker", "STNG", d, raw, raw)

    # 3. DSX carries BOTH distribution cases in one file, since a ticker has
    #    one series and the diagnostic checks it at two dates.
    #
    #    2011-01-19: 90% of value handed out, but the vendor applied a factor
    #                of 0.90 as though it had been 10%. An adjustment IS
    #                present, so a test that only asks "did the ratio step"
    #                passes it. The size check is what catches it.
    #    2021-11-30: 10% handed out and a factor of 0.90 applied, correct, but
    #                one session AFTER the date ACTIONS asks about. This is the
    #                real 2020.OL situation: the vendor's recorded ex-date and
    #                the session the market marked the price down are a day
    #                apart. The windowed test must absorb that and still pass.
    d, px = series("2010-12-01", 2900)
    k1, k2 = at(d, "2011-01-19"), at(d, "2021-11-30")   # note: 30th, not 29th

    close = px.copy()
    close[k1:] = close[k1:] * 0.10      # the real 90% distribution
    close[k2:] = close[k2:] * 0.90      # the real 10% distribution

    # Cumulative factor the vendor applied, walking back from the present.
    f_applied_2011, f_applied_2021 = 0.90, 0.90   # 2011 is the wrong one
    adj = close.copy()
    adj[:k2] = adj[:k2] * f_applied_2021
    adj[:k1] = adj[:k1] * f_applied_2011
    write(root, "drybulk", "DSX", d, close, adj)

    # 4. 2020.OL, special dividend 2026-04-29, NOT adjusted.
    #    80% of value paid out and Adj Close equals Close throughout, so the
    #    ratio runs flat straight through a collapse in price.
    d, px = series("2026-04-01", 60)
    k = at(d, "2026-04-29")
    close = px.copy()
    close[k:] = close[k:] * 0.20
    write(root, "drybulk", "2020.OL", d, close, close)


def verdicts(stdout):
    """Pull 'VERDICT TICKER DATE' off the diagnostic's table."""
    known = {"PASS", "FAIL", "WARN", "SKIP", "INCONCLUSIVE", "EXCLUDED", "NO"}
    out = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in known:
            verdict = "NO FILE" if parts[0] == "NO" else parts[0]
            offset = 1 if parts[0] == "NO" else 0
            if len(parts) >= 3 + offset:
                out[f"{parts[1 + offset]} {parts[2 + offset]}"] = verdict
    return out


def main():
    if not CHECKER.exists():
        print(f"cannot find {CHECKER.name} beside this file at {HERE}")
        return 2

    root = Path(tempfile.mkdtemp(prefix="ships-selftest-"))
    try:
        build(root)
        proc = subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root),
             "--windows", str(root / "no_such_windows.csv")],
            capture_output=True, text=True,
        )
        print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)

        got = verdicts(proc.stdout)

        print("=" * 62)
        ok = True
        for key, want in EXPECTED.items():
            actual = got.get(key, "MISSING")
            if actual != want:
                ok = False
            print(f"{'ok  ' if actual == want else 'BAD '} {key:22} "
                  f"expected {want:8} got {actual}")

        # Two of the four carry a real defect, so a working diagnostic must
        # exit non-zero. An exit of 0 here would mean it failed to fail.
        status_ok = proc.returncode == 1
        print(f"{'ok  ' if status_ok else 'BAD '} exit status           "
              f"expected 1        got {proc.returncode}")

        # Windows: with 2020.OL valid only to 2026-04-01, its 2026-04-28
        # special is out of scope. It must be EXCLUDED rather than FAIL, and
        # the other three verdicts must not move.
        wfile = root / "windows.csv"
        wfile.write_text("Ticker,ValidFrom,ValidTo,Reason\n"
                         "2020.OL,,2026-04-01,test window\n")
        proc2 = subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(root),
             "--windows", str(wfile)], capture_output=True, text=True)
        got2 = verdicts(proc2.stdout)
        for key, want in [("2020.OL 2026-04-28", "EXCLUDED"),
                          ("STNG 2019-01-18", "FAIL"),
                          ("DHT 2012-07-17", "PASS")]:
            actual = got2.get(key, "MISSING")
            good = actual == want
            ok = ok and good
            print(f"{'ok  ' if good else 'BAD '} windowed {key:22} "
                  f"expected {want:8} got {actual}")

        print()
        if ok and status_ok:
            print("SELF-TEST PASSED")
            return 0
        print("SELF-TEST FAILED. Do not trust the diagnostic's verdict on the "
              "real files until this passes.")
        return 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
