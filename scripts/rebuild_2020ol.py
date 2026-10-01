#!/usr/bin/env python3
"""
Rebuild the dividend adjustment for 2020.OL.

WHY

Yahoo under-adjusts this series by a roughly constant factor of about 9.4.
Two checked events, two years apart, show the same ratio:

    2024-04-15   payout 10.1-11.3% of price, adjustment encodes  1.2%
    2026-04-28   payout ~97% of price,       adjustment encodes 10.5%

A constant factor is a unit error rather than a mistake, and it matches
USD/NOK over the period. The same class of defect was found on GOGLO.ST,
where USD dividend amounts were applied to SEK prices.

The vendor's dividend AMOUNTS are correct and in NOK: it reports 132.2431 for
the 2026 special, against an announced NOK 129.5. Only the adjustment applied
to the price history is wrong. So the fix needs no exchange rate.

WHAT IT DOES

Recomputes Adj Close from Close and the dividend series, using the standard
back-adjustment: at each ex-date, every earlier close is scaled by

    f = 1 - D / P_cum

where P_cum is the close on the session before the ex-date. Adjusted returns
then equal total returns, which is what the rest of the project assumes of
every Adj Close column.

RUN IT AFTER pull_prices, the same way build_gogl.py runs after it. If you
re-pull and forget, check_corporate_actions.py will fail on this ticker again,
which is the gate working rather than a trap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

TICKER = "2020.OL"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TARGET = REPO / "data" / "equity" / "drybulk"
MANIFEST = TARGET / "_manifest.json"
WINDOWS = REPO / "data" / "series_windows.csv"
OVERRIDES = REPO / "data" / "overrides" / "2020.OL_dividend_dates.csv"
SNAPSHOT = REPO / "data" / "dividends" / "2020.OL.csv"

# A dividend cannot exceed the price it is paid out of. Anything above this
# share of the cum price is a data error rather than a large payout.
MAX_PAYOUT_SHARE = 0.995


def find_file() -> Path:
    for name in (f"{TICKER}.csv", f"{TICKER.replace('.', '_')}.csv"):
        p = TARGET / name
        if p.exists():
            return p
    sys.exit(f"no CSV for {TICKER} in {TARGET}")


def valid_through() -> pd.Timestamp | None:
    """Last valid date for this ticker: --through YYYY-MM-DD, else the window
    file. Dividends after it belong to a different business and are ignored."""
    for i, a in enumerate(sys.argv):
        if a == "--through" and i + 1 < len(sys.argv):
            return pd.Timestamp(sys.argv[i + 1])
    if WINDOWS.exists():
        w = pd.read_csv(WINDOWS)
        w = w[(w["Ticker"] == TICKER) & w["ValidTo"].notna()]
        if len(w):
            return pd.Timestamp(w["ValidTo"].iloc[0])
    return None


def apply_date_overrides(divs: pd.Series) -> pd.Series:
    """Move dividends the vendor dates wrongly onto the session that carries
    the price fall. Each override is a row with a reason in OVERRIDES."""
    if not OVERRIDES.exists():
        return divs
    ov = pd.read_csv(OVERRIDES, parse_dates=["yahoo_date", "ex_date"])
    idx = divs.index.to_series()
    for _, r in ov.iterrows():
        hit = idx == r["yahoo_date"]
        if not hit.any():
            sys.exit(f"override for {r['yahoo_date']:%Y-%m-%d} matches no "
                     f"vendor dividend; the vendor data may have changed")
        print(f"  date override {r['yahoo_date']:%Y-%m-%d} -> "
              f"{r['ex_date']:%Y-%m-%d}: {r['reason']}")
        idx[hit] = r["ex_date"]
    out = divs.copy()
    out.index = pd.DatetimeIndex(idx)
    return out.sort_index()


def fetch_live() -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is required to fetch the dividend series")
    divs = yf.Ticker(TICKER).dividends
    if divs is None or divs.empty:
        sys.exit(f"the vendor reports no dividends for {TICKER}")
    divs = divs.copy()
    divs.index = pd.to_datetime(divs.index, utc=True).tz_localize(None).normalize()
    return divs[divs > 0].sort_index()


def freeze(divs: pd.Series) -> None:
    """Write the raw vendor series, as dated and as valued by the vendor.

    Yahoo converts these amounts at a live exchange rate, so they move about
    0.1 percent from one day to the next. Reading the snapshot makes every
    later run reproduce the same adjustment. Date overrides and the validity
    window are applied on read, not here, so the file stays the vendor's own
    record and the overrides stay reviewable.
    """
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Date": divs.index.strftime("%Y-%m-%d"),
                  "Dividend": divs.to_numpy(), "Currency": "NOK",
                  "FetchDate": pd.Timestamp.today().strftime("%Y-%m-%d")}
                 ).to_csv(SNAPSHOT, index=False)
    print(f"  froze {len(divs)} vendor dividends to {SNAPSHOT.relative_to(REPO)}")


def fetch_dividends() -> pd.Series:
    if "--refresh" in sys.argv or not SNAPSHOT.exists():
        divs = fetch_live()
        if "--dry-run" in sys.argv:
            print("  dividends fetched live; --dry-run so no snapshot written")
        else:
            freeze(divs)
    else:
        snap = pd.read_csv(SNAPSHOT, parse_dates=["Date"])
        divs = pd.Series(snap["Dividend"].to_numpy(), index=pd.DatetimeIndex(snap["Date"]))
        print(f"  dividends from snapshot {SNAPSHOT.relative_to(REPO)} "
              f"(fetched {snap['FetchDate'].iloc[0]}); --refresh to re-fetch")
    divs = apply_date_overrides(divs)
    end = valid_through()
    if end is not None:
        dropped = int((divs.index > end).sum())
        print(f"  valid through {end:%Y-%m-%d}: ignoring {dropped} later dividends")
        divs = divs[divs.index <= end]
    return divs


def back_adjust(px: pd.DataFrame, divs: pd.Series) -> tuple[pd.Series, list]:
    """Standard back-adjustment, walking from the present into the past.

    The factor at time t is the product of (1 - D/P_cum) over every ex-date
    after t, so the most recent prices are unadjusted and the oldest carry the
    whole dividend history.
    """
    close = px["Close"].to_numpy(dtype=float).copy()
    dates = px["Date"].to_numpy()
    factor = pd.Series(1.0, index=px.index)
    applied = []

    for ex_date, amount in divs.items():
        # First session on or after the ex-date.
        pos = px.index[px["Date"] >= ex_date]
        if len(pos) == 0:
            continue                      # dividend after the file ends
        i = int(pos[0])
        if i == 0:
            continue                      # dividend before the file starts
        p_cum, p_ex = close[i - 1], close[i]
        if not p_cum or p_cum <= 0 or not p_ex or p_ex <= 0:
            continue
        share = amount / p_cum          # reported against the cum price
        if share >= MAX_PAYOUT_SHARE:
            sys.exit(f"payout {amount:,.4f} on {ex_date:%Y-%m-%d} is {share:.1%} "
                     f"of the {p_cum:,.4f} cum close, which is not a dividend. "
                     f"Check the currency before trusting this rebuild.")
        f = p_ex / (p_ex + amount)      # total-return consistent
        factor.iloc[:i] *= f
        applied.append((pd.Timestamp(dates[i]).date(), float(amount),
                        float(p_cum), share, f))

    return px["Close"] * factor, applied


def self_test() -> int:
    """Check the back-adjustment on cases where the answer is known.

    Run with --self-test. Touches no project data.
    """
    cases = [
        ("price falls by exactly the dividend",
         [100.0, 90.0, 90.0], {"2024-01-02": 10.0}, [0.0, 0.0]),
        ("two dividends compound",
         [100.0, 90.0, 100.0, 90.0],
         {"2024-01-02": 10.0, "2024-01-04": 10.0}, [0.0, 1 / 9, 0.0]),
        ("independent price move on the ex-date",
         [100.0, 99.0, 90.0], {"2024-01-03": 9.9},
         [-0.01, (90 + 9.9) / 99 - 1]),
        ("payout of nearly the entire price, as in April 2026",
         [136.0, 3.8, 3.9], {"2024-01-02": 132.24},
         [(3.8 + 132.24) / 136 - 1, 3.9 / 3.8 - 1]),
    ]
    ok = True
    for name, closes, divs, want in cases:
        px = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=len(closes)),
                           "Close": closes})
        d = pd.Series(list(divs.values()), index=pd.to_datetime(list(divs.keys())))
        adj, _ = back_adjust(px, d)
        got = (adj / adj.shift(1) - 1).tolist()[1:]
        passed = all(abs(g - w) < 1e-9 for g, w in zip(got, want))
        ok = ok and passed
        print(f"  {'ok  ' if passed else 'BAD '} {name}")
        print(f"         adjusted returns {[round(v, 7) for v in got]}")
        print(f"         total returns    {[round(v, 7) for v in want]}")
    print()
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    dry = "--dry-run" in sys.argv

    path = find_file()
    px = pd.read_csv(path)
    px["Date"] = pd.to_datetime(px["Date"], errors="coerce",
                                utc=True).dt.tz_localize(None)
    px = px.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    before = px["Adj Close"].iloc[0] / px["Close"].iloc[0]
    # A file that has already been rebuilt no longer shows the vendor's factor.
    # Once the manifest records it, that record is the vendor's figure.
    if MANIFEST.exists():
        for e in json.loads(MANIFEST.read_text(encoding="utf-8")):
            if e.get("ticker") == TICKER and e.get("cumulative_adjustment_factor_vendor"):
                before = 1 / e["cumulative_adjustment_factor_vendor"]
    divs = fetch_dividends()
    rebuilt, applied = back_adjust(px, divs)

    # ---- validation, before anything is written ----
    checks = []

    tail = abs(rebuilt.iloc[-1] / px["Close"].iloc[-1] - 1.0)
    checks.append(("most recent price is unadjusted", tail < 1e-9,
                   f"ratio {rebuilt.iloc[-1] / px['Close'].iloc[-1]:.9f}"))

    checks.append(("every adjusted price is positive", (rebuilt > 0).all(),
                   f"min {rebuilt.min():,.6f}"))

    ratio = rebuilt / px["Close"]
    checks.append(("adjustment never increases with time",
                   bool((ratio.diff().dropna() >= -1e-12).all()),
                   "monotonic non-decreasing"))

    after = ratio.iloc[0]
    checks.append(("rebuilt adjustment is larger than the vendor's",
                   after < before,
                   f"first-row ratio {before:.6f} -> {after:.6f}, "
                   f"cumulative {1 / before:,.4f}x -> {1 / after:,.4f}x"))

    print(f"{TICKER}: {len(px):,} rows, {len(applied)} dividends applied "
          f"of {len(divs)} reported\n")
    print("  validation")
    ok = True
    for name, passed, detail in checks:
        print(f"    {'ok  ' if passed else 'FAIL'} {name:44} {detail}")
        ok = ok and passed
    if not ok:
        print("\nvalidation failed, nothing written")
        return 1

    shares = sorted(r[3] for r in applied)
    n = len(shares)
    print("\n  payout as a share of the cum price, across all "
          f"{n} applied dividends")
    print(f"    median {shares[n // 2]:.2%}   "
          f"upper quartile {shares[int(n * 0.75)]:.2%}   "
          f"largest {shares[-1]:.2%}")
    print(f"    above 1%: {sum(1 for x in shares if x > 0.01)}    "
          f"above 3%: {sum(1 for x in shares if x > 0.03)}    "
          f"above 10%: {sum(1 for x in shares if x > 0.10)}")

    print("\n  five largest payouts")
    for d, amt, p_cum, share, f in sorted(applied, key=lambda r: -r[3])[:5]:
        print(f"    {d}  {amt:>10,.4f} on {p_cum:>10,.4f} = {share:6.1%}, "
              f"factor {f:.6f}")

    # The column set must not change. 2020.OL.csv is first alphabetically in
    # drybulk/, so it is probably the file Power Query took as its sample, and
    # an extra column there makes the folder expand demand that column from
    # every other file. The vendor's original values live in git history and
    # its cumulative factor is recorded in the manifest below.
    if dry:
        print("\n  --dry-run: nothing written")
        return 0

    px["Adj Close"] = rebuilt
    if "--out" in sys.argv:
        # Review copy: the project file and manifest are left untouched.
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        px.to_csv(out, index=False)
        print(f"\n  --out: wrote {out}; project file and manifest untouched")
        return 0
    px.to_csv(path, index=False)
    print(f"\n  wrote {path}, column set unchanged")

    # ---- manifest provenance ----
    if MANIFEST.exists():
        entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for e in entries:
            if e.get("ticker") == TICKER:
                e["adjustment_rebuilt"] = True
                e["adjustment_rebuilt_by"] = "scripts/rebuild_2020ol.py"
                e["adjustment_rebuilt_reason"] = (
                    "Vendor under-adjusted dividends by a roughly constant factor "
                    "of about 9 on 72 of 73 payouts, consistent with USD amounts "
                    "applied to NOK prices. Rebuilt from Close and a frozen "
                    "snapshot of the vendor's dividend series in "
                    "data/dividends/2020.OL.csv. Yahoo converts those NOK amounts "
                    "at a live rate, so they drift about 0.1 percent between "
                    "fetches; the snapshot makes re-runs exact. Dividends after "
                    "the series window and one vendor ex-date one session late "
                    "are handled through data/series_windows.csv and "
                    "data/overrides/.")
                e["dividend_snapshot"] = str(SNAPSHOT.relative_to(REPO)).replace("\\", "/")
                e["dividends_applied"] = len(applied)
                e["cumulative_adjustment_factor_vendor"] = round(1 / before, 4)
                e["cumulative_adjustment_factor"] = round(1 / after, 4)
                e["columns"] = list(px.columns)
        MANIFEST.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print(f"  recorded in {MANIFEST.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
