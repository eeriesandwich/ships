#!/usr/bin/env python3
"""
Check the equity price files for corporate actions the vendor failed to apply.

WHY TWO DIFFERENT TESTS

Under yfinance with auto_adjust=False, Yahoo returns both Close and Adj Close.
Close is ALREADY split-adjusted. Adj Close is split-adjusted AND
distribution-adjusted. So the two defects need different tests:

  A missing SPLIT appears identically in both columns, as a raw jump in Close of
  roughly the split factor on the effective date. Test Close.

  A missing DISTRIBUTION (spin-off, large special dividend) leaves Adj Close and
  Close moving together, so the ratio Adj Close / Close stays FLAT across the
  ex-date while the price falls by the distributed value. Test the ratio step.

A percentage-return threshold cannot separate these and cannot see a 10 percent
spin-off at all, which is why this replaces the earlier version.

USAGE

  python check_corporate_actions.py                 offline, file-based tests
  python check_corporate_actions.py --online        also cross-check against
                                                    Yahoo's own actions series
  python check_corporate_actions.py --root PATH     explicit data/equity path

Exit status is 0 when every test passes or is inconclusive for a benign reason,
and 1 when any test FAILS, so this can gate a build step.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Known corporate actions.
#
# kind "split":  factor is the price multiplier applied to pre-event prices by
#                a correct adjustment. A 1-for-12 reverse split has factor 12.
# kind "distribution": expected_drop is the approximate fraction of share value
#                handed out, or None when it is not known in advance.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    ticker: str
    folder: str
    date: str
    kind: str                      # "split" or "distribution"
    factor: float | None = None    # split only
    expected_drop: float | None = None   # distribution only, as a fraction
    amount: float | None = None    # distribution only: the payout per share,
                                   # in the file's own currency, from the vendor
    note: str = ""


ACTIONS = [
    # Reverse splits. Detected on Close.
    Action("DHT",  "tanker",  "2012-07-17", "split", factor=12,
           note="1-for-12 reverse split"),
    Action("STNG", "tanker",  "2019-01-18", "split", factor=10,
           note="1-for-10 reverse split"),
    Action("TNK",  "tanker",  "2019-11-25", "split", factor=8,
           note="1-for-8 reverse split"),
    Action("SBLK", "drybulk", "2008-11-25", "split", factor=1 / 1.026,
           note="2.6 percent stock dividend"),
    Action("SBLK", "drybulk", "2012-10-15", "split", factor=15,
           note="1-for-15 reverse split"),
    Action("SBLK", "drybulk", "2016-06-20", "split", factor=5,
           note="1-for-5 reverse split"),
    Action("GNK",  "drybulk", "2016-07-08", "split", factor=10,
           note="1-for-10 reverse split"),
    Action("GOGL", "drybulk", "2016-08-01", "split", factor=5,
           note="1-for-5 reverse split, reconstructed series"),

    # Distributions. Detected on the Adj Close / Close ratio step.
    Action("DSX",     "drybulk", "2011-01-19", "distribution",
           expected_drop=None,
           note="Diana Containerships spin-off, 0.032542 per share"),
    Action("DSX",     "drybulk", "2021-11-29", "distribution",
           expected_drop=0.10,
           note="OceanPal spin-off, 1 per 10 shares"),
    Action("2020.OL", "drybulk", "2026-04-28", "distribution",
           expected_drop=None,
           amount=132.2431, note="NOK 129.5 special dividend after the fleet sale"),

    # --- surfaced by the vendor scan, not by desk research ---
    Action("FRO",  "tanker",  "2016-02-03", "split", factor=5,
           note="1-for-5 reverse split"),
    Action("RIO",  "factors", "2010-04-30", "split", factor=1 / 4,
           note="4-for-1 forward split"),
    Action("VALE", "factors", "2004-09-07", "split", factor=1 / 3,
           note="3-for-1 forward split, before the file starts"),
    Action("VALE", "factors", "2006-06-07", "split", factor=1 / 2,
           note="2-for-1 forward split"),
    Action("VALE", "factors", "2007-09-13", "split", factor=1 / 2,
           note="2-for-1 forward split"),
    Action("EFA",  "factors", "2005-06-09", "split", factor=1 / 3,
           note="3-for-1 forward split, EFA is not a model control"),

    # Payouts worth more than a tenth of the share price on the day.
    Action("CMBT.BR", "tanker",  "2009-04-28", "distribution", amount=1.6000, note="14.1 percent payout"),
    Action("CMBT.BR", "tanker",  "2024-05-21", "distribution", amount=4.0253, note="21.0 percent payout"),
    Action("FRO",     "tanker",  "2005-02-03", "distribution", amount=29.2750, note="11.9 percent payout"),
    Action("FRO",     "tanker",  "2007-03-06", "distribution", amount=23.7200, note="15.4 percent payout"),
    Action("TNK",     "tanker",  "2008-12-01", "distribution", amount=8.5600, note="13.3 percent payout"),
    Action("DSX",     "drybulk", "2008-11-24", "distribution", amount=0.8029, note="12.1 percent payout"),
    Action("SB",      "drybulk", "2008-11-19", "distribution", amount=0.4750, note="12.9 percent payout"),
    Action("2020.OL", "drybulk", "2024-04-15", "distribution", amount=17.1533, note="11.3 percent payout"),
    Action("RIO",     "factors", "2009-07-08", "distribution", amount=5.7080, note="18.2 percent payout"),
    Action("VALE",    "factors", "2021-09-23", "distribution", amount=1.5590, note="10.5 percent payout"),
]

# How close an observed ratio must sit to the expected one to count as a match.
SPLIT_TOLERANCE = 0.15          # relative
RATIO_STEP_FLOOR = 0.005        # an Adj/Close step below this is "flat"
PRICE_FALL_FLOOR = 0.04         # a fall below this is not worth calling a defect
SIZE_TOLERANCE = 2.0            # the two distribution estimates may differ by this much
MIN_SPLIT_FACTOR = 1.5          # below this a split is indistinguishable from a normal day
WINDOW_PAD = 3                  # sessions either side, to absorb an ex-date off by a day or two
LARGE_DIVIDEND = 0.10           # a payout above this share of the price is worth checking
AMOUNT_TOLERANCE = 1.4          # slack on the known-payout test, for rounding and re-pulls
GROSS_MISMATCH = 5.0            # beyond this, market movement cannot explain the gap
WINDOW = 5                      # trading days of context either side
MAX_DATE_OFFSET = 7             # calendar days; beyond this the match is unsafe


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def resolve_root(explicit: str | None) -> Path:
    """Resolve data/equity, anchored on this file rather than the CWD."""
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        here = Path(__file__).resolve()
        candidates = [
            here.parent.parent / "data" / "equity",   # scripts/ -> repo root
            here.parent / "data" / "equity",          # repo root
            Path.cwd() / "data" / "equity",
            Path.cwd().parent / "data" / "equity",    # notebooks/
        ]
        root = next((c for c in candidates if c.is_dir()), candidates[0])
    if not root.is_dir():
        sys.exit(f"data/equity not found. Tried {root}. Pass --root explicitly.")
    return root


def load(root: Path, ticker: str, folder: str) -> pd.DataFrame | None:
    """Load one price file, or None if it is missing.

    Filenames are tried both with the dot intact and with it replaced by an
    underscore, because the pull script has used both conventions.
    """
    bases = [root / folder, root.parent / folder]   # equity/<x>, or data/factors
    for base in bases:
        for name in (f"{ticker}.csv", f"{ticker.replace('.', '_')}.csv"):
            path = base / name
            if path.exists():
                break
        else:
            continue
        break
    else:
        return None

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in ("Close", "Adj Close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # A non-positive price is not a price. Blank it rather than let it
            # produce an infinite ratio that wins every comparison.
            df.loc[df[col] <= 0, col] = pd.NA

    df.attrs["path"] = path
    return df


def locate(df: pd.DataFrame, target: pd.Timestamp) -> tuple[int | None, int]:
    """Index of the first session on or after target, and its offset in days."""
    on_or_after = df.index[df["Date"] >= target]
    if len(on_or_after) == 0:
        return None, 0
    idx = int(on_or_after[0])
    return idx, (df.at[idx, "Date"] - target).days


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def _nearby_steps(df: pd.DataFrame, idx: int, span: int = 10) -> str:
    """Where in the window the adjustment ratio actually moved, if anywhere.

    An ex-date taken from a completion or record date is often a day or two
    out. This points at the real one instead of leaving the reader guessing.
    """
    ratio = df["Adj Close"] / df["Close"]
    lo, hi = max(1, idx - span), min(len(df), idx + span + 1)
    steps = (ratio.iloc[lo:hi] / ratio.shift(1).iloc[lo:hi] - 1.0).dropna()
    steps = steps[steps.abs() >= RATIO_STEP_FLOOR]
    if steps.empty:
        return f"No ratio step anywhere within {span} sessions either side."
    hits = ", ".join(f"{df.at[j, 'Date']:%Y-%m-%d} {v:+.2%}"
                     for j, v in steps.items())
    return f"The ratio did step nearby: {hits}."


def test_split(df: pd.DataFrame, a: Action) -> tuple[str, str]:
    """A missing reverse split shows as a raw Close jump of about the factor."""
    target = pd.Timestamp(a.date)
    idx, offset = locate(df, target)
    if idx is None:
        return "SKIP", f"{a.date} is after the last session in the file"
    if idx == 0:
        return "SKIP", "event falls on the first session, no prior close"
    if offset > MAX_DATE_OFFSET:
        return "SKIP", (f"nearest session is {df.at[idx, 'Date']:%Y-%m-%d}, "
                        f"{offset} days after the event")

    lo_f, hi_f = 1 / MIN_SPLIT_FACTOR, MIN_SPLIT_FACTOR
    if lo_f < a.factor < hi_f:
        return "SKIP", (f"split factor {a.factor:.4g} is within [{lo_f:.2f}, "
                        f"{hi_f:.2f}], so a correct and a missing adjustment "
                        f"look the same as an ordinary day's move. Not testable "
                        f"this way.")

    prev, curr = df.at[idx - 1, "Close"], df.at[idx, "Close"]
    if pd.isna(prev) or pd.isna(curr):
        return "SKIP", "Close missing on the event date or the prior session"

    ratio = curr / prev
    rel = abs(ratio - a.factor) / a.factor

    detail = (f"Close {prev:,.4f} -> {curr:,.4f}, ratio {ratio:,.3f}, "
              f"split factor {a.factor:g}")
    if rel <= SPLIT_TOLERANCE:
        return "FAIL", f"{detail}. The split was NOT applied; multiply earlier prices by {a.factor:g}."
    if ratio > 1 + SPLIT_TOLERANCE:
        return "WARN", f"{detail}. A large rise that does not match the factor. Investigate."
    return "PASS", f"{detail}. Ordinary move, so the split was applied."


def test_distribution(df: pd.DataFrame, a: Action) -> tuple[str, str]:
    """A missing distribution leaves Adj Close / Close flat across the ex-date.

    A correct adjustment steps the ratio down by about (1 - D/P) at the ex-date,
    because every prior price is scaled while the current one is not.
    """
    if "Adj Close" not in df.columns:
        return "SKIP", "no Adj Close column, so the ratio test cannot run"

    target = pd.Timestamp(a.date)
    idx, offset = locate(df, target)
    if idx is None:
        return "SKIP", f"{a.date} is after the last session in the file"
    if idx == 0:
        return "SKIP", "event falls on the first session, no prior session"
    if offset > MAX_DATE_OFFSET:
        return "SKIP", (f"nearest session is {df.at[idx, 'Date']:%Y-%m-%d}, "
                        f"{offset} days after the event")

    ratio = df["Adj Close"] / df["Close"]

    # Measure across a window rather than a single boundary. A vendor's
    # recorded ex-date and the session the market actually marked the price
    # down on are often a day apart, and comparing one boundary can put the
    # adjustment on one side and the price move on the other.
    lo = max(0, idx - 1 - WINDOW_PAD)
    hi = min(len(df) - 1, idx + WINDOW_PAD)
    r_prev, r_curr = ratio.iat[lo], ratio.iat[hi]
    p_prev, p_curr = df.at[lo, "Close"], df.at[hi, "Close"]
    if any(pd.isna(v) for v in (r_prev, r_curr, p_prev, p_curr)):
        return "SKIP", "price or ratio missing around the ex-date"
    span = f"{df.at[lo, 'Date']:%Y-%m-%d} to {df.at[hi, 'Date']:%Y-%m-%d}"

    step = (r_curr / r_prev) - 1.0        # positive when the vendor adjusted
    move = (p_curr / p_prev) - 1.0

    # The two independent estimates of the distribution, as a share of the
    # cum price. They must agree, or the adjustment is the wrong SIZE even
    # when it is present.
    #
    #   f = ratio_prev / ratio_curr is the factor applied to prior prices.
    #   A correct adjustment sets f = Close_t / (Close_t + D), so the
    #   distribution it encodes is  1 - f  of the cum price.
    #
    #   The price fall on the ex-date is the other estimate: -move.
    #
    # Market movement means they never match exactly, so only a gross
    # disagreement is reported.
    f = r_prev / r_curr
    by_adjustment = 1.0 - f
    by_price_fall = -move

    detail = (f"over {span}: Adj/Close {r_prev:.5f} -> {r_curr:.5f} "
              f"(step {step:+.3%}), Close move {move:+.2%}")
    if a.expected_drop is not None:
        detail += f", expected distribution about {a.expected_drop:.0%}"

    applied = abs(step) >= RATIO_STEP_FLOOR
    fell = by_price_fall >= PRICE_FALL_FLOOR

    # When the payout is known, test the adjustment against it directly. This
    # is exact, and unlike the price fall it carries no market movement. A
    # window in October 2008 or March 2020 contains daily moves many times the
    # size of any dividend, so the price-fall estimate is only usable when
    # nothing better exists.
    if a.amount is not None:
        c_ex = df.at[idx, "Close"]
        if not pd.isna(c_ex) and c_ex > 0:
            # The ex-date close may be recorded cum or ex, which brackets the
            # true figure. Accept the adjustment if it falls in that range,
            # widened by the tolerance.
            lo_exp = a.amount / (c_ex + a.amount)      # c_ex is the ex close
            hi_exp = a.amount / c_ex                    # c_ex is the cum close
            band_lo = lo_exp / AMOUNT_TOLERANCE
            band_hi = hi_exp * AMOUNT_TOLERANCE
            sized = (f" Payout {a.amount:,.4f} on a {c_ex:,.4f} close is "
                     f"{lo_exp:.1%} to {hi_exp:.1%} of it; the adjustment "
                     f"encodes {by_adjustment:.1%}.")
            if not applied:
                return "FAIL", (f"{detail}.{sized} No adjustment at all.")
            if band_lo <= by_adjustment <= band_hi:
                return "PASS", f"{detail}.{sized} The adjustment matches the payout."
            return "FAIL", (f"{detail}.{sized} The adjustment does not match the "
                            f"payout.")

    if applied and fell:
        sizes = (f" Distribution implied by the adjustment {by_adjustment:.1%}, "
                 f"by the price fall {by_price_fall:.1%}.")
        # A ratio of the two beyond SIZE_TOLERANCE either way is a real
        # mismatch rather than a market move.
        bigger = max(by_adjustment, by_price_fall)
        smaller = min(by_adjustment, by_price_fall)
        if smaller > 0:
            gap = bigger / smaller
            if gap > GROSS_MISMATCH:
                return "FAIL", (f"{detail}.{sizes} The adjustment is off by a "
                                f"factor of {gap:,.1f}, far beyond what market "
                                f"movement in the window could explain.")
            if gap > SIZE_TOLERANCE:
                return "WARN", (f"{detail}.{sizes} The adjustment is off by a "
                                f"factor of {gap:,.1f}. No payout amount is known "
                                f"for this event and the price fall includes "
                                f"market movement, so look before acting.")
        return "PASS", (f"{detail}.{sizes} The adjustment matches the price fall.")

    if applied and not fell:
        return "PASS", (f"{detail}. The ratio stepped and the price did not fall "
                        f"materially, which is an ordinary dividend rather than "
                        f"the event looked for here.")

    if fell and not applied:
        return "FAIL", (f"{detail}. The ratio is flat while the price fell "
                        f"{by_price_fall:.1%}, so the distribution was NOT adjusted.")

    return "INCONCLUSIVE", (f"{detail}. Neither an adjustment nor a material fall "
                            f"on this date, so the ex-date is probably wrong. "
                            f"{_nearby_steps(df, idx)}")


def context(df: pd.DataFrame, a: Action) -> str:
    """The window's largest absolute move, purely as background."""
    target = pd.Timestamp(a.date)
    idx, _ = locate(df, target)
    if idx is None:
        return ""
    col = "Close"
    lo, hi = max(1, idx - WINDOW), min(len(df), idx + WINDOW + 1)
    win = df.iloc[lo:hi]
    prior = df[col].shift(1).iloc[lo:hi]
    moves = (win[col] / prior - 1.0).dropna()
    if moves.empty:
        return ""
    j = moves.abs().idxmax()
    return f"window max |move| {moves[j]:+.1%} on {df.at[j, 'Date']:%Y-%m-%d}"


# --------------------------------------------------------------------------
# Optional online cross-check
# --------------------------------------------------------------------------

def scan_online(root: Path) -> int:
    """Ask the vendor what corporate actions exist, for EVERY ticker loaded.

    pull_prices.py passes actions=False, so this information is not in the
    files. It is the authoritative list of what SHOULD have been adjusted, and
    scanning every ticker rather than only those already in ACTIONS turns this
    from a confirmation of desk research into a way of finding what it missed.

    Returns the number of candidates worth adding to ACTIONS.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance is not installed, skipping the online scan")
        return 0

    # Everything the model loads: both equity folders plus the factors.
    folders = [(root / "tanker", "tanker"), (root / "drybulk", "drybulk"),
               (root.parent / "factors", "factors")]
    holdings = []
    for path, label in folders:
        if not path.is_dir():
            continue
        for f in sorted(path.glob("*.csv")):
            holdings.append((f.stem, label, f))

    known_splits = {(a.ticker, a.date) for a in ACTIONS if a.kind == "split"}
    known_dists = {(a.ticker, a.date) for a in ACTIONS if a.kind == "distribution"}
    candidates = 0

    for ticker, label, path in holdings:
        try:
            acts = yf.Ticker(ticker).actions
        except Exception as exc:
            print(f"  {ticker:9} {label:8} could not fetch ({type(exc).__name__})")
            continue
        if acts is None or acts.empty:
            print(f"  {ticker:9} {label:8} no actions reported "
                  f"(delisted or reconstructed series)")
            continue

        acts = acts.copy()
        acts.index = pd.to_datetime(acts.index, utc=True).tz_localize(None)

        # Local prices, to size each dividend against the share price.
        try:
            px = pd.read_csv(path)
            px["Date"] = pd.to_datetime(px["Date"], errors="coerce",
                                        utc=True).dt.tz_localize(None)
            px = px.dropna(subset=["Date"]).sort_values("Date")
        except Exception:
            px = None

        lines = []

        if "Stock Splits" in acts:
            for when, row in acts[acts["Stock Splits"] != 0].iterrows():
                iso = when.strftime("%Y-%m-%d")
                mark = "" if (ticker, iso) in known_splits else "   <-- ADD TO ACTIONS"
                if mark:
                    candidates += 1
                lines.append(f"    split  {iso}  ratio {row['Stock Splits']:.6g}{mark}")

        if "Dividends" in acts and px is not None:
            divs = acts[acts["Dividends"] > 0]["Dividends"]
            for when, amount in divs.items():
                near = px[px["Date"] <= when]
                if near.empty:
                    continue
                price = near.iloc[-1]["Close"]
                if not price or price <= 0:
                    continue
                share = amount / price
                if share < LARGE_DIVIDEND:
                    continue
                iso = when.strftime("%Y-%m-%d")
                mark = "" if (ticker, iso) in known_dists else "   <-- ADD TO ACTIONS"
                if mark:
                    candidates += 1
                lines.append(f"    payout {iso}  {amount:,.4f} = {share:.1%} "
                             f"of the {price:,.4f} close{mark}")

        n_div = int((acts["Dividends"] > 0).sum()) if "Dividends" in acts else 0
        print(f"  {ticker:9} {label:8} {n_div:3} dividends"
              + (", nothing material" if not lines else ""))
        for line in lines:
            print(line)

    return candidates


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def load_windows(path: Path) -> pd.DataFrame:
    """series_windows.csv, or an empty frame if there is none."""
    cols = ["Ticker", "ValidFrom", "ValidTo", "Reason"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    w = pd.read_csv(path)
    for c in ("ValidFrom", "ValidTo"):
        w[c] = pd.to_datetime(w[c], errors="coerce")
    return w


def outside_window(windows: pd.DataFrame, ticker: str, date: str) -> str | None:
    """Reason string if the event falls outside the ticker's validity window.
    An event the model never reads cannot corrupt it, so it is out of scope
    rather than a failure."""
    d = pd.Timestamp(date)
    for _, r in windows[windows["Ticker"] == ticker].iterrows():
        if pd.notna(r["ValidTo"]) and d > r["ValidTo"]:
            return f"after ValidTo {r['ValidTo']:%Y-%m-%d}: {r['Reason']}"
        if pd.notna(r["ValidFrom"]) and d < r["ValidFrom"]:
            return f"before ValidFrom {r['ValidFrom']:%Y-%m-%d}: {r['Reason']}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="path to data/equity")
    ap.add_argument("--windows", help="path to series_windows.csv "
                    "(default: beside data/equity)")
    ap.add_argument("--online", action="store_true",
                    help="also cross-check against Yahoo's actions series")
    args = ap.parse_args()

    root = resolve_root(args.root)
    print(f"data root: {root}\n")
    windows = load_windows(Path(args.windows) if args.windows
                           else root.parent / "series_windows.csv")

    cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    results = []

    print(f"{'verdict':14} {'ticker':9} {'date':12} {'action':40} detail")
    print("-" * 150)

    for a in ACTIONS:
        key = (a.ticker, a.folder)
        if key not in cache:
            cache[key] = load(root, a.ticker, a.folder)
        df = cache[key]

        excluded = outside_window(windows, a.ticker, a.date)
        if excluded:
            verdict, detail = "EXCLUDED", excluded
        elif df is None:
            verdict, detail = "NO FILE", f"no CSV for {a.ticker} in {a.folder}/"
        elif a.kind == "split":
            verdict, detail = test_split(df, a)
        else:
            verdict, detail = test_distribution(df, a)

        if df is not None and verdict in ("FAIL", "WARN", "INCONCLUSIVE"):
            extra = context(df, a)
            if extra:
                detail = f"{detail}  [{extra}]"

        results.append((verdict, a, detail))
        print(f"{verdict:14} {a.ticker:9} {a.date:12} {a.note[:40]:40} {detail}")

    # ---- file inventory, so a silent wrong-root is impossible to miss ----
    print("\nFiles found")
    print("-" * 150)
    for folder in ("tanker", "drybulk"):
        d = root / folder
        names = sorted(p.name for p in d.glob("*.csv")) if d.is_dir() else []
        print(f"  {folder:9} {len(names):2} files: {', '.join(names) if names else 'NONE'}")

    # ---- optional online cross-check ----
    candidates = 0
    if args.online:
        print("\nVendor actions scan, every ticker the model loads")
        print("-" * 150)
        candidates = scan_online(root)
        if candidates:
            print(f"\n  {candidates} action(s) marked ADD TO ACTIONS are not in "
                  f"this script's list. Add them and re-run.")
        else:
            print("\n  Nothing the vendor reports is missing from the list.")

    # ---- summary and exit status ----
    counts: dict[str, int] = {}
    for verdict, _, _ in results:
        counts[verdict] = counts.get(verdict, 0) + 1
    print("\nSummary:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    failures = [r for r in results if r[0] in ("FAIL", "NO FILE")]
    if candidates:
        return 1
    if failures:
        print("\nAction needed:")
        for verdict, a, detail in failures:
            print(f"  {a.ticker} {a.date}  {a.note}")
        return 1

    if any(v in ("WARN", "INCONCLUSIVE") for v, _, _ in results):
        print("\nNothing failed, but some tests were inconclusive. Read those rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
