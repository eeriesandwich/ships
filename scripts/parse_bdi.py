"""
Parse a TradingView WebSocket capture (raw text pasted from DevTools)
into a clean CSV of [Date, Open, High, Low, Close].

Usage:
    1. Paste the full raw WS message text into a file, e.g. bdi_raw.txt
       (it can contain multiple ~m~<len>~m~{...} frames concatenated together,
       exactly as copied from DevTools).
    2. Run: python parse_bdi.py bdi_raw.txt bdi.csv
"""
import sys
import re
import json
import pandas as pd
from datetime import datetime, timedelta, timezone

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def ts_to_date(ts):
    """Convert a Unix timestamp (possibly negative/pre-1970) to an ISO date.

    Uses pure timedelta arithmetic instead of datetime.fromtimestamp, because
    on Windows fromtimestamp calls into the C runtime's localtime() and fails
    with OSError for any date before 1970 -- even though those are perfectly
    valid timestamps. This sidesteps that platform limitation entirely.
    """
    return (EPOCH + timedelta(seconds=ts)).date().isoformat()


def split_frames(raw_text: str):
    """Split TradingView's ~m~<len>~m~<json> framing into individual JSON strings."""
    frames = []
    i = 0
    while True:
        m = re.compile(r"~m~(\d+)~m~").search(raw_text, i)
        if not m:
            break
        length = int(m.group(1))
        start = m.end()
        payload = raw_text[start:start + length]
        frames.append(payload)
        i = start + length
    return frames



# Generous epoch bounds: 1900-01-01 to 2100-01-01. This is only meant to
# catch genuinely bogus sentinel values (e.g. a stray 1e+100 from an
# unrelated message field), not to second-guess real historical dates --
# freight indices like BDTI have real data going back to the 1970s/80s.
MIN_TS = -2208988800
MAX_TS = 4102444800


def extract_bars(frames):
    """Find timescale_update / du messages and pull out [timestamp, o, h, l, c] bars."""
    bars = {}
    skipped = []
    for f in frames:
        try:
            msg = json.loads(f)
        except json.JSONDecodeError:
            continue
        if msg.get("m") not in ("timescale_update", "du"):
            continue
        params = msg.get("p", [])
        if len(params) < 2 or not isinstance(params[1], dict):
            continue
        for series_key, series_val in params[1].items():
            if not isinstance(series_val, dict):
                continue
            for bar in series_val.get("s", []):
                v = bar.get("v")
                if not v or len(v) < 5:
                    continue
                ts = v[0]
                try:
                    ts_ok = isinstance(ts, (int, float)) and MIN_TS <= ts <= MAX_TS
                except TypeError:
                    ts_ok = False
                if not ts_ok:
                    skipped.append((series_key, v))
                    continue
                bars[ts] = v[:5]  # de-dupe by timestamp, keep latest
    if skipped:
        print(f"Skipped {len(skipped)} entries with implausible timestamps, e.g.:")
        for series_key, v in skipped[:5]:
            print(f"  series={series_key!r} v={v!r}")
    return bars


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_bdi.py <raw_input.txt> <output.csv>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    frames = split_frames(raw_text)
    bars = extract_bars(frames)

    if not bars:
        print("No OHLC bars found. Check that the input contains a "
              "timescale_update/du message with an 's' array of {'i','v'} bars.")
        sys.exit(1)

    rows = []
    for ts, v in sorted(bars.items()):
        try:
            date = ts_to_date(ts)
        except (OverflowError, ValueError):
            print(f"Skipping unparseable timestamp: {ts!r}")
            continue
        rows.append({"Date": date, "Open": v[1], "High": v[2], "Low": v[3], "Close": v[4]})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Parsed {len(df)} bars, spanning {df['Date'].min()} to {df['Date'].max()}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
