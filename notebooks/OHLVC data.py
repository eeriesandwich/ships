#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import date

def pull_prices(tickers: dict, start: str, out_dir: Path, lineage: str):
    """Pull full available daily OHLCV history (start date through today).
    Date-window and era filtering happen downstream in Power Query, not here."""
    out_dir.mkdir(parents=True, exist_ok=True)
    end = date.today().isoformat()
    for ticker, name in tickers.items():
        print(f"Pulling {ticker} ({name})...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)
        if df.empty:
            print(f"  WARNING: no data returned for {ticker}")
            continue
        df = df.reset_index()
        df["Ticker"] = ticker
        df["Lineage"] = lineage
        out_path = out_dir / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        print(f"  saved {len(df)} rows to {out_path} "
              f"(spans {df['Date'].min().date()} to {df['Date'].max().date()})")

TANKER_TICKERS = {
    "CMBT.BR": "Euronav / CMB.TECH (Brussels listing, full history)",
    "FRO":  "Frontline",
    "DHT":  "DHT Holdings",
    "TNK":  "Teekay Tankers",
    "STNG": "Scorpio Tankers",
}
pull_prices(TANKER_TICKERS, start="2004-01-01", out_dir=Path("../raw/tanker"), lineage="Tanker")


# In[ ]:




