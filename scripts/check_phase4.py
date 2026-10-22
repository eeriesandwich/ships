"""
Phase 4 cross-check: independent Python computation of the Power BI DAX
measures, written out as reference CSVs under data/checks/.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
MIN_CONSTITUENTS = 3
MIN_QUARTER_COVERAGE = 0.95
LINEAGES = ["Tanker", "DryBulk"]


def load_fx(ticker):
    f = REPO / "data" / "fx" / f"{ticker}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Close"].ffill()


def load_session_calendar():
    df = pd.read_csv(REPO / "data" / "factors" / "ACWI.csv", parse_dates=["Date"])
    return pd.DatetimeIndex(sorted(df["Date"].unique()))


def load_constituent_returns(lineage, use_adj=True, in_usd=True):
    lineage_dir = REPO / "data" / "equity" / lineage.lower()
    windows = pd.read_csv(REPO / "data" / "series_windows.csv",
                          parse_dates=["ValidFrom", "ValidTo"])
    companies = pd.read_csv(REPO / "data" / "companies.csv")
    fx_map = {"EUR": load_fx("EURUSD=X"), "NOK": load_fx("NOKUSD=X")}
    out = {}
    for csv in sorted(lineage_dir.glob("*.csv")):
        ticker = csv.stem
        df = pd.read_csv(csv, parse_dates=["Date"]).set_index("Date").sort_index()
        col = "Adj Close" if use_adj else "Close"
        prices = df[col].astype(float)
        ccy_rows = companies.loc[companies["Ticker"] == ticker, "ListingCurrency"]
        ccy = ccy_rows.iloc[0] if len(ccy_rows) else "USD"
        if in_usd and ccy in fx_map and fx_map[ccy] is not None:
            fx = fx_map[ccy].reindex(prices.index).ffill()
            prices = prices * fx
        returns = np.log(prices).diff().dropna()
        w = windows[(windows["Ticker"] == ticker) &
                    (windows["WindowType"].isin(["Membership", "Load"]))]
        if len(w):
            lo = w["ValidFrom"].min() if w["ValidFrom"].notna().any() else returns.index.min()
            hi = w["ValidTo"].max() if w["ValidTo"].notna().any() else returns.index.max()
            returns = returns[(returns.index >= lo) & (returns.index <= hi)]
        out[ticker] = returns
    return pd.concat(out, axis=1)


def compute_portfolio_returns(lineage, sessions, use_adj=True, in_usd=True):
    panel = load_constituent_returns(lineage, use_adj=use_adj, in_usd=in_usd).reindex(sessions)
    n_valid = panel.notna().sum(axis=1)
    port = panel.mean(axis=1, skipna=True)
    port[n_valid < MIN_CONSTITUENTS] = np.nan
    return port


def quarterly_return_panel(daily, sessions):
    sess_df = pd.DataFrame({"Date": sessions})
    sess_df["Year"] = sess_df["Date"].dt.year
    sess_df["Quarter"] = sess_df["Date"].dt.quarter
    sess_counts = sess_df.groupby(["Year", "Quarter"]).size().rename("UsDays")
    returns = daily.reindex(sessions).to_frame("r")
    returns["Year"] = returns.index.year
    returns["Quarter"] = returns.index.quarter
    valid = returns.dropna(subset=["r"])
    agg = valid.groupby(["Year", "Quarter"]).agg(
        n_gated=("r", "size"), q_return=("r", "sum"),
    )
    agg = agg.join(sess_counts).reset_index()
    agg["coverage"] = agg["n_gated"] / agg["UsDays"]
    agg["q_return"] = np.where(agg["coverage"] >= MIN_QUARTER_COVERAGE,
                               agg["q_return"], np.nan)
    return agg[["Year", "Quarter", "q_return"]]


def seasonal_measures(quarterly):
    pivot = quarterly.pivot(index="Year", columns="Quarter", values="q_return")
    full = pivot.dropna(how="any")
    year_means = full.mean(axis=1)
    deviations = full.sub(year_means, axis=0)
    rows = []
    for q in [1, 2, 3, 4]:
        dev = deviations[q]
        n = int(dev.count())
        mean = float(dev.mean())
        sd = float(dev.std(ddof=1))
        if n > 1:
            t_crit = stats.t.ppf(0.975, df=n - 1)
            half = t_crit * sd / np.sqrt(n)
            t_lo, t_hi = mean - half, mean + half
            mde = (stats.t.ppf(0.975, df=n - 1) + stats.t.ppf(0.80, df=n - 1)) * sd / np.sqrt(n)
        else:
            t_lo = t_hi = mde = np.nan
        rows.append({
            "Quarter": q, "n_years": n,
            "seasonal_mean": float(full[q].mean()),
            "seasonal_deviation": mean, "sd": sd,
            "t_lower": t_lo, "t_upper": t_hi, "mde": mde,
        })
    return pd.DataFrame(rows)


def freight_measures(name, change_mode):
    df = pd.read_csv(REPO / "data" / "freight" / f"{name}.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    closes = df["Close"].astype(float)
    quarter_end_closes = closes.resample("QE").last().dropna()
    frame = pd.DataFrame({"Close": quarter_end_closes})
    frame["Year"] = frame.index.year
    frame["Quarter"] = frame.index.quarter
    frame["PrevClose"] = frame["Close"].shift(1)
    if change_mode == "Points":
        frame["chg"] = frame["Close"] - frame["PrevClose"]
    else:
        frame["chg"] = (frame["Close"] - frame["PrevClose"]) / frame["PrevClose"].abs()
    frame = frame.dropna(subset=["chg"])
    pivot = frame.pivot(index="Year", columns="Quarter", values="chg").dropna(how="any")
    year_means = pivot.mean(axis=1)
    deviations = pivot.sub(year_means, axis=0)
    rows = []
    for q in [1, 2, 3, 4]:
        dev = deviations[q]
        chg = pivot[q]
        n = int(dev.count())
        mean_dev = float(dev.mean())
        mean_chg = float(chg.mean())
        sd = float(chg.std(ddof=1))
        if n > 1:
            t_crit = stats.t.ppf(0.975, df=n - 1)
            half = t_crit * sd / np.sqrt(n)
            t_lo, t_hi = mean_chg - half, mean_chg + half
            mde = (stats.t.ppf(0.975, df=n - 1) + stats.t.ppf(0.80, df=n - 1)) * sd / np.sqrt(n)
        else:
            t_lo = t_hi = mde = np.nan
        rows.append({
            "Series": name, "Quarter": q, "n_years": n,
            "quarter_change_mean": mean_chg,
            "within_year_deviation": mean_dev,
            "sd": sd, "t_lower": t_lo, "t_upper": t_hi, "mde": mde,
        })
    return pd.DataFrame(rows)


def equity_contrast_tanker(quarterly):
    pivot = quarterly.pivot(index="Year", columns="Quarter", values="q_return").dropna(how="any")
    contrast = (pivot[4] + pivot[1] - pivot[2] - pivot[3]) / 2
    n = int(contrast.count())
    mean = float(contrast.mean())
    sd = float(contrast.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    p = float(1 - stats.t.cdf(t, df=n - 1)) if n > 1 and not np.isnan(t) else np.nan
    return {"lineage": "Tanker", "contrast": "Q4+Q1 vs Q2+Q3",
            "point": mean, "sd": sd, "n_years": n, "t_stat": t, "p_one_sided_upper": p}


def drybulk_q1_test(seasonal):
    row = seasonal[seasonal["Quarter"] == 1].iloc[0]
    n = int(row["n_years"])
    mean = float(row["seasonal_deviation"])
    sd = float(row["sd"])
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    p = float(stats.t.cdf(t, df=n - 1)) if n > 1 and not np.isnan(t) else np.nan
    return {"lineage": "DryBulk", "contrast": "Q1 one-sided negative",
            "point": mean, "sd": sd, "n_years": n, "t_stat": t, "p_one_sided_lower": p}


def decomposition(lineage, sessions):
    total = compute_portfolio_returns(lineage, sessions, use_adj=True, in_usd=True)
    price = compute_portfolio_returns(lineage, sessions, use_adj=False, in_usd=True)
    local = compute_portfolio_returns(lineage, sessions, use_adj=True, in_usd=False)

    def quarterly(series):
        return quarterly_return_panel(series, sessions)[["Year", "Quarter", "q_return"]]

    tot_q = quarterly(total).rename(columns={"q_return": "total"})
    pri_q = quarterly(price).rename(columns={"q_return": "price"})
    loc_q = quarterly(local).rename(columns={"q_return": "local"})
    merged = tot_q.merge(pri_q, on=["Year", "Quarter"]).merge(loc_q, on=["Year", "Quarter"])
    merged["payout_effect"] = merged["total"] - merged["price"]
    merged["fx_effect"] = merged["total"] - merged["local"]
    out = merged.groupby("Quarter").agg(
        n_years=("total", "count"),
        total_mean=("total", "mean"),
        price_mean=("price", "mean"),
        local_mean=("local", "mean"),
        payout_effect_mean=("payout_effect", "mean"),
        fx_effect_mean=("fx_effect", "mean"),
    ).reset_index()
    out.insert(0, "lineage", lineage)
    return out


def main():
    today = date.today().isoformat()
    sessions = load_session_calendar()
    out_dir = REPO / "data" / "checks"
    out_dir.mkdir(exist_ok=True)

    equity_rows, contrasts = [], []
    for lineage in LINEAGES:
        print(f"Computing {lineage} portfolio...")
        port = compute_portfolio_returns(lineage, sessions)
        qr = quarterly_return_panel(port, sessions)
        seasonal = seasonal_measures(qr)
        seasonal.insert(0, "lineage", lineage)
        equity_rows.append(seasonal)
        if lineage == "Tanker":
            contrasts.append(equity_contrast_tanker(qr))
        if lineage == "DryBulk":
            contrasts.append(drybulk_q1_test(seasonal))
    equity = pd.concat(equity_rows, ignore_index=True)
    equity.to_csv(out_dir / f"phase4_reference_equity_{today}.csv", index=False)
    print(f"\nEquity reference:\n{equity.round(4).to_string(index=False)}")

    freight_rows = []
    for series, mode in [("BDTI", "Percent"), ("BDI", "Percent"), ("BCI", "Points")]:
        print(f"\nComputing {series} freight panel...")
        freight_rows.append(freight_measures(series, mode))
    freight = pd.concat(freight_rows, ignore_index=True)
    freight.to_csv(out_dir / f"phase4_reference_freight_{today}.csv", index=False)
    print(f"\nFreight reference:\n{freight.round(4).to_string(index=False)}")

    con_df = pd.DataFrame(contrasts)
    con_df.to_csv(out_dir / f"phase4_reference_contrasts_{today}.csv", index=False)
    print(f"\nContrasts:\n{con_df.round(4).to_string(index=False)}")

    decomp_rows = []
    for lineage in LINEAGES:
        print(f"\nComputing {lineage} decomposition...")
        decomp_rows.append(decomposition(lineage, sessions))
    decomp = pd.concat(decomp_rows, ignore_index=True)
    decomp.to_csv(out_dir / f"phase4_reference_decomposition_{today}.csv", index=False)
    print(f"\nDecomposition:\n{decomp.round(4).to_string(index=False)}")

    print(f"\nAll reference CSVs written to {out_dir}")


if __name__ == "__main__":
    main()
