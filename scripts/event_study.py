"""
Event study for Phase 4 Step 10.

For each bounded-horizon event in data/events.csv, estimates a market model
(portfolio return against ACWI) over an estimation window ending 10 trading
days before the event, then computes abnormal returns and the cumulative
abnormal return (CAR) over the event window. Writes one row per event to
data/checks/event_study_<date>.csv.

Methodology:
  Estimation window: 500 US sessions ending 10 sessions before the event
    date, excluding any session flagged as an event day for the same lineage.
    Minimum 250 valid sessions, otherwise the event is skipped.
  Market model: r_portfolio_t = alpha + beta * r_market_t + epsilon_t,
    fit by OLS over the estimation window.
  Event window length by horizon:
    Day = 1 session, Week = 5, Month = 21, Quarter = 63.
    Regime and None are excluded.
  Abnormal return per event-window session:
    AR_t = r_portfolio_t - (alpha + beta * r_market_t)
  Cumulative abnormal return:
    CAR = sum of AR over the event-window sessions
  Standard error of CAR under the market model:
    SE_CAR = sigma_epsilon * sqrt(L)
  Test statistic:
    t = CAR / SE_CAR
  Two-sided p-value from the t distribution with (N - 2) degrees of freedom.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import date
from scipy import stats

REPO = Path(__file__).resolve().parents[1]

HORIZON_SESSIONS = {"Day": 1, "Week": 5, "Month": 21, "Quarter": 63}
EST_WINDOW = 500
EST_GAP = 10
EST_MIN = 250
MARKET_TICKER = "ACWI"


def load_market_returns() -> pd.Series:
    df = pd.read_csv(REPO / "data" / "factors" / f"{MARKET_TICKER}.csv",
                     parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return np.log(df["Adj Close"]).diff().dropna()


def load_fx(ticker: str):
    fx_file = REPO / "data" / "fx" / f"{ticker}.csv"
    if not fx_file.exists():
        return None
    df = pd.read_csv(fx_file, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Close"].ffill()


def load_portfolio_returns(lineage: str) -> pd.Series:
    lineage_dir = REPO / "data" / "equity" / lineage.lower()
    windows = pd.read_csv(REPO / "data" / "series_windows.csv",
                          parse_dates=["ValidFrom", "ValidTo"])
    companies = pd.read_csv(REPO / "data" / "companies.csv")
    fx_map = {
        "EUR": load_fx("EURUSD=X"),
        "NOK": load_fx("NOKUSD=X"),
    }

    per_ticker_returns = {}
    for csv in sorted(lineage_dir.glob("*.csv")):
        ticker = csv.stem
        df = pd.read_csv(csv, parse_dates=["Date"]).set_index("Date").sort_index()
        prices = df["Adj Close"].astype(float)

        ccy_rows = companies.loc[companies["Ticker"] == ticker, "ListingCurrency"]
        if len(ccy_rows) and ccy_rows.iloc[0] in fx_map and fx_map[ccy_rows.iloc[0]] is not None:
            fx = fx_map[ccy_rows.iloc[0]].reindex(prices.index).ffill()
            prices = prices * fx

        returns = np.log(prices).diff().dropna()

        w = windows[(windows["Ticker"] == ticker) &
                    (windows["WindowType"].isin(["Membership", "Load"]))]
        if len(w):
            lo = w["ValidFrom"].min() if w["ValidFrom"].notna().any() else returns.index.min()
            hi = w["ValidTo"].max() if w["ValidTo"].notna().any() else returns.index.max()
            returns = returns[(returns.index >= lo) & (returns.index <= hi)]

        per_ticker_returns[ticker] = returns

    panel = pd.concat(per_ticker_returns, axis=1)
    return panel.mean(axis=1, skipna=True).dropna()


def load_event_day_set(lineage: str) -> set:
    df = pd.read_csv(REPO / "data" / "events.csv", parse_dates=["EventDate"])
    df = df[df["Lineage"] == lineage]
    return set(df["EventDate"].dt.normalize())


def run_one_event(ev: pd.Series,
                  r_portfolio: pd.Series,
                  r_market: pd.Series,
                  event_days: set):
    event_date = ev["EventDate"].normalize()
    horizon = ev["Horizon"]
    L = HORIZON_SESSIONS[horizon]

    sessions = r_portfolio.index.intersection(r_market.index)

    before_event = sessions[sessions < event_date]
    if len(before_event) < EST_GAP + EST_MIN:
        return None
    est_end = before_event[-EST_GAP - 1]
    est_sessions = sessions[sessions <= est_end][-EST_WINDOW:]
    est_sessions = pd.DatetimeIndex([d for d in est_sessions if d.normalize() not in event_days])
    if len(est_sessions) < EST_MIN:
        return None

    y = r_portfolio.reindex(est_sessions).dropna()
    x = r_market.reindex(y.index).dropna()
    y = y.reindex(x.index)
    if len(y) < EST_MIN:
        return None

    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X).fit()
    alpha = model.params[0]
    beta = model.params[1]
    sigma_e = np.sqrt(model.mse_resid)
    r_squared = model.rsquared
    n_est = int(model.nobs)

    on_or_after = sessions[sessions >= event_date]
    if len(on_or_after) < 1:
        return None
    event_sessions = on_or_after[:L]
    if len(event_sessions) < 1:
        return None

    y_ev = r_portfolio.reindex(event_sessions).dropna()
    x_ev = r_market.reindex(y_ev.index).dropna()
    y_ev = y_ev.reindex(x_ev.index)
    if len(y_ev) < 1:
        return None

    ar = y_ev.values - (alpha + beta * x_ev.values)
    car = float(np.sum(ar))
    se_car = sigma_e * np.sqrt(len(y_ev))
    t_stat = car / se_car if se_car > 0 else np.nan
    df_resid = max(n_est - 2, 1)
    p_two_sided = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df_resid)))

    return {
        "event_id": ev["EventID"],
        "event_date": event_date.date().isoformat(),
        "event_name": ev["EventName"],
        "lineage": ev["Lineage"],
        "horizon": horizon,
        "horizon_days": L,
        "n_estimation": n_est,
        "alpha": alpha,
        "beta": beta,
        "sigma_epsilon": sigma_e,
        "r_squared": r_squared,
        "car": car,
        "se_car": se_car,
        "t_stat": t_stat,
        "p_two_sided": p_two_sided,
        "direction": ev["Direction"],
        "category": ev["Category"],
    }


def main():
    events = pd.read_csv(REPO / "data" / "events.csv", parse_dates=["EventDate"])
    events = events[events["Horizon"].isin(HORIZON_SESSIONS.keys())].copy()
    events = events[events["Lineage"].isin(["Tanker", "DryBulk"])].copy()
    print(f"Bounded events to process: {len(events)}")

    market = load_market_returns()
    portfolios = {
        "Tanker":  load_portfolio_returns("Tanker"),
        "DryBulk": load_portfolio_returns("DryBulk"),
    }
    event_days = {
        "Tanker":  load_event_day_set("Tanker"),
        "DryBulk": load_event_day_set("DryBulk"),
    }

    results = []
    skipped = 0
    for _, ev in events.iterrows():
        r = run_one_event(ev, portfolios[ev["Lineage"]], market, event_days[ev["Lineage"]])
        if r is None:
            skipped += 1
            continue
        results.append(r)

    df = pd.DataFrame(results)
    for col in ["alpha", "beta", "sigma_epsilon", "r_squared",
                "car", "se_car", "t_stat", "p_two_sided"]:
        df[col] = df[col].round(6)

    out = REPO / "data" / "checks" / f"event_study_{date.today().isoformat()}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nWrote {out}")
    print(f"Rows: {len(df)}, Skipped: {skipped}")
    print(f"\nAverage CAR across events: {df['car'].mean():.6f}")
    print(f"Average |CAR|: {df['car'].abs().mean():.6f}")
    print(f"Share of events with p < 0.05: {(df['p_two_sided'] < 0.05).mean():.2%}")
    print("\nFirst five rows:")
    print(df[["event_id", "event_date", "event_name", "lineage", "horizon",
              "beta", "car", "t_stat", "p_two_sided"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
