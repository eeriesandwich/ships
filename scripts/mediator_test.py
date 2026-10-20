"""
Mediator test for Phase 4 Step 3, with optional event-day exclusion.

Three regressions per (freight, factor, focus quarter):
  1. freight_q = raw_alpha(q) + u
  2. factor_q  = delta(q) + v
  3. freight_q = alpha(q) + beta * factor_q + e
Identity: raw_alpha(q) = alpha(q) + beta * delta(q)
Mediated share for q = beta * delta(q) / raw_alpha(q)

Each combination runs twice: Include (every trading day) and Exclude
(dates flagged Scope=Sector AND Direction!=Neutral in data/events.csv for
the matching lineage). The quarterly change for Include equals the
close-to-close quarterly log return (log last_Q - log last_prev_Q), so
the Include numbers match the earlier mediator-test output exactly.
For Exclude the sum of event-day daily log returns in each quarter is
subtracted from that baseline, which cleanly reconstructs the quarterly
log return had event-day moves been zero without inflating sample size.

Bootstrap: resample complete years with replacement, 2000 iterations.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import date

REPO = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20261019)
N_BOOT = 2000


def load_freight(name: str) -> pd.Series:
    df = pd.read_csv(REPO / "data" / "freight" / f"{name}.csv", parse_dates=["Date"])
    return df.set_index("Date")["Close"].sort_index()


def load_factor(ticker: str) -> pd.Series:
    df = pd.read_csv(REPO / "data" / "factors" / f"{ticker}.csv", parse_dates=["Date"])
    return df.set_index("Date")["Adj Close"].sort_index()


def load_event_dates(lineage: str) -> set:
    """Sector-wide directional event dates for the given lineage."""
    df = pd.read_csv(REPO / "data" / "events.csv", parse_dates=["EventDate"])
    mask = (
        (df["Scope"] == "Sector")
        & (df["Direction"] != "Neutral")
        & (df["Lineage"] == lineage)
    )
    return set(df.loc[mask, "EventDate"].dt.normalize())


def quarterly_log_change(s: pd.Series, event_dates: set | None = None) -> pd.DataFrame:
    """Close-to-close quarterly log change.

    Include (default): log(last_Q) - log(last_prev_Q)
    Exclude: Include minus sum of event-day daily log returns in that quarter.
    The first partial quarter of the series gets a NaN chg and is dropped,
    matching the original behaviour.
    """
    qe = s.resample("QE").last().dropna()
    base = pd.DataFrame({
        "Year": qe.index.year,
        "Q": qe.index.quarter,
        "log": np.log(qe.values),
    })
    base["chg"] = base["log"].diff()
    out = base.dropna(subset=["chg"]).reset_index(drop=True)[["Year", "Q", "chg"]]

    if event_dates:
        daily = np.log(s.astype(float)).diff().dropna()
        mask = pd.Index(daily.index).normalize().isin(event_dates)
        if mask.any():
            ev = daily[mask]
            ec = pd.DataFrame({
                "Year": ev.index.year,
                "Q": ev.index.quarter,
                "ec": ev.values,
            }).groupby(["Year", "Q"], as_index=False)["ec"].sum()
            out = out.merge(ec, on=["Year", "Q"], how="left").fillna({"ec": 0.0})
            out["chg"] = out["chg"] - out["ec"]
            out = out.drop(columns=["ec"])
    return out


def align_overlap(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    j = y.merge(x, on=["Year", "Q"], suffixes=("_y", "_x")).dropna()
    counts = j.groupby("Year").size()
    good = counts[counts == 4].index
    j = j[j["Year"].isin(good)].reset_index(drop=True)
    return j


def fit_alpha_only(data: pd.DataFrame, ycol: str) -> np.ndarray:
    D = pd.get_dummies(data["Q"], prefix="Q").astype(float)
    y = data[ycol].values
    model = sm.OLS(y, D.values).fit()
    return model.params


def fit_alpha_beta(data: pd.DataFrame, ycol: str, xcol: str):
    D = pd.get_dummies(data["Q"], prefix="Q").astype(float)
    X = pd.concat([D, data[[xcol]].astype(float)], axis=1)
    y = data[ycol].values
    model = sm.OLS(y, X.values).fit()
    alphas = model.params[:4]
    beta = model.params[4]
    return alphas, beta


def mediated_share(data: pd.DataFrame, ycol: str, xcol: str, focus_q: int) -> dict:
    raw_a = fit_alpha_only(data, ycol)
    delta_a = fit_alpha_only(data, xcol)
    direct_a, beta = fit_alpha_beta(data, ycol, xcol)
    idx = focus_q - 1
    raw = raw_a[idx]
    med = beta * delta_a[idx]
    direct = direct_a[idx]
    share = med / raw if raw != 0 else np.nan
    return {
        "raw_alpha": raw,
        "direct_alpha": direct,
        "beta": beta,
        "delta_focus": delta_a[idx],
        "mediated_component": med,
        "mediated_share": share,
    }


def bootstrap_shares(data: pd.DataFrame, ycol: str, xcol: str, focus_q: int,
                     n: int = N_BOOT) -> np.ndarray:
    years = sorted(data["Year"].unique())
    k = len(years)
    out = np.full(n, np.nan)
    for i in range(n):
        picks = RNG.choice(years, size=k, replace=True)
        parts = [data[data["Year"] == y] for y in picks]
        boot = pd.concat(parts, ignore_index=True)
        try:
            r = mediated_share(boot, ycol, xcol, focus_q)
            out[i] = r["mediated_share"]
        except Exception:
            pass
    return out[~np.isnan(out)]


def decision_rule(ci_low: float, ci_high: float) -> str:
    if ci_high < 1 / 3:
        return "sector_valid"
    if ci_low > 2 / 3:
        return "sector_over_controlled"
    return "spans_threshold"


def run_one(label: str, y: pd.DataFrame, x: pd.DataFrame, ycol: str, xcol: str,
            focus_q: int, events_scope: str, note: str = "") -> dict:
    panel = align_overlap(y, x)
    n_years = panel["Year"].nunique()
    first_y = panel["Year"].min()
    last_y = panel["Year"].max()
    point = mediated_share(panel, ycol, xcol, focus_q)
    boot = bootstrap_shares(panel, ycol, xcol, focus_q)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    decision = decision_rule(ci_low, ci_high)
    return {
        "label": label,
        "focus_q": focus_q,
        "events": events_scope,
        "window": f"{first_y} to {last_y}",
        "n_years": n_years,
        "raw_alpha": point["raw_alpha"],
        "direct_alpha": point["direct_alpha"],
        "beta": point["beta"],
        "delta_focus": point["delta_focus"],
        "mediated_component": point["mediated_component"],
        "mediated_share": point["mediated_share"],
        "boot_lo": ci_low,
        "boot_hi": ci_high,
        "decision": decision,
        "note": note,
    }


def main():
    bdti_daily = load_freight("BDTI")
    bdi_daily = load_freight("BDI")
    bz_daily = load_factor("BZ=F")
    cl_daily = load_factor("CL=F")
    mchi_daily = load_factor("MCHI")
    vale_daily = load_factor("VALE")
    rio_daily = load_factor("RIO")

    tanker_events = load_event_dates("Tanker")
    drybulk_events = load_event_dates("DryBulk")

    def iron_q(ev=None):
        v = quarterly_log_change(vale_daily, ev).rename(columns={"chg": "vale"})
        r = quarterly_log_change(rio_daily, ev).rename(columns={"chg": "rio"})
        j = v.merge(r, on=["Year", "Q"])
        j["factor"] = 0.5 * j["vale"] + 0.5 * j["rio"]
        return j[["Year", "Q", "factor"]]

    results = []
    for scope, tev, dev in [("Include", None, None),
                            ("Exclude", tanker_events, drybulk_events)]:
        bdti = quarterly_log_change(bdti_daily, tev).rename(columns={"chg": "freight"})
        bdi = quarterly_log_change(bdi_daily, dev).rename(columns={"chg": "freight"})
        bz = quarterly_log_change(bz_daily, tev).rename(columns={"chg": "factor"})
        cl = quarterly_log_change(cl_daily, tev).rename(columns={"chg": "factor"})
        mchi = quarterly_log_change(mchi_daily, dev).rename(columns={"chg": "factor"})
        iron = iron_q(dev)

        results.append(run_one("BDTI vs Brent (BZ=F)", bdti, bz, "freight", "factor", 4,
                               scope, note="headline; roll-yield caveat for BZ=F"))
        results.append(run_one("BDTI vs WTI (CL=F) [robustness]", bdti, cl, "freight", "factor", 4,
                               scope, note="robustness rerun for Brent"))
        results.append(run_one("BDI vs Iron ore miners (VALE+RIO)", bdi, iron, "freight", "factor", 3,
                               scope, note="dry bulk supply channel, Q3"))
        results.append(run_one("BDI vs Iron ore miners (VALE+RIO)", bdi, iron, "freight", "factor", 1,
                               scope, note="dry bulk supply channel, Q1"))
        results.append(run_one("BDI vs China (MCHI)", bdi, mchi, "freight", "factor", 3,
                               scope, note="dry bulk demand channel, Q3"))
        results.append(run_one("BDI vs China (MCHI)", bdi, mchi, "freight", "factor", 1,
                               scope, note="dry bulk demand channel, Q1"))

    df = pd.DataFrame(results)
    for col in ["raw_alpha", "direct_alpha", "beta", "delta_focus",
                "mediated_component", "mediated_share", "boot_lo", "boot_hi"]:
        df[col] = df[col].round(4)

    out_dir = REPO / "data" / "checks"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"mediator_test_{date.today().isoformat()}.csv"
    df.to_csv(out_path, index=False)

    print(f"\nWrote {out_path}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
