"""
NOVEL CURVE-RESIDUALIZATION TEST
================================
Purpose
-------
Test a genuinely stronger hypothesis than ordinary basis/curve momentum:

    Does the *local front-end* movement of a commodity futures curve contain
    predictive information that survives removal of conventional commodity
    momentum, carry/basis, global curve shape, and cross-commodity common factors?

This script is intentionally an IDENTIFICATION exercise, not a Sharpe optimizer.
There is one pre-specified primary specification and a small number of clearly
labeled diagnostics.

Data contract
-------------
The existing px_clean.parquet contains contract_0 and contract_1. That is enough
for the existing strategy, but NOT enough to identify curvature. For the primary
novel test, the parquet must contain at least contract_0..contract_3 and matching
settle_0..settle_3 and expiry_0..expiry_3.

The script therefore:
  1. Rebuilds monthly returns strictly inside each individual contract life.
  2. Builds 12m cumulative momentum for four curve points.
  3. Builds conventional controls:
       - front momentum
       - front basis / curve slope
       - broad commodity factor momentum
       - same-commodity remote-curve momentum
       - global curve slope
       - global curve curvature
  4. Constructs the primary LOCAL FRONT-END RESIDUAL:
       local = slope(0,1) - projection on the rest of the curve
     with the projection estimated only from information available before t.
  5. Runs monthly predictive cross-sectional tests:
       next-month front return ~ controls + local residual
     using only lagged predictors.
  6. Forms a simple rank long-short portfolio from the local residual and
     reports cost-free and cost-stressed diagnostics.

Important
---------
The data may have different contract availability across assets. The code never
backfills an unavailable contract. It requires contemporaneous valid observations.

Usage
-----
    python novel_curve_test.py --prices data/px_clean.parquet

Optional:
    --formation 12       cumulative months used for momentum
    --fit-window 60      rolling months for within-commodity residualization
    --n-curve 4          number of curve points required (minimum 4)
    --cost-bps 10        round-trip cost proxy per trade side in bps

Interpretation
--------------
The result we WANT is not merely a high Sharpe.
The strongest evidence is:
  (A) local residual has positive predictive coefficient after instrument-varying controls,
  (B) the effect remains in rolling / walk-forward form,
  (C) the rank portfolio earns an incremental return/IC,
  (D) placebo permutations kill it, and
  (E) remote curve maturities do not reproduce the effect.

If (A)-(E) fail, that is a useful negative result: it tells us the existing
basis/curve-momentum factor spans the proposed novelty.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    BY_SYMBOL = {}


CAPITAL = 450_000.0
DEFAULT_FORMATION = 12
DEFAULT_FIT = 60
DEFAULT_CURVE = 4
MIN_CS = 6


@dataclass
class Result:
    name: str
    value: float


def load(path: str, n_curve: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    req = ["date", "symbol"]
    for k in range(n_curve):
        req += [f"contract_{k}", f"settle_{k}", f"expiry_{k}"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(
            "PRIMARY NOVEL TEST CANNOT RUN. Missing columns: "
            + ", ".join(missing)
            + "\nThe current two-leg file is enough for the existing BM strategy, "
            "but not for identifying a distinct curvature/local-front component."
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for k in range(n_curve):
        df[f"expiry_{k}"] = pd.to_datetime(df[f"expiry_{k}"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    if BY_SYMBOL:
        df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
        df = df[df["asset"] == "commodity"].copy()

    # Drop duplicate symbol/date records deterministically, keeping the latest OI if present.
    keys = ["symbol", "date"]
    if "oi_0" in df.columns:
        df = (df.sort_values(keys + ["oi_0"], na_position="first")
                .drop_duplicates(keys, keep="last"))
    else:
        df = df.drop_duplicates(keys, keep="last")

    # Return each leg strictly within its own contract life: when contract ID changes,
    # the first observation has no artificial return from the prior contract.
    for k in range(n_curve):
        c = f"contract_{k}"
        p = f"settle_{k}"
        blk = df.groupby("symbol")[c].transform(lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[p].shift(1)
        rr = np.log(df[p] / prev)
        df[f"r{k}"] = rr.where(np.isfinite(rr))

    return df


def monthly_panel(df: pd.DataFrame, formation: int, n_curve: int) -> pd.DataFrame:
    """Monthly observations on a fixed first-trading-day grid."""
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M")
    d["dom"] = d.groupby(["symbol", "ym"]).cumcount()
    marks = (d[d["dom"] == 0][["symbol", "ym", "date"]]
             .rename(columns={"date": "mark"}))

    # cumulative log returns within each listed contract life
    for k in range(n_curve):
        d[f"c{k}"] = d.groupby("symbol")[f"r{k}"].transform(
            lambda s: s.fillna(0.0).cumsum())

    keep = ["symbol", "ym", "date", "mark"]
    for k in range(n_curve):
        keep += [f"c{k}", f"settle_{k}", f"expiry_{k}"]
    snap = d.merge(marks, left_on=["symbol", "ym", "date"],
                   right_on=["symbol", "ym", "mark"], how="inner")
    snap = snap[keep].sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = snap.groupby("symbol", sort=False)

    for k in range(n_curve):
        snap[f"m{k}"] = g[f"c{k}"].diff().rolling(formation, min_periods=formation).sum()

    # The expression above is subtly wrong if .rolling crosses groups on some pandas versions;
    # overwrite with a guaranteed per-group transform.
    for k in range(n_curve):
        snap[f"ret{k}"] = g[f"c{k}"].diff()
        snap[f"m{k}"] = g[f"ret{k}"].transform(
            lambda s: s.rolling(formation, min_periods=formation).sum())

    # One-month forward front-contract return. The monthly observation is formed at t,
    # and the return t->t+1 is the trading target.
    snap["fwd0"] = g["ret0"].shift(-1)

    # Curve geometry in return space.
    # Local front slope momentum is the familiar BM object, but we also compute deeper
    # segment slopes and a curvature measure to isolate the front-end-specific part.
    snap["s01"] = snap["m0"] - snap["m1"]
    snap["s12"] = snap["m1"] - snap["m2"]
    snap["s23"] = snap["m2"] - snap["m3"]
    snap["remote_slope"] = (snap["s12"] + snap["s23"]) / 2.0
    snap["front_curv"] = snap["s01"] - snap["s12"]

    # A global commodity momentum factor is the cross-sectional average of front momentum.
    snap["market_mom"] = snap.groupby("ym")["m0"].transform("mean")

    # A global curve factor set across the cross-section.
    snap["global_slope"] = snap.groupby("ym")["s01"].transform("mean")
    snap["global_remote_slope"] = snap.groupby("ym")["remote_slope"].transform("mean")
    snap["global_curvature"] = snap.groupby("ym")["front_curv"].transform("mean")

    # Instantaneous front basis / slope. This is intentionally a price-level control,
    # not a return, so it can be differentiated from BM.
    p0 = snap["settle_0"]
    p1 = snap["settle_1"]
    snap["basis01"] = np.log(p0 / p1)

    # Basis expressed as an annualized-ish curve slope where expiry dates are known.
    # This is a diagnostic; the raw log price basis is the main control.
    tau = (snap["expiry_1"] - snap["expiry_0"]).dt.days / 365.25
    snap["ann_basis01"] = snap["basis01"] / tau.replace(0, np.nan)

    return snap


def rolling_time_residual(panel: pd.DataFrame, y: str, xs: list[str], window: int) -> pd.Series:
    """Within-commodity rolling OLS residual, fitted only on historical rows."""
    out = pd.Series(index=panel.index, dtype=float)
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("ym")
        arr_y = g[y].to_numpy(dtype=float)
        arr_x = g[xs].to_numpy(dtype=float)
        vals = np.full(len(g), np.nan)
        for i in range(len(g)):
            lo = max(0, i - window)
            hist_y = arr_y[lo:i]
            hist_x = arr_x[lo:i]
            valid = np.isfinite(hist_y) & np.isfinite(hist_x).all(axis=1)
            if valid.sum() < max(24, len(xs) * 5):
                continue
            X = np.column_stack([np.ones(valid.sum()), hist_x[valid]])
            yy = hist_y[valid]
            try:
                beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(arr_y[i]) or not np.isfinite(arr_x[i]).all():
                continue
            vals[i] = arr_y[i] - np.array([1.0, *arr_x[i]]) @ beta
        out.loc[g.index] = vals
    return out


def residualize_cross_section(panel: pd.DataFrame, y: str, xs: list[str]) -> pd.Series:
    """Month-by-month cross-sectional residual. All regressors are observed at t."""
    out = pd.Series(index=panel.index, dtype=float)
    for ym, g in panel.groupby("ym", sort=False):
        cols = [y] + xs
        z = g[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < max(MIN_CS, len(xs) + 3):
            continue
        X = np.column_stack([np.ones(len(z)), z[xs].to_numpy(float)])
        yy = z[y].to_numpy(float)
        try:
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = yy - X @ beta
        out.loc[z.index] = resid
    return out


def make_signals(panel: pd.DataFrame, fit_window: int) -> pd.DataFrame:
    p = panel.copy()

    # Primary novelty candidate:
    # residualize the front 0-1 slope momentum against the *rest of the same curve*
    # using a rolling within-commodity model. This is deliberately not PCA across
    # commodities and does not use any future information.
    p["local_resid"] = rolling_time_residual(
        p,
        y="s01",
        xs=["s12", "s23", "m0"],
        window=fit_window,
    )

    # More demanding residual: remove broad cross-sectional commodity factors from the
    # local residual. This asks whether the local curve piece survives conventional factors.
    p["local_resid_xs"] = residualize_cross_section(
        p,
        y="local_resid",
        xs=["m0", "basis01", "global_slope", "global_remote_slope", "global_curvature", "market_mom"],
    )

    # Conventional benchmark signals.
    p["bm"] = p["s01"]
    p["front_mom"] = p["m0"]
    p["remote_mom"] = p["remote_slope"]

    return p


def hac_mean_t(x: np.ndarray, lags: int = 3) -> tuple[float, float]:
    """Newey-West t-statistic for the mean of a monthly coefficient series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 24:
        return np.nan, np.nan
    mu = float(x.mean())
    u = x - mu
    gamma0 = float(np.dot(u, u) / n)
    var = gamma0
    for L in range(1, min(lags, n - 1) + 1):
        gamma = float(np.dot(u[L:], u[:-L]) / n)
        weight = 1.0 - L / (lags + 1.0)
        var += 2.0 * weight * gamma
    se = math.sqrt(max(var, 0.0) / n)
    return mu, (mu / se if se > 0 else np.nan)


def fmb(panel: pd.DataFrame, y: str, x: str) -> dict:
    betas = []
    for ym, g in panel.groupby("ym", sort=False):
        z = g[[y, x]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < MIN_CS:
            continue
        X = np.column_stack([np.ones(len(z)), z[x].to_numpy(float)])
        yy = z[y].to_numpy(float)
        try:
            b, *_ = np.linalg.lstsq(X, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        betas.append(b[1])
    if len(betas) < 24:
        return {"months": len(betas), "beta": np.nan, "t_hac3": np.nan}
    beta, t = hac_mean_t(np.asarray(betas), lags=3)
    return {"months": len(betas), "beta": beta, "t_hac3": t}


def fmb_multivariate(panel: pd.DataFrame, y: str, xs: list[str]) -> dict:
    """Fama-MacBeth monthly slopes with HAC(3) inference across months.

    Note that variables which are constant across the cross-section in a given month
    are automatically absorbed by the monthly intercept and provide no FMB information.
    The primary specification therefore uses instrument-varying controls only.
    """
    betas = []
    for ym, g in panel.groupby("ym", sort=False):
        cols = [y] + xs
        z = g[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < max(MIN_CS, len(xs) + 3):
            continue
        X = np.column_stack([np.ones(len(z)), z[xs].to_numpy(float)])
        yy = z[y].to_numpy(float)
        try:
            b, *_ = np.linalg.lstsq(X, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        # Reject numerically singular months rather than silently accepting arbitrary
        # pseudoinverse coefficients when two controls are nearly duplicates.
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        betas.append(b[1:])
    if len(betas) < 24:
        return {"months": len(betas), **{f"{x}_beta": np.nan for x in xs},
                **{f"{x}_t_hac3": np.nan for x in xs}}
    B = np.asarray(betas)
    out = {"months": len(B)}
    for j, x in enumerate(xs):
        beta, t = hac_mean_t(B[:, j], lags=3)
        out[f"{x}_beta"] = beta
        out[f"{x}_t_hac3"] = t
    return out

def rank_portfolio(panel: pd.DataFrame, signal: str, cost_bps_side: float = 0.0) -> pd.Series:
    rows = []
    prev = {}
    for ym, g in panel.groupby("ym", sort=True):
        z = g[["symbol", signal, "fwd0"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < MIN_CS:
            continue
        # Demeaned ranks. Purely cross-sectional; no parameter search.
        r = z[signal].rank(method="average")
        w = r - r.mean()
        gross = np.abs(w).sum()
        if gross <= 0:
            continue
        w = w / gross
        pnl = 0.0
        cost = 0.0
        for _, row in z.iterrows():
            sym = row["symbol"]
            wi = w.loc[row.name]
            pnl += wi * row["fwd0"]
            # Simple turnover cost proxy in return units: |change in rank weight| * 2 sides.
            old = prev.get(sym, 0.0)
            cost += abs(wi - old) * (2.0 * cost_bps_side / 1e4)
            prev[sym] = wi
        rows.append((ym, pnl - cost))
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(dict(rows)).sort_index()


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 48:
        return {"n": len(r), "ann": np.nan, "vol": np.nan, "sharpe": np.nan, "t": np.nan, "dd": np.nan}
    ann = r.mean() * 12
    vol = r.std(ddof=1) * math.sqrt(12)
    sr = ann / vol if vol > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {"n": len(r), "ann": ann, "vol": vol, "sharpe": sr, "t": sr * math.sqrt(len(r)/12), "dd": dd}


def placebo(panel: pd.DataFrame, signal: str, n: int = 200, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    real = rank_portfolio(panel, signal)
    real_stats = stats(real)
    months = panel["ym"].sort_values().unique()
    vals = []
    for _ in range(n):
        q = panel.copy()
        # Shuffle signal cross-sectionally within each month; preserves the signal's
        # marginal distribution and the target return process.
        arr = []
        for ym, g in q.groupby("ym", sort=False):
            s = g[signal].to_numpy(copy=True)
            rng.shuffle(s)
            arr.extend(zip(g.index, s))
        for idx, v in arr:
            q.loc[idx, signal] = v
        rr = rank_portfolio(q, signal)
        stt = stats(rr)
        if np.isfinite(stt["sharpe"]):
            vals.append(stt["sharpe"])
    if not vals:
        return {"real_sharpe": real_stats["sharpe"], "placebo_mean": np.nan, "placebo_sd": np.nan, "z": np.nan}
    mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    return {"real_sharpe": real_stats["sharpe"], "placebo_mean": mu, "placebo_sd": sd,
            "z": (real_stats["sharpe"] - mu) / sd if sd > 0 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--formation", type=int, default=DEFAULT_FORMATION)
    ap.add_argument("--fit-window", type=int, default=DEFAULT_FIT)
    ap.add_argument("--n-curve", type=int, default=DEFAULT_CURVE)
    ap.add_argument("--cost-bps", type=float, default=10.0,
                    help="per-side cost proxy for diagnostic rank portfolio")
    args = ap.parse_args()

    print("=" * 88)
    print("LOCAL FRONT-END CURVE RESIDUAL — IDENTIFICATION TEST")
    print("=" * 88)
    print(f"formation={args.formation}m  rolling-fit={args.fit_window}m  curve-points={args.n_curve}")
    print()

    df = load(args.prices, args.n_curve)
    print(f"rows={len(df):,}  instruments={df['symbol'].nunique()}  days={df['date'].nunique():,}")

    p = monthly_panel(df, args.formation, args.n_curve)
    p = make_signals(p, args.fit_window)

    print("\n[1] SIGNAL GEOMETRY")
    print(f"corr(BM, local residual): {p[['bm','local_resid']].corr().iloc[0,1]:+.4f}")
    print(f"corr(local residual, local residual after cross-sectional controls): "
          f"{p[['local_resid','local_resid_xs']].corr().iloc[0,1]:+.4f}")

    print("\n[2] PREDICTIVE CROSS-SECTIONAL TESTS")
    for name in ["bm", "front_mom", "remote_mom", "local_resid", "local_resid_xs"]:
        print(f"  {name:20s}", fmb(p, "fwd0", name))

    print("\n[3] KEY SPANNING REGRESSION — DOES THE LOCAL PIECE SURVIVE CONVENTIONAL FACTORS?")
    controls = [
        "front_mom",
        "basis01",
        "remote_mom",
        "front_curv",
        "local_resid",
    ]
    print(fmb_multivariate(p, "fwd0", controls))

    print("\n[4] STRICTER SPANNING — LOCAL RESIDUAL AFTER CROSS-SECTIONAL CONTROLS")
    controls2 = [
        "front_mom",
        "basis01",
        "remote_mom",
        "front_curv",
        "local_resid_xs",
    ]
    print(fmb_multivariate(p, "fwd0", controls2))

    print("\n[5] RANK PORTFOLIOS — DIAGNOSTIC, NOT THE IDENTIFICATION PROOF")
    for sig in ["bm", "local_resid", "local_resid_xs"]:
        r0 = rank_portfolio(p, sig, cost_bps_side=0.0)
        rc = rank_portfolio(p, sig, cost_bps_side=args.cost_bps)
        print(f"  {sig:20s} gross", stats(r0))
        print(f"  {sig:20s} @ {args.cost_bps:g}bp/side", stats(rc))

    print("\n[6] PLACEBO — 200 WITHIN-MONTH SIGNAL SHUFFLES")
    print("  bm:", placebo(p, "bm", n=200, seed=11))
    print("  local residual:", placebo(p, "local_resid", n=200, seed=12))
    print("  local residual XS:", placebo(p, "local_resid_xs", n=200, seed=13))

    print("\n[7] REMOTE-MATURITY COUNTERFACTUALS")
    # The requested novelty should be front-end local. If remote versions work equally well,
    # the story weakens materially.
    for y in ["s01", "s12", "s23", "front_curv", "local_resid"]:
        print(f"  {y:20s}", fmb(p, "fwd0", y))

    print("\n[8] WHAT WOULD COUNT AS A SUCCESS?")
    print("  1) local_resid or local_resid_xs remains positive with a meaningful t-stat after spanning controls.")
    print("  2) The local signal outperforms remote slope/curvature counterfactuals.")
    print("  3) The placebo distribution centers near zero while the real signal is an outlier.")
    print("  4) The effect persists with costs and without changing parameters to maximize Sharpe.")
    print("  5) It is not merely a relabeled BM: local_resid should differ materially from bm.")
    print("\nIf the local residual dies in the spanning regression, DO NOT call the idea novel.\n"
          "That negative result means the conventional curve factors span the purported new component.")


if __name__ == "__main__":
    main()
