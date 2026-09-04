"""
test_curve.py — step 3. Three questions, one dataset, placebos included.

    python test_curve.py --prices data/px_clean.parquet

WHAT THIS ANSWERS

  0  Data hygiene. Duplicate dates land on roll days, which are exactly the dates the
     delivery test depends on. They are found and removed before anything is measured.

  1  POSITIVE CONTROL — does the basis predict returns?
     This is the control step 2 could not deliver. A 12-month momentum signal needs 52
     weeks of history and our sample starts 2010-06, so the pre-2011 window was empty
     before the test began. The basis needs one lag. Szymanowska, de Roon, Nijman and van
     den Goorbergh (JF 2014) report spot premia of 5-14% a year from basis sorts over
     1986-2010 — the strongest published commodity predictor, on a sample ending exactly
     where ours starts.
     If the basis works here, the alignment machinery is validated and the null on hedger
     net trading is real. If the basis fails too, something upstream is wrong.

  2  KILL TEST A — the delivery cycle.
     Event-time profile of the front-versus-second spread against days to expiry. The
     hypothesis is that longs who cannot take delivery are forced out on a published
     calendar, depressing the front relative to the second.

     THE CONFOUND, AND HOW IT IS HANDLED: the front converges toward the second as expiry
     approaches, mechanically, because of the basis. A raw profile would rediscover carry
     and mislabel it forced selling. So the profile is also computed on spread returns
     with the basis effect regressed out. Only the residual profile is evidence of flow.

  3  KILL TEST B — time-series carry.
     Non-overlapping monthly observations, within-instrument, date-clustered standard
     errors. Non-overlapping because 21-day forward returns sampled daily are serially
     dependent and would inflate every t-statistic in sight.

  4  PLACEBOS for both, run in the same script rather than after it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

MIN_DTE, MAX_DTE = 0, 90


# ----------------------------------------------------------------------------------
# hygiene
# ----------------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("date", "expiry_0"):
        df[c] = pd.to_datetime(df[c])
    n0 = len(df)

    bad_same = (df["contract_0"] == df["contract_1"]).sum()
    df = df[df["contract_0"] != df["contract_1"]]

    dup = df.duplicated(["date", "symbol"], keep=False).sum()
    # On a roll date the continuous mapping can emit both the outgoing and incoming
    # instrument. .n.0 is defined by open-interest rank, so the row with the larger
    # open interest is the one the series actually resolves to.
    # na_position="first" is load-bearing. pandas sorts NaN LAST by default, and oi_0 is
    # NaN wherever the open-interest record carries no session key, so keep="last" was
    # selecting the leg WITHOUT open interest — the exact opposite of the intent, on roll
    # dates, which is what the delivery test depends on.
    df = (df.sort_values(["symbol", "date", "oi_0"], na_position="first")
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"]).reset_index(drop=True))

    print("=" * 78)
    print("0. DATA HYGIENE")
    print("=" * 78)
    print(f"  rows in {n0:,}  ->  {len(df):,} out")
    print(f"  dropped {bad_same:,} rows where the two legs were the same contract")
    print(f"  resolved {dup:,} duplicate (date, symbol) rows, kept the higher-OI leg")
    per = df.groupby("symbol")["date"].agg(["size", "min", "max"])
    yrs = (per["max"] - per["min"]).dt.days / 365.25
    per["days_per_year"] = (per["size"] / yrs).round(0)
    print(f"\n  trading days per year by product (expect ~250):")
    print("   ", "  ".join(f"{s}:{int(v)}" for s, v in per["days_per_year"].items()))
    off = per[(per["days_per_year"] < 235) | (per["days_per_year"] > 265)]
    if len(off):
        print(f"  STILL OFF: {list(off.index)} — investigate before trusting results")
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # within-contract returns for each leg; never chain across a contract change
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        df[f"blk_{leg}"] = blk
        df[f"prev_{leg}"] = df.groupby(["symbol", f"blk_{leg}"])[f"settle_{leg}"].shift(1)
        df[f"ret_{leg}"] = df[f"settle_{leg}"] / df[f"prev_{leg}"] - 1.0

    # long second / short front. Positive when the front underperforms.
    df["spread_ret"] = df["ret_1"] - df["ret_0"]
    # log basis. Positive = backwardation = front above second.
    df["basis"] = np.log(df["settle_0"] / df["settle_1"])
    df["dte"] = (df["expiry_0"] - df["date"]).dt.days
    return df


def sanity(df: pd.DataFrame) -> None:
    b = df["basis"].dropna()
    print(f"\n  basis: median {b.median():+.4f}  sd {b.std():.4f}  "
          f"|basis|>0.25 in {(b.abs() > 0.25).mean():.2%} of rows")
    print(f"  share of days in backwardation: {(b > 0).mean():.1%}")
    d = df["dte"].dropna()
    print(f"  days to front expiry: median {d.median():.0f}  "
          f"p5 {d.quantile(.05):.0f}  p95 {d.quantile(.95):.0f}")
    s = df["spread_ret"].dropna()
    print(f"  spread return: sd {s.std()*100:.3f}%/day  "
          f"|>10%| in {(s.abs() > 0.10).mean():.3%} of rows")


# ----------------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------------

def cluster_ols(y: np.ndarray, x: np.ndarray, cl: np.ndarray) -> tuple[float, float, int]:
    """Univariate OLS with an intercept, standard errors clustered on `cl`."""
    ok = np.isfinite(y) & np.isfinite(x)
    y, x, cl = y[ok], x[ok], cl[ok]
    if len(y) < 50:
        return np.nan, np.nan, len(y)
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    meat = np.zeros((2, 2))
    for g in np.unique(cl):
        m = cl == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(max(V[1, 1], 0.0))
    # Relative, not exact. A collinear fit drives se to ~1e-17 rather than exactly zero,
    # which returned a t-statistic around 1e14 instead of NaN.
    if not np.isfinite(se) or se <= np.finfo(float).eps * max(abs(beta[1]), 1.0):
        return beta[1], np.nan, len(y)
    return beta[1], beta[1] / se, len(y)


def demean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby("symbol")[c].transform("mean")
    return out


def monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping month-end basis with next-month front-contract return."""
    d = df.dropna(subset=["basis"]).copy()
    d["ym"] = d["date"].dt.to_period("M")
    last = d.groupby(["symbol", "ym"]).tail(1)[["symbol", "ym", "date", "basis"]]
    r = (d.assign(r=d["ret_0"].fillna(0.0))
           .groupby(["symbol", "ym"])["r"]
           .apply(lambda s: np.expm1(np.log1p(s).sum())).rename("mret").reset_index())
    r["ym_prev"] = r["ym"] - 1
    m = last.merge(r[["symbol", "ym_prev", "mret"]],
                   left_on=["symbol", "ym"], right_on=["symbol", "ym_prev"], how="inner")
    return m.dropna(subset=["basis", "mret"])


# ----------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------

def basis_control(df: pd.DataFrame, seeds: int) -> bool:
    print("\n" + "=" * 78)
    print("1. POSITIVE CONTROL / KILL TEST B — does the basis predict returns?")
    print("=" * 78)
    m = monthly_panel(df)
    print(f"  {len(m):,} non-overlapping instrument-months, "
          f"{m['symbol'].nunique()} products, {m['ym'].nunique()} months")

    w = demean(m, ["basis", "mret"])
    b, t, n = cluster_ols(w["mret"].to_numpy(), w["basis"].to_numpy(),
                          w["ym"].astype(str).to_numpy())
    print(f"\n  within-instrument, date-clustered:  slope {b:+.4f}   t {t:+.2f}   n {n:,}")
    print(f"  expected sign: POSITIVE (backwardation predicts positive returns)")

    z = m.copy()
    z["bz"] = z.groupby("symbol")["basis"].transform(lambda s: (s - s.mean()) / s.std())
    zb, zt, _ = cluster_ols(z["mret"].to_numpy(), z["bz"].to_numpy(),
                            z["ym"].astype(str).to_numpy())
    print(f"  standardised basis:                 slope {zb:+.4f}   t {zt:+.2f}")
    print(f"  -> a one-sd move in basis shifts next-month return by "
          f"{zb*100:+.2f}%  ({zb*12*100:+.1f}%/yr)")

    rng = np.random.default_rng(0)
    ts = []
    for _ in range(seeds):
        p = m.copy()
        p["basis"] = p.groupby("symbol")["basis"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        pw = demean(p, ["basis", "mret"])
        _, pt, _ = cluster_ols(pw["mret"].to_numpy(), pw["basis"].to_numpy(),
                               pw["ym"].astype(str).to_numpy())
        ts.append(pt)
    ts = np.array(ts)
    print(f"\n  placebo (basis shuffled in time, within instrument): "
          f"t = {ts.mean():+.2f} ± {ts.std(ddof=1):.2f} over {seeds} seeds")

    passed = np.isfinite(t) and t > 2 and b > 0
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
    if passed:
        print("  The panel detects the strongest published commodity predictor with the")
        print("  right sign. Alignment is validated, the hedger-flow null is real, and")
        print("  carry is live in this universe as a candidate in its own right.")
    else:
        print("  The most robust documented effect in commodity futures does not appear.")
        print("  Do NOT interpret any other null from this panel until that is explained.")
    return passed


def delivery_profile(df: pd.DataFrame, seeds: int) -> None:
    print("\n" + "=" * 78)
    print("2. KILL TEST A — the delivery cycle")
    print("=" * 78)
    d = df.dropna(subset=["spread_ret", "dte", "basis"]).copy()
    d = d[(d["dte"] >= MIN_DTE) & (d["dte"] <= MAX_DTE)]
    print(f"  {len(d):,} instrument-days inside {MIN_DTE}-{MAX_DTE} days to expiry")

    # Residualise on the basis. Convergence toward the second contract is mechanical;
    # only what survives that is evidence of forced flow.
    ok = np.isfinite(d["spread_ret"]) & np.isfinite(d["basis"])
    X = np.column_stack([np.ones(ok.sum()), d.loc[ok, "basis"].to_numpy()])
    yv = d.loc[ok, "spread_ret"].to_numpy()
    coef = np.linalg.pinv(X.T @ X) @ (X.T @ yv)
    d.loc[ok, "resid"] = yv - X @ coef
    print(f"  basis explains the spread return with slope {coef[1]:+.4f} "
          f"— removed before profiling")

    buckets = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 45), (46, 90)]
    print(f"\n  {'days to expiry':>16s} {'raw':>22s} {'basis-residual':>22s}   n")
    rows = []
    for lo, hi in buckets:
        s = d[(d["dte"] >= lo) & (d["dte"] <= hi)]
        if len(s) < 100:
            continue
        cl = s["date"].astype(str).to_numpy()
        for col, out in (("spread_ret", "raw"), ("resid", "res")):
            v = s[col].to_numpy()
            b, t, n = cluster_ols(v, np.ones(len(v)) * 0 + 1e-12, cl)
        # mean with date-clustered t: regress on a constant via group means
        def mean_t(col):
            g = s.groupby("date")[col].mean().dropna()
            return (g.mean() * 100, g.mean() / (g.std(ddof=1) / np.sqrt(len(g))), len(g))
        rm, rt, nd = mean_t("spread_ret")
        em, et, _ = mean_t("resid")
        rows.append((lo, hi, rm, rt, em, et, nd))
        print(f"  {lo:>6d}-{hi:<3d}      {rm:>+9.4f}%/d t={rt:>+5.2f}   "
              f"{em:>+9.4f}%/d t={et:>+5.2f}   {nd:,}")

    near = d[d["dte"] <= 10]
    far = d[(d["dte"] >= 31) & (d["dte"] <= 90)]
    gn = near.groupby("date")["resid"].mean().dropna()
    gf = far.groupby("date")["resid"].mean().dropna()
    if len(gn) > 30 and len(gf) > 30:
        diff = gn.mean() - gf.mean()
        se = np.sqrt(gn.var(ddof=1) / len(gn) + gf.var(ddof=1) / len(gf))
        print(f"\n  near (<=10d) minus far (31-90d), basis-residualised: "
              f"{diff*100:+.4f}%/day  t={diff/se:+.2f}")

    rng = np.random.default_rng(1)
    ds = []
    for _ in range(seeds):
        p = d.copy()
        p["dte"] = p.groupby("symbol")["dte"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        n_ = p[p["dte"] <= 10].groupby("date")["resid"].mean().dropna()
        f_ = p[(p["dte"] >= 31) & (p["dte"] <= 90)].groupby("date")["resid"].mean().dropna()
        if len(n_) > 30 and len(f_) > 30:
            se_ = np.sqrt(n_.var(ddof=1) / len(n_) + f_.var(ddof=1) / len(f_))
            ds.append((n_.mean() - f_.mean()) / se_)
    if ds:
        ds = np.array(ds)
        print(f"  placebo (days-to-expiry shuffled): t = {ds.mean():+.2f} "
              f"± {ds.std(ddof=1):.2f} over {len(ds)} seeds")

    print("\n  READ IT: a forced pre-delivery exit should make the basis-residual profile")
    print("  RISE as expiry approaches, with near-minus-far positive and t > 2. If the")
    print("  raw profile slopes but the residual is flat, you have found convergence,")
    print("  which is carry, and candidate 1 is dead.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    df = enrich(load(a.prices))
    sanity(df)
    ok = basis_control(df, a.seeds)
    delivery_profile(df, a.seeds)

    print("\n" + "=" * 78)
    print("WHERE THIS LEAVES US")
    print("=" * 78)
    if not ok:
        print("  The control failed. Nothing else here is interpretable. Debug the panel.")
    else:
        print("  Control passed, so both kill tests are interpretable.")
        print("  Delivery residual profile rising with t > 2  -> candidate 1 survives,")
        print("      build the specification around an FND-dated calendar.")
        print("  Delivery flat, basis strong  -> candidate 1 dead, carry lives. Pitch")
        print("      carry honestly as carry, with the failed hypothesis documented.")
        print("  Both flat -> the universe is the problem, not the hypothesis. Widen it.")


if __name__ == "__main__":
    main()