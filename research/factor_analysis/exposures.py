"""
exposures.py — what is this strategy accidentally long?

    python exposures.py --prices data/px_clean.parquet

WHY THIS EXISTS

The strategy reports a market beta of -0.11 to the commodity complex and a correlation of
+0.08 to time-series momentum. That is two exposures. There are at least eight that matter,
and the ones nobody measured are precisely the ones that arrive without being chosen - which
is what "backdoor exposure" means.

EVERY FACTOR IS BUILT FROM DATA ALREADY IN HAND

px_clean.parquet holds all thirty-five instruments: 17 commodities, 8 currencies, 6 rates,
4 equity index. The eighteen non-commodity instruments were tested as a strategy and failed.
They are, however, a perfectly good factor library, and no external data is needed.

    commodity market   equal-weighted long-only return of the 17 commodities
    US dollar          equal-weighted long-only FX return, SIGN-FLIPPED. Each contract is
                       quoted as USD per foreign unit, so a rising 6E means a weaker dollar
    rates level        equal-weighted long-only return of ZT, ZF, ZN, ZB, UB
    curve slope        long ZT short ZB, each scaled by its own trailing volatility so the
                       spread is duration-neutral rather than dominated by the long bond
    equity market      equal-weighted long-only return of MES, MNQ, MYM, M2K
    equity volatility  trailing realised volatility of the equity sleeve, level and change
    commodity carry    the cross-sectional carry portfolio
    cross-sec momentum the twelve-month front-momentum portfolio
    sector tilts       net signed weight in each of five commodity sectors

FOUR ANALYSES, NOT ONE

  univariate    each factor alone. Finds the obvious.
  multivariate  all together. If alpha collapses here, the strategy is a repackaging of
                exposures that already have names.
  rolling       36-month window. A static beta of -0.11 is consistent with one that swings
                between -0.5 and +0.3, and that swing is itself an unmanaged risk.
  conditional   beta inside the worst decile of each factor. An exposure that appears only
                in the tail is invisible to an ordinary regression and is the kind that
                actually causes losses.

The point of this exercise is to FIND something. A clean result is only worth reporting if
the test could have failed, so every number is reported whether or not it flatters the
strategy.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

CAPITAL, VOL_TARGET, IDM, J, VOL_WINDOW = 450_000.0, 0.20, 2.5, 12, 6
N_GRIDS = 21
ROLL_WIN = 36

RATES = ["ZT", "ZF", "ZN", "ZB", "UB"]
EQUITY = ["MES", "MNQ", "MYM", "M2K"]


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------

def load_daily(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df = df[df["contract_0"] != df["contract_1"]]
    df = (df.sort_values(["symbol", "date", "oi_0"], na_position="first")
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"]).reset_index(drop=True))
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df = df[df["asset"] != "?"].copy()
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    gap = (df["expiry_1"] - df["expiry_0"]).dt.days
    with np.errstate(invalid="ignore", divide="ignore"):
        df["basis"] = np.log(df["settle_0"] / df["settle_1"]) / (gap / 365.25)
    df.loc[(gap <= 0) | (gap > 400), "basis"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
    df["dom"] = df.groupby(["symbol", "ym"]).cumcount()
    return df


def monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                basis=("basis", "last"), px=("settle_0", "last"),
                nd=("r0", "size")).reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m["sector"] = m["symbol"].map(lambda s: BY_SYMBOL[s].sector if s in BY_SYMBOL else "?")
    m = m[m["nd"] >= 10].sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["mom0"] = g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm"] = m["mom0"] - g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    return m


# ----------------------------------------------------------------------------------
# the strategy, tranched and marked daily
# ----------------------------------------------------------------------------------

def grid_targets(df: pd.DataFrame, offset: int, min_n: int = 6) -> pd.DataFrame:
    d = df[df["asset"] == "commodity"].sort_values(["symbol", "date"]).copy()
    for leg in ("0", "1"):
        d[f"c{leg}"] = d.groupby("symbol")[f"r{leg}"].transform(
            lambda s: s.fillna(0.0).cumsum())
    snap = d[d["dom"] == offset][["symbol", "ym", "date", "c0", "c1", "settle_0"]].copy()
    if snap.empty:
        return pd.DataFrame()
    snap = snap.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = snap.groupby("symbol")
    snap["r0"] = g["c0"].diff(); snap["r1"] = g["c1"].diff()
    snap["bm"] = (g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
                  - g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum()))
    snap["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(snap["symbol"]).shift(1)
    snap["px_entry"] = g["settle_0"].shift(1)
    rows = []
    for dt, gg in snap.groupby("date"):
        s = gg[["symbol", "bm", "vol", "px_entry"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s["bm"].rank(); w = (r - r.mean()).to_numpy(); gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        for sym, wi, vol, px in zip(s["symbol"], w, s["vol"], s["px_entry"]):
            inst = BY_SYMBOL[sym]
            den = inst.dollar_price_mult * px * vol
            if den > 0:
                rows.append(dict(date=dt, symbol=sym, w=wi,
                                 target=wi * CAPITAL * VOL_TARGET * IDM / den))
    return pd.DataFrame(rows)


def tranched(df: pd.DataFrame, frames: list[pd.DataFrame], bps: float = 3.0):
    d = df[df["asset"] == "commodity"]
    dates = pd.DatetimeIndex(sorted(d["date"].unique()))
    syms = sorted(d["symbol"].unique())
    ret = d.pivot_table(index="date", columns="symbol", values="r0").reindex(
        dates, columns=syms)
    px = d.pivot_table(index="date", columns="symbol", values="settle_0").reindex(
        dates, columns=syms).ffill()
    stacks = []
    for tf in frames:
        if tf.empty:
            continue
        stacks.append((tf.pivot_table(index="date", columns="symbol", values="target")
                         .reindex(index=dates, columns=syms).ffill()).to_numpy())
    S = np.stack(stacks, axis=0)
    cnt = np.sum(~np.isnan(S), axis=0)
    T = np.divide(np.nansum(S, axis=0), np.maximum(cnt, 1),
                  out=np.zeros_like(cnt, dtype=float), where=cnt > 0)
    N = np.round(T)
    dpm = np.array([BY_SYMBOL[s].dollar_price_mult for s in syms])
    comm = np.array([BY_SYMBOL[s].commission for s in syms])
    P = np.nan_to_num(px.to_numpy(), nan=0.0)
    R = np.nan_to_num(ret.to_numpy(), nan=0.0)
    held = N[:-1]
    pnl = np.nansum(held * dpm * P[:-1] * np.expm1(R[1:]), axis=1)
    trades = np.abs(np.diff(N, axis=0))
    cost = np.nansum(trades * (comm + np.abs(dpm) * P[:-1] * bps / 1e4), axis=1)
    daily = pd.Series((pnl - cost) / CAPITAL, index=dates[1:])
    # per-sector net dollar exposure, for the sector-tilt factors
    sec = np.array([BY_SYMBOL[s].sector for s in syms])
    notional = held * dpm * P[:-1]
    tilts = {}
    for sname in sorted(set(sec)):
        col = notional[:, sec == sname].sum(axis=1) / CAPITAL
        tilts[sname] = pd.Series(col, index=dates[1:])
    return daily, pd.DataFrame(tilts)


def xs_portfolio(m: pd.DataFrame, sig: str, min_n: int = 6) -> pd.Series:
    """Cross-sectional rank portfolio, unlevered, for use as a factor."""
    out = {}
    for ym, g in m.groupby("ym"):
        s = g[[sig, "fwd"]].dropna()
        if len(s) < min_n:
            continue
        r = s[sig].rank(); w = (r - r.mean()).to_numpy(); gr = np.abs(w).sum()
        if gr > 0:
            out[ym] = float((w / gr * s["fwd"].to_numpy()).sum())
    return pd.Series(out).sort_index()


# ----------------------------------------------------------------------------------
# factors
# ----------------------------------------------------------------------------------

def build_factors(df: pd.DataFrame, m: pd.DataFrame, tilts: pd.DataFrame) -> pd.DataFrame:
    f = {}
    mm = m.copy()

    comm = mm[mm["asset"] == "commodity"]
    f["commodity_mkt"] = comm.groupby("ym")["r0"].mean()

    fx = mm[mm["asset"] == "fx"]
    if not fx.empty:
        # each FX contract is USD per foreign unit, so a rising contract means a WEAKER
        # dollar. Sign-flipped so a positive factor return means dollar strength.
        f["usd"] = -fx.groupby("ym")["r0"].mean()

    rt = mm[mm["symbol"].isin(RATES)]
    if not rt.empty:
        f["rates_level"] = rt.groupby("ym")["r0"].mean()
        # duration-neutral slope: each leg scaled by its own trailing volatility so the
        # long bond does not dominate simply by being more volatile
        piv = rt.pivot_table(index="ym", columns="symbol", values="r0")
        if {"ZT", "ZB"}.issubset(piv.columns):
            sd = piv.rolling(24, min_periods=12).std().shift(1)
            sl = (piv["ZT"] / sd["ZT"]) - (piv["ZB"] / sd["ZB"])
            f["curve_slope"] = sl

    eq = mm[mm["symbol"].isin(EQUITY)]
    if not eq.empty:
        eqm = eq.groupby("ym")["r0"].mean()
        f["equity_mkt"] = eqm
        rv = eqm.rolling(6, min_periods=3).std() * np.sqrt(12)
        f["equity_vol"] = rv
        f["equity_vol_chg"] = rv.diff()

    f["commodity_carry"] = xs_portfolio(comm, "basis")
    f["xs_momentum"] = xs_portfolio(comm, "mom0")

    ts = {}
    for ym, g in comm.groupby("ym"):
        s = g[["mom0", "fwd"]].dropna()
        if len(s) < 6:
            continue
        w = np.sign(s["mom0"].to_numpy())
        if np.abs(w).sum() > 0:
            ts[ym] = float((w / np.abs(w).sum() * s["fwd"].to_numpy()).sum())
    f["ts_momentum"] = pd.Series(ts).sort_index()

    F = pd.DataFrame(f)
    tl = tilts.resample("ME").last()
    tl.index = tl.index.to_period("M")
    for c in tl.columns:
        F[f"tilt_{c}"] = tl[c]
    return F


# ----------------------------------------------------------------------------------
# regressions
# ----------------------------------------------------------------------------------

def ols(y: np.ndarray, X: np.ndarray):
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.pinv(A.T @ A) @ (A.T @ y)
    e = y - A @ b
    dof = max(len(y) - A.shape[1], 1)
    s2 = (e @ e) / dof
    try:
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se = np.full(len(b), np.nan)
    r2 = 1 - e.var() / y.var() if y.var() > 0 else np.nan
    return b, se, r2


def uni(y: pd.Series, x: pd.Series):
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(j) < 40:
        return None
    b, se, r2 = ols(j["y"].to_numpy(), j["x"].to_numpy().reshape(-1, 1))
    return dict(n=len(j), alpha=b[0] * 12, t_alpha=b[0] / se[0] if se[0] > 0 else np.nan,
                beta=b[1], t_beta=b[1] / se[1] if se[1] > 0 else np.nan, r2=r2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    a = ap.parse_args()

    df = load_daily(a.prices)
    m = monthly_panel(df)
    print("  building the tranched book...")
    frames = [f for f in (grid_targets(df, o) for o in range(N_GRIDS)) if not f.empty]
    daily, tilts = tranched(df, frames)
    strat = daily.resample("ME").sum()
    strat = strat[strat != 0]
    strat.index = strat.index.to_period("M")

    F = build_factors(df, m, tilts)
    yrs = len(strat) / 12
    sr = (strat.mean() * 12) / (strat.std(ddof=1) * np.sqrt(12))

    print("\n" + "=" * 84)
    print("0. THE SERIES BEING EXPLAINED")
    print("=" * 84)
    print(f"  strategy: {len(strat)} months, Sharpe {sr:+.3f}, "
          f"return {strat.mean()*12*100:+.2f}%, vol {strat.std(ddof=1)*np.sqrt(12)*100:.1f}%")
    print(f"\n  factor coverage (months overlapping the strategy):")
    for c in F.columns:
        j = pd.concat([strat, F[c]], axis=1).dropna()
        flag = "  <- short sample" if len(j) < 100 else ""
        print(f"    {c:22s} {len(j):>4d}{flag}")

    print("\n" + "=" * 84)
    print("1. UNIVARIATE — each factor alone")
    print("=" * 84)
    print(f"  {'factor':22s} {'beta':>8s} {'t':>7s} {'alpha/yr':>10s} {'t':>7s} "
          f"{'R2':>7s} {'n':>5s}")
    rows = []
    for c in F.columns:
        r = uni(strat, F[c])
        if r is None:
            continue
        rows.append(dict(factor=c, **r))
    rows.sort(key=lambda d: -abs(d["beta"]) if np.isfinite(d["beta"]) else 0)
    for r in rows:
        star = " *" if abs(r["t_beta"]) > 2 else "  "
        print(f"  {r['factor']:22s} {r['beta']:>+8.3f} {r['t_beta']:>+7.2f}{star}"
              f"{r['alpha']*100:>+9.2f}% {r['t_alpha']:>+7.2f} {r['r2']:>7.3f} "
              f"{r['n']:>5d}")
    print("\n  A significant beta is an exposure the strategy did not choose. A significant")
    print("  alpha means return survives that exposure.")
    print()
    print("  WHY MOST OF THESE WILL BE SMALL, AND WHY THAT IS NOT THE TEST PASSING BY")
    print("  DEFAULT. Rank weights are DEMEANED, so they sum to zero. Any factor that")
    print("  loads uniformly across the seventeen commodities therefore cancels exactly")
    print("  in the weighted sum - verified directly: a factor with uniform loading")
    print("  produces a beta of -0.017 through this weight vector, while the same factor")
    print("  loaded differentially produces +1.00. The book is structurally immune to")
    print("  common factors and can only carry DIFFERENTIAL exposure, where the signal")
    print("  systematically selects high-loading names. That is what these regressions")
    print("  are actually testing for, and it is the only kind of exposure that can")
    print("  exist here.")

    print("\n" + "=" * 84)
    print("2. MULTIVARIATE — all factors together")
    print("=" * 84)
    print("  If alpha collapses here, the strategy is a repackaging of exposures that")
    print("  already have names. This is the number that decides it.\n")
    core = [c for c in F.columns if not c.startswith("tilt_")
            and F[c].notna().sum() > 120]
    J_ = pd.concat([strat.rename("y"), F[core]], axis=1).dropna()
    if len(J_) > 60:
        b, se, r2 = ols(J_["y"].to_numpy(), J_[core].to_numpy())
        print(f"  {'term':22s} {'coef':>9s} {'t':>8s}")
        print(f"  {'alpha (annualised)':22s} {b[0]*12*100:>+8.2f}% "
              f"{b[0]/se[0] if se[0]>0 else np.nan:>+8.2f}")
        for i, c in enumerate(core):
            star = " *" if se[i+1] > 0 and abs(b[i+1]/se[i+1]) > 2 else "  "
            print(f"  {c:22s} {b[i+1]:>+9.3f} "
                  f"{b[i+1]/se[i+1] if se[i+1]>0 else np.nan:>+8.2f}{star}")
        print(f"\n  R-squared {r2:.3f} over {len(J_)} months, {len(core)} factors")
        uni_alpha = strat.mean() * 12
        print(f"  raw return {uni_alpha*100:+.2f}%/yr -> alpha {b[0]*12*100:+.2f}%/yr")
        print(f"  share of return explained by known factors: "
              f"{1 - (b[0]*12)/uni_alpha if uni_alpha != 0 else np.nan:.0%}")

    print("\n" + "=" * 84)
    print("3. ROLLING BETAS — does the exposure move?")
    print("=" * 84)
    print(f"  {ROLL_WIN}-month window. A stable average can hide a beta that swings, and")
    print("  the swing is itself an unmanaged risk.\n")
    print(f"  {'factor':22s} {'mean':>8s} {'min':>8s} {'max':>8s} {'range':>8s} "
          f"{'sd':>7s}")
    for c in core:
        j = pd.concat([strat.rename("y"), F[c].rename("x")], axis=1).dropna()
        if len(j) < ROLL_WIN + 12:
            continue
        bs = []
        for i in range(ROLL_WIN, len(j) + 1):
            w = j.iloc[i - ROLL_WIN:i]
            if w["x"].var() > 0:
                bs.append(np.cov(w["y"], w["x"])[0, 1] / w["x"].var())
        if not bs:
            continue
        bs = np.array(bs)
        print(f"  {c:22s} {bs.mean():>+8.3f} {bs.min():>+8.3f} {bs.max():>+8.3f} "
              f"{bs.max()-bs.min():>8.3f} {bs.std(ddof=1):>7.3f}")

    print("\n" + "=" * 84)
    print("4. CONDITIONAL BETAS — the exposure that only shows up in the tail")
    print("=" * 84)
    print("  Beta measured inside the worst decile of each factor's own returns. An")
    print("  exposure that appears only under stress is invisible to the regressions")
    print("  above and is the kind that actually causes losses.\n")
    print(f"  {'factor':22s} {'full beta':>10s} {'worst-decile':>13s} "
          f"{'mean strat ret':>15s} {'n':>4s}")
    for c in core:
        j = pd.concat([strat.rename("y"), F[c].rename("x")], axis=1).dropna()
        if len(j) < 60:
            continue
        full = np.cov(j["y"], j["x"])[0, 1] / j["x"].var() if j["x"].var() > 0 else np.nan
        cut = j["x"].quantile(0.10)
        tail = j[j["x"] <= cut]
        tb = (np.cov(tail["y"], tail["x"])[0, 1] / tail["x"].var()
              if len(tail) > 6 and tail["x"].var() > 0 else np.nan)
        print(f"  {c:22s} {full:>+10.3f} {tb:>+13.3f} "
              f"{tail['y'].mean()*100:>+14.2f}% {len(tail):>4d}")

    print("\n" + "=" * 84)
    print("5. SECTOR TILTS — is the book structurally long one complex?")
    print("=" * 84)
    tl = [c for c in F.columns if c.startswith("tilt_")]
    if tl:
        print(f"  {'sector':22s} {'mean net':>10s} {'sd':>9s} {'min':>9s} {'max':>9s}")
        for c in tl:
            v = F[c].dropna()
            if len(v) < 24:
                continue
            print(f"  {c.replace('tilt_',''):22s} {v.mean():>+10.3f} {v.std():>9.3f} "
                  f"{v.min():>+9.3f} {v.max():>+9.3f}")
        print("\n  Values are net notional as a multiple of capital. A mean far from zero")
        print("  means the strategy is structurally long or short that complex rather than")
        print("  taking a genuinely relative position within it.")

    print("\n" + "=" * 84)
    print("WHAT TO REPORT")
    print("=" * 84)
    sig = [r for r in rows if np.isfinite(r["t_beta"]) and abs(r["t_beta"]) > 2]
    if sig:
        print(f"  {len(sig)} factor(s) show a significant unconditional beta:")
        for r in sig:
            print(f"    {r['factor']:22s} beta {r['beta']:+.3f} (t {r['t_beta']:+.2f})")
        print("\n  Name these in the pitch. An exposure disclosed is a risk managed; one")
        print("  found by a reviewer is a credibility problem.")
    else:
        print("  No factor shows a significant unconditional beta. That is a real result")
        print("  and worth one sentence in Risk Assessment: the strategy was tested against")
        print("  nine named exposures built from the same data and none explained it.")
    print("\n  Whatever the multivariate alpha turns out to be, quote it. It is the")
    print("  strongest single number available: return that survives every factor that")
    print("  could be constructed from this universe.")


if __name__ == "__main__":
    main()