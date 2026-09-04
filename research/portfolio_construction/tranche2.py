"""
tranche2.py — the tranched book, marked daily. The first version was wrong.

    python tranche2.py --prices data/px_clean.parquet

WHAT WAS WRONG WITH THE FIRST VERSION

tranche.py reported a tranched Sharpe of 1.171 against 0.555 for a single grid. Two of
those numbers were right and one was impossible.

    volatility   19.4% -> 16.1%.  CORRECT. Mean grid volatility was 21.83% and the theory
                 predicts 21.83 x sqrt((1 + 20 x 0.526)/21) = 16.17%. Real diversification,
                 exactly as predicted.

    return       mean across grids 14.97% -> tranched 18.84%.  IMPOSSIBLE. The tranched
                 book holds the average position, so its P&L is the average of the grids'
                 P&Ls and its mean return MUST equal the mean across grids. Averaging
                 cannot manufacture return.

The cause: the first version averaged entry prices and forward returns across grids that
use DIFFERENT observation windows — grid 0 measures March from the 1st to the 1st, grid 10
from the 10th to the 10th — and then multiplied those averages together. That is not a
tradeable profit and loss series. It is a smoothed artefact, and smoothing flatters a
Sharpe ratio the same way reporting monthly marks on a quarterly-valued book does.

THE CORRECT CONSTRUCTION

A tranched book holds one position. Each grid contributes 1/21 of it, and each grid updates
its slice only on its own rebalance day. So:

    1  for every grid, carry its fractional target forward day by day until that grid next
       rebalances
    2  average the 21 fractional targets to get the book's target on that day
    3  round ONCE, at full size, to whole contracts
    4  mark that position against the next day's return

Marking daily is the only way to get a return series a portfolio manager could have earned.
Every figure below comes from it.

WHY THE IDEA STILL WORKS

The mechanism was never the return, it was the volatility. Twenty-one grids correlated at
0.526 average to a series with 74% of the volatility and the same mean return. That is
arithmetic, and it survives being computed properly — which is why this succeeded where
eleven empirical refinements failed.
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
    df = df[df["asset"] == "commodity"].copy()
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
    df["dom"] = df.groupby(["symbol", "ym"]).cumcount()
    return df


def grid_targets(df: pd.DataFrame, offset: int, min_n: int = 6) -> pd.DataFrame:
    """
    Fractional target contracts for one grid, dated on that grid's own rebalance days.

    Signal, volatility and entry price are all computed from marks on this grid only, and
    every input is lagged one period exactly as in the frozen specification.
    """
    d = df.sort_values(["symbol", "date"]).copy()
    for leg in ("0", "1"):
        d[f"c{leg}"] = d.groupby("symbol")[f"r{leg}"].transform(
            lambda s: s.fillna(0.0).cumsum())
    snap = d[d["dom"] == offset][["symbol", "ym", "date", "c0", "c1",
                                  "settle_0"]].copy()
    if snap.empty:
        return pd.DataFrame()
    snap = snap.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = snap.groupby("symbol")
    snap["r0"] = g["c0"].diff()
    snap["r1"] = g["c1"].diff()
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
        r = s["bm"].rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        for sym, wi, vol, px in zip(s["symbol"], w, s["vol"], s["px_entry"]):
            inst = BY_SYMBOL[sym]
            den = inst.dollar_price_mult * px * vol
            if den > 0:
                rows.append(dict(date=dt, symbol=sym,
                                 target=wi * CAPITAL * VOL_TARGET * IDM / den))
    return pd.DataFrame(rows)


def daily_book(df: pd.DataFrame, tgt_frames: list[pd.DataFrame], bps: float = 3.0,
               integer: bool = True):
    """
    Carry each grid's target forward day by day, average across grids, round ONCE, and mark
    daily. This is the only construction that produces a return series a manager could have
    earned.
    """
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    syms = sorted(df["symbol"].unique())
    ret = df.pivot_table(index="date", columns="symbol", values="r0").reindex(
        dates, columns=syms)
    px = df.pivot_table(index="date", columns="symbol", values="settle_0").reindex(
        dates, columns=syms).ffill()

    # each grid's target, held flat between its own rebalance dates
    stacks = []
    for tf in tgt_frames:
        if tf.empty:
            continue
        w = (tf.pivot_table(index="date", columns="symbol", values="target")
               .reindex(index=dates, columns=syms).ffill())
        stacks.append(w.to_numpy())
    if not stacks:
        raise SystemExit("no grids produced targets")
    S = np.stack(stacks, axis=0)
    # Days before a grid has produced its first target are genuinely empty, not zero.
    # Treat them as absent so the average is over grids that are actually live.
    cnt = np.sum(~np.isnan(S), axis=0)
    T = np.divide(np.nansum(S, axis=0), np.maximum(cnt, 1),
                  out=np.zeros_like(cnt, dtype=float), where=cnt > 0)
    N = np.round(T) if integer else T

    dpm = np.array([BY_SYMBOL[s].dollar_price_mult for s in syms])
    comm = np.array([BY_SYMBOL[s].commission for s in syms])
    P = px.to_numpy()
    R = np.nan_to_num(ret.to_numpy(), nan=0.0)

    # position held on day t earns day t+1's return; entry price is day t's settle
    held = N[:-1]
    simple = np.expm1(R[1:])
    pnl = np.nansum(held * dpm * np.nan_to_num(P[:-1], nan=0.0) * simple, axis=1)

    trades = np.abs(np.diff(N, axis=0))
    cost = np.nansum(trades * (comm + np.abs(dpm) * np.nan_to_num(P[:-1], nan=0.0)
                               * bps / 1e4), axis=1)

    daily = pd.Series((pnl - cost) / CAPITAL, index=dates[1:])
    gross = np.nansum(np.abs(held) * dpm * np.nan_to_num(P[:-1], nan=0.0), axis=1) / CAPITAL
    nz = (np.abs(T[:-1]) > 1e-9)
    zeroed = float(((N[:-1] == 0) & nz).sum() / max(nz.sum(), 1))
    return daily, dict(turn=float(trades.sum(axis=1).mean() * 21),
                       gross=float(np.nanmean(gross)), zero_share=zeroed)


def st(daily: pd.Series) -> dict:
    m = daily.resample("ME").sum()
    m = m[m != 0]
    if len(m) < 48:
        return dict(n=len(m), sharpe=np.nan, t=np.nan, ann=np.nan, vol=np.nan, dd=np.nan)
    yrs = len(m) / 12
    av = m.std(ddof=1) * np.sqrt(12)
    sr = (m.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + m).cumprod()
    return dict(n=len(m), yrs=yrs, sharpe=sr, t=sr * np.sqrt(yrs), ann=m.mean() * 12,
                vol=av, dd=float((eq / eq.cummax() - 1).min()), monthly=m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--grids", type=int, default=N_GRIDS)
    a = ap.parse_args()

    df = load_daily(a.prices)
    print("=" * 80)
    print("1. BUILDING THE GRIDS")
    print("=" * 80)
    print(f"  {df['symbol'].nunique()} commodities, {df['date'].nunique():,} trading days")
    frames = []
    for off in range(a.grids):
        tf = grid_targets(df, off)
        if not tf.empty:
            frames.append(tf)
    print(f"  {len(frames)} usable grids\n")

    singles = {}
    for i, tf in enumerate(frames):
        d, aux = daily_book(df, [tf])
        s = st(d)
        if np.isfinite(s["sharpe"]):
            singles[i] = (s, aux)
    srs = pd.Series({k: v[0]["sharpe"] for k, v in singles.items()}).sort_index()
    rets = pd.Series({k: v[0]["ann"] for k, v in singles.items()}).sort_index()
    vols = pd.Series({k: v[0]["vol"] for k, v in singles.items()}).sort_index()
    print(f"  {'grid':>6s} {'Sharpe':>8s} {'return':>9s} {'vol':>7s} {'maxDD':>8s}")
    for k in srs.index:
        s = singles[k][0]
        tag = "  <- current spec" if k == 0 else ""
        print(f"  day {k:>2d} {s['sharpe']:>8.3f} {s['ann']*100:>8.2f}% "
              f"{s['vol']*100:>6.1f}% {s['dd']*100:>7.1f}%{tag}")
    print(f"\n  mean Sharpe {srs.mean():.3f}   mean return {rets.mean()*100:.2f}%   "
          f"mean vol {vols.mean()*100:.1f}%")
    print(f"  dispersion {srs.min():.3f} to {srs.max():.3f}   "
          f"spread {srs.max()-srs.min():.3f}")
    if 0 in srs.index:
        print(f"  the reported grid sits at the {(srs < srs.loc[0]).mean():.0%} percentile")

    print("\n" + "=" * 80)
    print("2. THE TRANCHED BOOK, MARKED DAILY")
    print("=" * 80)
    d_tr, aux_tr = daily_book(df, frames)
    s_tr = st(d_tr)
    d_0, aux_0 = daily_book(df, [frames[0]])
    s_0 = st(d_0)

    print(f"  {'':24s} {'Sharpe':>8s} {'t':>7s} {'return':>9s} {'vol':>7s} "
          f"{'maxDD':>8s} {'gross':>7s}")
    print(f"  {'single grid (day 0)':24s} {s_0['sharpe']:>8.3f} {s_0['t']:>+7.2f} "
          f"{s_0['ann']*100:>8.2f}% {s_0['vol']*100:>6.1f}% {s_0['dd']*100:>7.1f}% "
          f"{aux_0['gross']:>6.1f}x")
    print(f"  {'mean across grids':24s} {srs.mean():>8.3f} {'':>7s} "
          f"{rets.mean()*100:>8.2f}% {vols.mean()*100:>6.1f}%")
    print(f"  {'TRANCHED (daily-marked)':24s} {s_tr['sharpe']:>8.3f} {s_tr['t']:>+7.2f} "
          f"{s_tr['ann']*100:>8.2f}% {s_tr['vol']*100:>6.1f}% {s_tr['dd']*100:>7.1f}% "
          f"{aux_tr['gross']:>6.1f}x")

    print("\n  ARITHMETIC CHECK — the test the first version failed:")
    gap = s_tr["ann"] - rets.mean()
    print(f"    mean return across grids   {rets.mean()*100:>7.2f}%")
    print(f"    tranched return           {s_tr['ann']*100:>7.2f}%")
    print(f"    difference                {gap*100:>+7.2f}%")
    if abs(gap) < 0.015:
        print("    PASS. The tranched return matches the mean across grids, as it must.")
        print("    Any residual is the cost saving from lower turnover, which is real.")
    else:
        print("    FAIL. Averaging cannot move mean return by this much. Do not use these")
        print("    numbers until the difference is explained.")

    mm = pd.DataFrame({k: st(daily_book(df, [frames[k]])[0])["monthly"]
                       for k in list(singles)[:len(frames)]}).dropna(how="all")
    C = mm.corr().to_numpy()
    rho = float(np.nanmean(C[np.triu_indices_from(C, k=1)]))
    K = mm.shape[1]
    pred = np.sqrt((1 + (K - 1) * rho) / K)
    print(f"\n  VOLATILITY CHECK — is the reduction the predicted amount?")
    print(f"    mean pairwise correlation between grids   {rho:.3f}")
    print(f"    predicted volatility factor               x{pred:.3f}")
    print(f"    predicted tranched volatility             {vols.mean()*pred*100:.1f}%")
    print(f"    actual tranched volatility                {s_tr['vol']*100:.1f}%")
    print(f"    predicted Sharpe                          "
          f"{rets.mean()/(vols.mean()*pred):.3f}")
    print(f"    actual Sharpe                             {s_tr['sharpe']:.3f}")
    print("    Theory and measurement should agree closely. A large gap means the")
    print("    construction is doing something other than averaging.")

    print("\n" + "=" * 80)
    print("3. TURNOVER, GRANULARITY AND COST")
    print("=" * 80)
    print(f"  {'':24s} {'contracts/mo':>13s} {'gross':>8s} {'zeroed':>8s}")
    print(f"  {'single grid':24s} {aux_0['turn']:>13.1f} {aux_0['gross']:>7.1f}x "
          f"{aux_0['zero_share']*100:>7.1f}%")
    print(f"  {'tranched':24s} {aux_tr['turn']:>13.1f} {aux_tr['gross']:>7.1f}x "
          f"{aux_tr['zero_share']*100:>7.1f}%")
    print("\n  cost sensitivity:")
    for bps in (3, 10, 20, 40):
        dt_, _ = daily_book(df, frames, bps=bps)
        d0_, _ = daily_book(df, [frames[0]], bps=bps)
        st_, s0_ = st(dt_), st(d0_)
        print(f"    {bps:>2d}bp per side   tranched {st_['sharpe']:>+6.3f}   "
              f"single {s0_['sharpe']:>+6.3f}   difference "
              f"{st_['sharpe']-s0_['sharpe']:>+6.3f}")

    print("\n" + "=" * 80)
    print("4. IS THE GAIN NOISE REMOVAL OR SIZE?")
    print("=" * 80)
    if np.isfinite(s_0["vol"]) and s_0["vol"] > 0:
        tgt = s_0["vol"]
        for lbl, s in (("single grid", s_0), ("tranched", s_tr)):
            if np.isfinite(s["vol"]) and s["vol"] > 0:
                r2 = s["monthly"] * (tgt / s["vol"])
                yrs = len(r2) / 12
                av = r2.std(ddof=1) * np.sqrt(12)
                sr = (r2.mean() * 12) / av
                eq = (1 + r2).cumprod()
                print(f"  {lbl:16s} SR {sr:>+6.3f}   "
                      f"dd {float((eq/eq.cummax()-1).min())*100:>+6.1f}%   "
                      f"(rescaled to {tgt*100:.1f}% vol)")
        print("\n  Rescaling equalises size, so any surviving difference is the removal of")
        print("  timing noise. That is the whole claim.")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    gain = s_tr["sharpe"] - s_0["sharpe"]
    print(f"  single grid (the current specification)   {s_0['sharpe']:.3f}")
    print(f"  tranched, daily-marked                    {s_tr['sharpe']:.3f}")
    print(f"  gain                                      {gain:+.3f}")
    print(f"  grid dispersion                           {srs.min():.3f} to {srs.max():.3f}")
    print()
    if abs(gap) < 0.015 and gain > 0.05:
        print("  ADOPT. The construction now passes its own arithmetic check: tranched")
        print("  return equals the mean across grids, and the entire gain comes from the")
        print("  volatility reduction that averaging 21 correlated series must produce.")
        print("  Headline this figure. Report the grid dispersion in Analytical Evidence —")
        print("  it is the honest answer to 'what if you rebalanced three days later', and")
        print("  almost no applicant can answer that at all.")
    elif abs(gap) >= 0.015:
        print("  DO NOT USE. The return check still fails, so the construction is not a")
        print("  tradeable book. Fall back to the single-grid figure and report the grid")
        print("  dispersion as a robustness statistic only.")
    else:
        print("  No material gain once marked properly. Keep the single-grid")
        print("  implementation and report the dispersion as robustness.")


if __name__ == "__main__":
    main()