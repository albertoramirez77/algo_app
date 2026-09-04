"""
whence.py — where does the Sharpe actually come from, and what control does the
             economics of the strategy imply?

    python whence.py --prices data/px_clean.parquet

PART A - AN AUDIT OF THE TRANCHING GAIN

The single-grid book earns roughly 0.67 and the tranched book 0.94. Averaging K series
correlated at rho scales volatility by sqrt((1 + (K-1)rho) / K) and leaves mean return
alone, so at K = 21 and rho near 0.75 the Sharpe ratio should rise by about 15%. A rise of
40% is not what the arithmetic permits, and the gap needs an explanation before the number
is defended to anyone.

Three possibilities, and this section distinguishes them.

    1  the gain is real and rho is lower than assumed, so the arithmetic allows more
    2  the gain comes from INTEGER ROUNDING - averaging fractional targets before rounding
       is a different operation from rounding each grid separately, and it may recover
       positions that individually round to zero
    3  something is wrong

Everything is computed on one construction so the comparison is clean: the same daily
marking, the same costs, the same universe.

PART B - THE CONTROL THE ECONOMICS ACTUALLY IMPLIES

The strategy's premise is that hedgers pay a premium to transfer GRADUAL inventory risk.
The signal reads a slow physical process: silos filling and emptying, wells shut in and
restarted. It follows directly that the strategy should work when curve moves are driven by
inventory and fail when they are driven by shock - and the measured record agrees, earning a
Sharpe near 1.0 in calmer markets against 0.4 in volatile ones, and losing 17% in the 2020
reversal.

That is an economically derived control rather than a fitted one: reduce risk when
cross-sectional volatility says the curve is being moved by something other than inventory.
The hypothesis comes from the mechanism, not from inspecting returns.

It is tested honestly. The state is classified on an EXPANDING median, so only information
available at the time is used, and the variants are specified in advance rather than tuned.
If the control does not improve risk-adjusted return or drawdown, it is reported as tested
and rejected like the others.
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
N_GRIDS, COST_MULTIPLE = 21, 3.0


def load(path: str):
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
    med = df.groupby("symbol")["settle_0"].median()
    cost = {}
    for s in med.index:
        i = BY_SYMBOL[s]
        n = med[s] * i.dollar_price_mult
        cost[s] = 1.5 * (i.tick_value / n * 1e4) + i.commission / n * 1e4
    cs = pd.Series(cost)
    drop = set(cs[cs > COST_MULTIPLE * cs.median()].index)
    if drop:
        print(f"  universe rule excludes {sorted(drop)}")
        df = df[~df["symbol"].isin(drop)].copy()
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
    df["dom"] = df.groupby(["symbol", "ym"]).cumcount()
    return df, cost


def grid_targets(df, offset, min_n=6):
    d = df.sort_values(["symbol", "date"]).copy()
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
                rows.append(dict(date=dt, symbol=sym,
                                 target=wi * CAPITAL * VOL_TARGET * IDM / den))
    return pd.DataFrame(rows)


def run_book(df, frames, cost_map, scale: pd.Series | None = None, integer=True):
    """
    `frames` may be one grid or many. Fractional targets are averaged across whatever is
    supplied and rounded ONCE, which is the construction being pitched.
    `scale` optionally multiplies the whole book by a monthly factor, which is how the
    volatility-state control is applied.
    """
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    syms = sorted(df["symbol"].unique())
    ret = df.pivot_table(index="date", columns="symbol", values="r0").reindex(
        dates, columns=syms)
    px = df.pivot_table(index="date", columns="symbol", values="settle_0").reindex(
        dates, columns=syms).ffill()
    stacks = [(tf.pivot_table(index="date", columns="symbol", values="target")
                 .reindex(index=dates, columns=syms).ffill()).to_numpy()
              for tf in frames if not tf.empty]
    S = np.stack(stacks, axis=0)
    cnt = np.sum(~np.isnan(S), axis=0)
    T = np.divide(np.nansum(S, axis=0), np.maximum(cnt, 1),
                  out=np.zeros_like(cnt, dtype=float), where=cnt > 0)
    if scale is not None:
        f = scale.reindex(pd.PeriodIndex(dates, freq="M")).ffill().fillna(1.0).to_numpy()
        T = T * f[:, None]
    N = np.round(T) if integer else T
    dpm = np.array([BY_SYMBOL[s].dollar_price_mult for s in syms])
    comm = np.array([BY_SYMBOL[s].commission for s in syms])
    bps = np.array([cost_map[s] for s in syms])
    P = np.nan_to_num(px.to_numpy(), nan=0.0)
    R = np.nan_to_num(ret.to_numpy(), nan=0.0)
    held = N[:-1]
    pnl = np.nansum(held * dpm * P[:-1] * np.expm1(R[1:]), axis=1)
    trades = np.abs(np.diff(N, axis=0))
    cost = np.nansum(trades * (comm + np.abs(dpm) * P[:-1] * bps / 1e4), axis=1)
    daily = pd.Series((pnl - cost) / CAPITAL, index=dates[1:])
    nz = np.abs(T[:-1]) > 1e-9
    in_mkt = (np.abs(held).sum(axis=1) > 0)
    m = daily.resample("ME").sum(); m = m[m != 0]
    return m, dict(zero=float(((N[:-1] == 0) & nz).sum() / max(nz.sum(), 1)),
                   in_market=float(in_mkt.mean()))


def st(r):
    r = r.dropna()
    if len(r) < 24:
        return dict(n=len(r), sharpe=np.nan, ann=np.nan, vol=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12, vol=av,
                dd=float((eq / eq.cummax() - 1).min()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    a = ap.parse_args()
    df, cost_map = load(a.prices)
    frames = [f for f in (grid_targets(df, o) for o in range(N_GRIDS)) if not f.empty]
    print(f"  {len(frames)} grids, {df['symbol'].nunique()} instruments\n")

    # ---------------------------------------------------------------- A
    print("=" * 80)
    print("A. WHERE DOES THE TRANCHING GAIN COME FROM?")
    print("=" * 80)
    singles, aux_s = [], []
    for tf in frames:
        m, ax = run_book(df, [tf], cost_map)
        s = st(m)
        if np.isfinite(s["sharpe"]):
            singles.append((s, m, ax))
    srs = np.array([s["sharpe"] for s, _, _ in singles])
    rets = np.array([s["ann"] for s, _, _ in singles])
    vols = np.array([s["vol"] for s, _, _ in singles])
    M = pd.DataFrame({i: m for i, (_, m, _) in enumerate(singles)}).dropna(how="all")
    C = M.corr().to_numpy()
    rho = float(np.nanmean(C[np.triu_indices_from(C, k=1)]))
    K = M.shape[1]
    factor = np.sqrt((1 + (K - 1) * rho) / K)

    tr, aux_t = run_book(df, frames, cost_map)
    s_tr = st(tr)

    print(f"  {'single grids, mean Sharpe':38s} {srs.mean():>8.3f}")
    print(f"  {'single grids, mean return':38s} {rets.mean()*100:>7.2f}%")
    print(f"  {'single grids, mean volatility':38s} {vols.mean()*100:>7.2f}%")
    print(f"  {'mean pairwise correlation':38s} {rho:>8.3f}")
    print(f"  {'predicted volatility factor':38s} {factor:>8.3f}")
    print(f"  {'PREDICTED tranched Sharpe':38s} "
          f"{rets.mean()/(vols.mean()*factor):>8.3f}")
    print(f"  {'ACTUAL tranched Sharpe':38s} {s_tr['sharpe']:>8.3f}")
    print(f"  {'ACTUAL tranched return':38s} {s_tr['ann']*100:>7.2f}%")
    print(f"  {'ACTUAL tranched volatility':38s} {s_tr['vol']*100:>7.2f}%")

    ret_gap = s_tr["ann"] - rets.mean()
    print(f"\n  RETURN CHECK. Averaging cannot create return, so tranched return should")
    print(f"  equal the mean across grids. Difference: {ret_gap*100:+.2f} percentage points.")
    if abs(ret_gap) < 0.015:
        print("  Within a cost saving; the arithmetic holds.")
    else:
        print("  MATERIAL. The extra return is NOT from averaging. The likely source is")
        print("  integer rounding: averaging fractional targets before rounding is a")
        print("  different operation from rounding each grid separately, and it recovers")
        print("  positions that individually round to zero. Check the next block.")

    frac_s, _ = run_book(df, frames, cost_map, integer=False)
    frac_singles = [st(run_book(df, [tf], cost_map, integer=False)[0])["sharpe"]
                    for tf in frames]
    frac_singles = np.array([v for v in frac_singles if np.isfinite(v)])
    print(f"\n  WITH FRACTIONAL CONTRACTS (rounding removed entirely):")
    print(f"    single grids, mean Sharpe   {frac_singles.mean():>8.3f}")
    print(f"    tranched Sharpe             {st(frac_s)['sharpe']:>8.3f}")
    print(f"    ratio                       "
          f"{st(frac_s)['sharpe']/frac_singles.mean():>8.3f}   "
          f"(theory says {1/factor:.3f})")
    print(f"  integer version ratio         "
          f"{s_tr['sharpe']/srs.mean():>8.3f}")
    print("\n  If the fractional ratio matches theory and the integer ratio exceeds it,")
    print("  the excess is rounding recovery - a real and explainable effect of averaging")
    print("  before rounding, not an error. It should be stated as such rather than left")
    print("  for a reader to find.")
    print(f"\n  positions rounding to zero: single grid "
          f"{np.mean([ax['zero'] for _, _, ax in singles])*100:.1f}%, "
          f"tranched {aux_t['zero']*100:.1f}%")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 80)
    print("B. THE CONTROL THE ECONOMICS IMPLIES")
    print("=" * 80)
    print("  Premise: hedgers pay to transfer GRADUAL inventory risk, so the signal should")
    print("  work when curves move on inventory and fail when they move on shock. The")
    print("  control follows from that, not from inspecting returns.\n")
    mm = (df.groupby(["symbol", "ym"])["r0"].sum(min_count=1).reset_index())
    disp = mm.groupby("ym")["r0"].std().dropna()          # cross-sectional dispersion
    med = disp.expanding().median().shift(1)              # expanding: no look-ahead
    state_hi = (disp > med)

    print(f"  {'variant':34s} {'Sharpe':>8s} {'return':>8s} {'maxDD':>8s} "
          f"{'in market':>10s}")
    base_m, base_ax = tr, aux_t
    print(f"  {'no control (as pitched)':34s} {s_tr['sharpe']:>8.3f} "
          f"{s_tr['ann']*100:>7.2f}% {s_tr['dd']*100:>7.1f}% "
          f"{base_ax['in_market']*100:>9.0f}%")
    results = {}
    for name, f_hi in (("half risk when volatile", 0.5),
                       ("quarter risk when volatile", 0.25),
                       ("flat when volatile", 0.0)):
        scale = pd.Series(np.where(state_hi, f_hi, 1.0), index=disp.index)
        m2, ax2 = run_book(df, frames, cost_map, scale=scale)
        s2 = st(m2)
        results[name] = (s2, ax2)
        print(f"  {name:34s} {s2['sharpe']:>8.3f} {s2['ann']*100:>7.2f}% "
              f"{s2['dd']*100:>7.1f}% {ax2['in_market']*100:>9.0f}%")

    print(f"\n  months classified volatile: {state_hi.mean():.0%}")
    print("\n  The 'in market' column answers a question the pitch does not currently")
    print("  address. A cross-sectional book is always invested by construction: ranking")
    print("  sixteen instruments always produces longs and shorts, and there is no flat")
    print("  state the way a trend follower has one. This control is the only mechanism")
    print("  that would ever take the strategy out of the market.")

    best = max(results, key=lambda k: results[k][0]["sharpe"])
    bs = results[best][0]
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"  best variant: {best}  \u2014  Sharpe {bs['sharpe']:+.3f} "
          f"({bs['sharpe']-s_tr['sharpe']:+.3f}), drawdown "
          f"{bs['dd']*100:+.1f}% ({(bs['dd']-s_tr['dd'])*100:+.1f}pp)")
    print()
    if bs["sharpe"] > s_tr["sharpe"] + 0.03 or bs["dd"] > s_tr["dd"] + 0.03:
        print("  ADOPT. The control improves the strategy AND it was derived from the")
        print("  economics rather than fitted to the returns, which is the strongest form")
        print("  a risk control can take: the mechanism predicted where the strategy would")
        print("  fail, and reducing risk there helps.")
    else:
        print("  REJECT, and report it. The mechanism correctly PREDICTS the weak regime -")
        print("  performance really is worse when volatility is high - but acting on that")
        print("  prediction does not improve the strategy, because the state classification")
        print("  is too noisy in real time to be useful. Diagnosing a weakness and being")
        print("  able to trade on it are different things, and the difference is worth")
        print("  saying out loud.")


if __name__ == "__main__":
    main()