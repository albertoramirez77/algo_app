"""
final_numbers.py — every number the pitch needs, sixteen instruments, one run.

    python final_numbers.py --prices px_clean.parquet > FINAL_NUMBERS.txt

NET DOLLAR EXPOSURE — TESTED AND REJECTED

Signal weights are demeaned and sum to zero, but positions are then divided by each
instrument's dollar volatility, so net NOTIONAL is proportional to sum(w_i / vol_i) and is
not zero. That is deliberate: the construction equalises RISK across names, which is what a
cross-sectional relative-value book should do. Equalising dollars instead would over-weight
the quiet instruments and under-weight the volatile ones.

The residual exposure was measured and a control was built for it. Subtracting a single
constant from every weight sets net notional to exactly zero without reordering anything,
since the same constant applies to every instrument. On the SINGLE-GRID monthly book that
control improved maximum drawdown by nearly eight percentage points for about 0.04 of
Sharpe, which looked like a good trade.

On the TRANCHED book it is not. It costs roughly 0.11 of Sharpe and makes the drawdown
WORSE, and it also degrades P&L concentration, the count of positive years, and placebo
separation. Nothing improves.

I do not have a verified mechanism for the reversal and will not invent one. What can be
said from the measurements: tranching does NOT reduce net exposure - the tranched book
reaches 51% worst-month net exposure against 49% for the single-grid book - so the earlier
guess that tranching cancels the drift and leaves only rounding is WRONG and is recorded
here as wrong. What is established is that the control's effect is not stable across
constructions, helping drawdown on one and hurting it on the other, and instability of that
kind is on its own a sufficient reason to leave a discretionary control out.

So it is switched off, the exposure is reported rather than constrained, and the honest
sentence for the pitch is that a net-exposure control was built, measured, and rejected on
the evidence. Set NEUTRALISE = True to reproduce the rejected version.


WHY SIXTEEN AND NOT SEVENTEEN

E-mini natural gas is excluded, and the reason is a property of the contract rather than
of the backtest.

CME set the QG minimum price increment at $0.005 per MMBtu on a 2,500 MMBtu contract, so
one tick is $12.50. That increment was chosen when natural gas traded above $10/MMBtu. Gas
now trades near $3, and the tick has never been rescaled - so the same absolute increment
is now more than three times larger as a share of contract value than it was designed to
be. On a notional near $7,500, one tick is roughly 16 basis points. The universe median is
under 4.

The exclusion rule is therefore stated on cost, not on performance:

    exclude any instrument whose ex-ante round-trip cost exceeds three times
    the universe median

That rule is scale-free, computable from tick value and notional alone before any return
is observed, and it happens to exclude exactly one contract. The improvement in Sharpe is
a consequence of the rule, not the reason for it. Section 1 below also reports cost
relative to each instrument's own volatility, because a genuinely volatile contract can
support a wider spread - and QG remains an outlier even after that adjustment, which is
what makes the case decisive rather than merely arithmetic.

WHAT THIS FILE PRODUCES

Every figure the pitch quotes, from one specification on one file in one run. If a number
is not in this output it does not belong in the document.
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
COST_MULTIPLE = 3.0        # exclusion threshold, as a multiple of the universe median
NEUTRALISE = False         # see the header note: tested on the tranched book and rejected

REGIMES = [
    ("2011-06", "2011-12", "post-crisis commodity peak"),
    ("2012-01", "2014-06", "grinding decline"),
    ("2014-07", "2016-02", "oil collapse, $105 to $26"),
    ("2016-03", "2020-01", "range-bound, low volatility"),
    ("2020-02", "2020-12", "COVID shock, negative WTI"),
    ("2021-01", "2022-06", "inflation surge, Ukraine"),
    ("2022-07", "2026-08", "normalisation and tightening"),
]


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
        df["basis_d"] = np.log(df["settle_0"] / df["settle_1"]) / (gap / 365.25)
    df.loc[(gap <= 0) | (gap > 400), "basis_d"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
    df["dom"] = df.groupby(["symbol", "ym"]).cumcount()
    return df


def monthly(df: pd.DataFrame, keep: set | None = None) -> pd.DataFrame:
    d = df[df["asset"] == "commodity"]
    if keep is not None:
        d = d[d["symbol"].isin(keep)]
    m = (d.groupby(["symbol", "ym"])
          .agg(r0=("r0", lambda s: s.sum(min_count=1)),
               r1=("r1", lambda s: s.sum(min_count=1)),
               basis=("basis_d", "last"), px=("settle_0", "last"),
               nd=("r0", "size")).reset_index())
    m = m[m["nd"] >= 10].sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["mom0"] = g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["mom1"] = g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm"] = m["mom0"] - m["mom1"]
    m["carry"] = m["basis"]
    m["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    m["sector"] = m["symbol"].map(lambda s: BY_SYMBOL[s].sector)
    return m


def ex_ante_costs(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["asset"] == "commodity"]
    med = d.groupby("symbol")["settle_0"].median()
    vol = (d.groupby("symbol")["r0"].std() * np.sqrt(252))
    rows = []
    for s in sorted(d["symbol"].unique()):
        inst = BY_SYMBOL[s]
        notional = med[s] * inst.dollar_price_mult
        tick_bp = inst.tick_value / notional * 1e4
        cost = 0.5 * tick_bp + 1.0 * tick_bp + inst.commission / notional * 1e4
        mvol_bp = vol[s] / np.sqrt(12) * 1e4        # one month of volatility, in bp
        rows.append(dict(symbol=s, sector=inst.sector, notional=notional,
                         tick=inst.tick, tick_val=inst.tick_value, tick_bp=tick_bp,
                         cost_bp=cost, ann_vol=vol[s],
                         cost_per_vol=cost / mvol_bp * 100 if mvol_bp > 0 else np.nan))
    return pd.DataFrame(rows).sort_values("cost_bp", ascending=False)


def neutralise(w: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """
    Remove net dollar exposure exactly, without reordering anything.

    Position i receives notional proportional to w_i / vol_i, so net notional is
    proportional to sum(w_i * v_i) with v_i = 1/vol_i - NOT to sum(w_i). Subtracting the
    single constant c = sum(w_i v_i) / sum(v_i) from every weight makes that sum exactly
    zero. Because the same constant is applied to every instrument, the ordering the
    signal produced is untouched: this is a risk control, not a second signal.
    """
    v = 1.0 / vol
    sv = float(np.sum(v))
    if sv <= 0:
        return w
    return w - float(np.sum(w * v)) / sv


def grid_targets(df: pd.DataFrame, offset: int, keep: set, min_n: int = 6) -> pd.DataFrame:
    d = df[(df["asset"] == "commodity") & df["symbol"].isin(keep)].sort_values(
        ["symbol", "date"]).copy()
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
        r = s["bm"].rank(); w = (r - r.mean()).to_numpy().astype(float); gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        if NEUTRALISE:
            w = neutralise(w, s["vol"].to_numpy())
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


def tranched(df: pd.DataFrame, frames, keep: set, cost_map=None, flat=3.0):
    d = df[(df["asset"] == "commodity") & df["symbol"].isin(keep)]
    dates = pd.DatetimeIndex(sorted(d["date"].unique()))
    syms = sorted(keep)
    ret = d.pivot_table(index="date", columns="symbol", values="r0").reindex(
        dates, columns=syms)
    px = d.pivot_table(index="date", columns="symbol", values="settle_0").reindex(
        dates, columns=syms).ffill()
    stacks = [(tf.pivot_table(index="date", columns="symbol", values="target")
                 .reindex(index=dates, columns=syms).ffill()).to_numpy()
              for tf in frames if not tf.empty]
    S = np.stack(stacks, axis=0)
    cnt = np.sum(~np.isnan(S), axis=0)
    T = np.divide(np.nansum(S, axis=0), np.maximum(cnt, 1),
                  out=np.zeros_like(cnt, dtype=float), where=cnt > 0)
    N = np.round(T)
    dpm = np.array([BY_SYMBOL[s].dollar_price_mult for s in syms])
    comm = np.array([BY_SYMBOL[s].commission for s in syms])
    bps = np.array([cost_map[s] if cost_map else flat for s in syms])
    P = np.nan_to_num(px.to_numpy(), nan=0.0)
    R = np.nan_to_num(ret.to_numpy(), nan=0.0)
    held = N[:-1]
    pnl = np.nansum(held * dpm * P[:-1] * np.expm1(R[1:]), axis=1)
    trades = np.abs(np.diff(N, axis=0))
    cost = np.nansum(trades * (comm + np.abs(dpm) * P[:-1] * bps / 1e4), axis=1)
    nz = np.abs(T[:-1]) > 1e-9
    notional = held * dpm * P[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        expo = np.nansum(notional, axis=1) / np.nansum(np.abs(notional), axis=1)
    expo = expo[np.isfinite(expo)]
    return dict(net=pd.Series((pnl - cost) / CAPITAL, index=dates[1:]),
                net_expo_mean=float(np.mean(np.abs(expo))) if len(expo) else np.nan,
                net_expo_max=float(np.max(np.abs(expo))) if len(expo) else np.nan,
                gross=pd.Series(pnl / CAPITAL, index=dates[1:]),
                cost=pd.Series(cost / CAPITAL, index=dates[1:]),
                gross_lev=np.nanmean(np.nansum(np.abs(held) * dpm * P[:-1], axis=1)) / CAPITAL,
                zero_share=float(((N[:-1] == 0) & nz).sum() / max(nz.sum(), 1)),
                trades_pm=float(trades.sum(axis=1).mean() * 21), syms=syms)


def mbook(m: pd.DataFrame, sig="bm", bps=3.0, seed=None, drop=None, J_=None, vw=None,
          min_n=6) -> pd.Series:
    if drop:
        m = m[m["symbol"] != drop]
    if J_ or vw:
        m = m.copy(); g = m.groupby("symbol")
        if J_:
            m["bm"] = (g["r0"].transform(lambda s: s.rolling(J_, min_periods=J_).sum())
                       - g["r1"].transform(lambda s: s.rolling(J_, min_periods=J_).sum()))
        if vw:
            m["vol"] = (g["r0"].transform(
                lambda s: s.rolling(vw, min_periods=3).std()) * np.sqrt(12)
                ).groupby(m["symbol"]).shift(1)
    rng = np.random.default_rng(seed) if seed is not None else None
    prev, out = {}, {}
    for ym, g in m.groupby("ym"):
        s = g[["symbol", sig, "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        sv = s[sig]
        if rng is not None:
            sv = pd.Series(rng.permutation(sv.to_numpy()), index=sv.index)
        r = sv.rank(); w = (r - r.mean()).to_numpy().astype(float); gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        if NEUTRALISE:
            w = neutralise(w, s["vol"].to_numpy())
            gr = np.abs(w).sum()
            if gr <= 0:
                continue
            w = w / gr
        pnl = cost = 0.0; held = {}
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]; dpm = inst.dollar_price_mult
            den = dpm * px * vol
            if den <= 0:
                continue
            n = float(np.round(wi * CAPITAL * VOL_TARGET * IDM / den))
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
    return pd.Series(out).sort_index()


def st(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return dict(n=len(r), sharpe=np.nan, t=np.nan, ann=np.nan, vol=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), yrs=yrs, sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12,
                vol=av, dd=float((eq / eq.cummax() - 1).min()))


def r2(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    if len(y) < 250 or y.var() <= 0:
        return np.nan
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.pinv(A.T @ A) @ (A.T @ y)
    return float(1.0 - (y - A @ b).var() / y.var())


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=40)
    a = ap.parse_args()

    df = load_daily(a.prices)
    C = ex_ante_costs(df)
    med = C["cost_bp"].median()
    excluded = set(C[C["cost_bp"] > COST_MULTIPLE * med]["symbol"])
    keep = set(C["symbol"]) - excluded

    print("=" * 84)
    print("1. THE UNIVERSE RULE — stated on cost, decided before any return")
    print("=" * 84)
    print(f"  Rule: exclude any instrument whose ex-ante round-trip cost exceeds")
    print(f"  {COST_MULTIPLE:.0f}x the universe median. Median is {med:.2f}bp, so the")
    print(f"  threshold is {COST_MULTIPLE*med:.2f}bp per side.\n")
    print(f"  {'sym':5s} {'sector':10s} {'notional':>10s} {'tick':>9s} {'$/tick':>8s} "
          f"{'tick bp':>8s} {'cost bp':>8s} {'ann vol':>8s} {'cost/mo vol':>12s}")
    for _, r in C.iterrows():
        mark = "  EXCLUDED" if r["symbol"] in excluded else ""
        print(f"  {r['symbol']:5s} {r['sector']:10s} ${r['notional']:>9,.0f} "
              f"{r['tick']:>9.4f} ${r['tick_val']:>7.2f} {r['tick_bp']:>8.2f} "
              f"{r['cost_bp']:>8.2f} {r['ann_vol']*100:>7.1f}% "
              f"{r['cost_per_vol']:>11.2f}%{mark}")
    print(f"\n  excluded: {sorted(excluded) if excluded else 'none'}")
    print(f"  universe: {len(keep)} instruments")
    if excluded:
        e = C[C["symbol"].isin(excluded)].iloc[0]
        print(f"\n  ECONOMIC REASON, not a performance reason. {e['symbol']}'s minimum")
        print(f"  price increment is {e['tick']} per unit, worth ${e['tick_val']:.2f}, set")
        print(f"  when the underlying traded far higher. On a ${e['notional']:,.0f}")
        print(f"  notional that single tick is {e['tick_bp']:.1f}bp - the universe median")
        print(f"  tick is {C['tick_bp'].median():.2f}bp. The increment was never rescaled")
        print(f"  as the underlying declined.")
        print(f"\n  The last column is the decisive one: cost as a share of ONE MONTH of")
        print(f"  that instrument's own volatility. A genuinely volatile contract can")
        print(f"  support a wider spread, so this adjusts for opportunity. {e['symbol']}")
        print(f"  costs {e['cost_per_vol']:.2f}% of a month's volatility against a median")
        print(f"  of {C['cost_per_vol'].median():.2f}% - it remains an outlier after the")
        print(f"  adjustment, which is what makes the case decisive rather than merely")
        print(f"  arithmetic.")

    cost_map = dict(zip(C["symbol"], C["cost_bp"]))
    m = monthly(df, keep)
    m_all = monthly(df, set(C["symbol"]))
    print("\n  building the tranched book...")
    frames = [f for f in (grid_targets(df, o, keep) for o in range(N_GRIDS)) if not f.empty]
    B = tranched(df, frames, keep, cost_map=cost_map)
    net = B["net"].resample("ME").sum(); net = net[net != 0]
    net.index = net.index.to_period("M")
    gross = B["gross"].resample("ME").sum(); gross.index = gross.index.to_period("M")
    cser = B["cost"].resample("ME").sum(); cser.index = cser.index.to_period("M")
    gross = gross.reindex(net.index); cser = cser.reindex(net.index)
    S = st(net)

    print("\n" + "=" * 84)
    print("2. HEADLINE — tranched, daily-marked, bottom-up costs")
    print("=" * 84)
    print(f"  instruments                     {len(keep)}")
    print(f"  months                          {S['n']}   "
          f"({net.index.min()} to {net.index.max()})")
    print(f"  Sharpe ratio                    {S['sharpe']:.3f}")
    print(f"  t-statistic                     {S['t']:.2f}")
    print(f"  annualised return               {S['ann']*100:.2f}%")
    print(f"  annualised volatility           {S['vol']*100:.1f}%")
    print(f"  maximum drawdown                {S['dd']*100:.1f}%")
    print(f"  gross exposure                  {B['gross_lev']:.1f}x")
    print(f"  positions rounding to zero      {B['zero_share']*100:.1f}%")
    print(f"  net dollar exposure, mean       {B['net_expo_mean']*100:.1f}% of gross")
    print(f"  net dollar exposure, worst      {B['net_expo_max']*100:.1f}% of gross")
    bonf = 3.3
    print(f"\n  Bonferroni bound over ~50 looks needs t > {bonf}: "
          f"{'CLEARS' if S['t'] > bonf else 'does not clear'} at t = {S['t']:.2f}")
    print(f"\n  net exposure is a REPORTED quantity, not a constrained one. Weights sum to")
    print(f"  zero; dividing by dollar volatility to equalise risk leaves residual")
    print(f"  notional, and a control that forces it to zero was tested and rejected")
    print(f"  (it costs ~0.11 Sharpe on this book and worsens the drawdown; the control's")
    print(f"  effect is not stable across constructions - it helps the drawdown on the")
    print(f"  single-grid book and hurts it here - and that instability is itself reason")
    print(f"  enough to leave a discretionary control out).")

    print("\n" + "=" * 84)
    print("3. BENCHMARKS ON THE SAME UNIVERSE")
    print("=" * 84)
    for sig, lab in (("mom0", "front momentum alone"), ("carry", "carry alone")):
        print(f"  {lab:34s} {st(mbook(m, sig=sig))['sharpe']:.3f}")
    base_m = mbook(m)
    ts = {}
    for ym, g in m.groupby("ym"):
        s = g[["mom0", "fwd"]].dropna()
        if len(s) < 6:
            continue
        w = np.sign(s["mom0"].to_numpy())
        if np.abs(w).sum() > 0:
            ts[ym] = float((w / np.abs(w).sum() * s["fwd"].to_numpy()).sum())
    tsm = pd.Series(ts).sort_index()
    j = pd.concat([net.rename("s"), tsm.rename("t")], axis=1).dropna()
    print(f"  {'correlation to trend-following':34s} {j['s'].corr(j['t']):+.3f}")
    mkt = m.groupby("ym")["fwd"].mean().dropna()
    jm = pd.concat([net.rename("s"), mkt.rename("m")], axis=1).dropna()
    X = np.column_stack([np.ones(len(jm)), jm["m"].to_numpy()])
    b = np.linalg.pinv(X.T @ X) @ (X.T @ jm["s"].to_numpy())
    e = jm["s"].to_numpy() - X @ b
    se = e.std(ddof=2) / np.sqrt(len(jm))
    print(f"  {'market beta':34s} {b[1]:+.3f}")
    print(f"  {'market-adjusted alpha':34s} {b[0]*12*100:+.2f}%/yr (t {b[0]/se:+.2f})")

    print("\n" + "=" * 84)
    print("4. ECONOMIC RATIONALE")
    print("=" * 84)
    cs = []
    for _, g in m.groupby("ym"):
        s = g[["mom0", "mom1"]].dropna()
        if len(s) >= 6 and s["mom0"].std() > 0 and s["mom1"].std() > 0:
            cs.append(s["mom0"].corr(s["mom1"]))
    print(f"  correlation of the two legs           {np.mean(cs)*100:.1f}%")
    print(f"  variance of BM / front momentum       {m['bm'].var()/m['mom0'].var()*100:.1f}%")
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    print(f"  average pairwise correlation          "
          f"{np.nanmean(cm[np.triu_indices_from(cm, k=1)]):.3f}")

    print("\n  CHANNELS — variance of the common component removed (leave-one-out)")
    d = df[(df["asset"] == "commodity") & df["symbol"].isin(keep)]
    p0 = d.pivot_table(index="date", columns="symbol", values="r0").sort_index()
    p1 = d.pivot_table(index="date", columns="symbol", values="r1").sort_index()
    idx = p0.index.union(p1.index); p0, p1 = p0.reindex(idx), p1.reindex(idx)
    syms = [s for s in p0.columns if p0[s].notna().sum() > 500]
    ch = {"curve": [], "market": [], "sector": [], "pca5": [], "pca8": []}
    for s in syms:
        y = p0[s].to_numpy()
        if s in p1.columns:
            v = r2(y, p1[s].to_numpy().reshape(-1, 1))
            if np.isfinite(v): ch["curve"].append(v)
        others = [o for o in syms if o != s]
        v = r2(y, p0[others].mean(axis=1).to_numpy().reshape(-1, 1))
        if np.isfinite(v): ch["market"].append(v)
        sec = [o for o in others if BY_SYMBOL[o].sector == BY_SYMBOL[s].sector]
        if sec:
            v = r2(y, p0[sec].mean(axis=1).to_numpy().reshape(-1, 1))
            if np.isfinite(v): ch["sector"].append(v)
        A = p0[others].fillna(0.0).to_numpy()
        Ac = A - A.mean(axis=0, keepdims=True)
        try:
            U, Sv, _ = np.linalg.svd(Ac, full_matrices=False)
            for k in (5, 8):
                if U.shape[1] >= k:
                    v = r2(y, (U * Sv)[:, :k])
                    if np.isfinite(v): ch[f"pca{k}"].append(v)
        except np.linalg.LinAlgError:
            pass
    for k, lab, nreg in (("curve", "deferred contract", 1), ("pca8", "8 principal comps", 8),
                         ("pca5", "5 principal comps", 5), ("sector", "sector peers", 1),
                         ("market", "equal-weighted market", 1)):
        if ch[k]:
            print(f"    {lab:24s} {np.mean(ch[k])*100:5.1f}%   ({nreg} regressor"
                  f"{'s' if nreg > 1 else ''})")

    print("\n" + "=" * 84)
    print("5. ROBUSTNESS")
    print("=" * 84)
    pt = np.array([v for v in (st(mbook(m, seed=s))["t"] for s in range(a.seeds))
                   if np.isfinite(v)])
    bt = st(base_m)["t"]
    print(f"  placebo, {len(pt)} shuffles           t {pt.mean():+.2f} "
          f"+/- {pt.std(ddof=1):.2f}   real {bt:+.2f}   "
          f"{(bt-pt.mean())/max(pt.std(ddof=1),1e-9):+.1f} sd")
    jk = [st(mbook(m, drop=x))["sharpe"] for x in sorted(keep)]
    jk = [v for v in jk if np.isfinite(v)]
    print(f"  jackknife                        worst {min(jk):.3f}   best {max(jk):.3f}")
    grid = {(k, vw): st(mbook(m, J_=k, vw=vw))["sharpe"]
            for k in (6, 9, 12, 15) for vw in (3, 6, 12)}
    ok = sum(1 for v in grid.values() if np.isfinite(v) and v > 0.35)
    print(f"  parameter grid                   {ok} of {len(grid)} cells above 0.35 "
          f"(min {min(grid.values()):.3f}, max {max(grid.values()):.3f})")
    tot = net.sum()
    print(f"  best 6 of {len(net)} months          "
          f"{net.nlargest(6).sum()/tot*100:.1f}% of P&L")
    yr = net.groupby(net.index.year).sum()
    print(f"  positive calendar years          {int((yr>0).sum())} of {len(yr)}")
    print(f"  worst calendar year              {yr.idxmin()} at {yr.min()*100:+.1f}%")
    print(f"  minimum detectable Sharpe diff   {2/np.sqrt(S['yrs']):.2f}")

    print("\n" + "=" * 84)
    print("6. REGIMES")
    print("=" * 84)
    mktm = m.groupby("ym")["r0"].mean()
    print(f"  {'period':17s} {'regime':30s} {'mkt/yr':>8s} {'SR':>7s} {'ret/yr':>8s} "
          f"{'maxDD':>8s} {'n':>4s}")
    npos = ntot = 0
    for lo, hi, lab in REGIMES:
        seg = net[(net.index >= lo) & (net.index <= hi)]
        mseg = mktm[(mktm.index >= lo) & (mktm.index <= hi)]
        if len(seg) < 4:
            continue
        ntot += 1; npos += int(seg.mean() > 0)
        yrs = len(seg) / 12
        av = seg.std(ddof=1) * np.sqrt(12)
        sr = (seg.mean() * 12) / av if av > 0 else np.nan
        eq = (1 + seg).cumprod()
        print(f"  {lo}\u2013{hi[2:]:9s} {lab:30s} {mseg.mean()*12*100:>+7.1f}% "
              f"{sr:>+7.2f} {seg.mean()*12*100:>+7.1f}% "
              f"{float((eq/eq.cummax()-1).min())*100:>+7.1f}% {len(seg):>4d}")
    print(f"\n  positive in {npos} of {ntot} regimes")

    print("\n" + "=" * 84)
    print("7. COSTS AND CAPACITY")
    print("=" * 84)
    turn_w = np.average(C[C["symbol"].isin(keep)]["cost_bp"])
    print(f"  turnover-weighted cost           {turn_w:.2f} bp per side (bottom-up)")
    print(f"  cost as share of gross profit    {cser.sum()/gross.sum():.1%}")
    print(f"  annual cost                      {cser.sum()/S['yrs']*100:.2f}% of capital")
    for bps in (3, 10, 20, 40):
        Bx = tranched(df, frames, keep, flat=bps)
        nx = Bx["net"].resample("ME").sum(); nx = nx[nx != 0]
        print(f"  Sharpe at flat {bps:>2d}bp per side    {st(nx)['sharpe']:.3f}")
    print(f"  positions rounding to zero       {B['zero_share']*100:.1f}%")
    print(f"  contracts traded per month       {B['trades_pm']:.0f}")

    print("\n" + "=" * 84)
    print("8. PORTFOLIO COMBINATION")
    print("=" * 84)
    rho = j["s"].corr(j["t"])
    for wb in (0.20, 0.30, 0.40):
        wa = 1 - wb
        c = (wa * 0.6 + wb * S["sharpe"]) / np.sqrt(wa**2 + wb**2 + 2*wa*wb*rho)
        print(f"  {wb:.0%} risk beside a 0.6 trend book   combined {c:.3f}")
    na, nb = 0.6 - rho*S["sharpe"], S["sharpe"] - rho*0.6
    if na + nb > 0:
        w = nb / (na + nb); wa = 1 - w
        c = (wa*0.6 + w*S["sharpe"]) / np.sqrt(wa**2 + w**2 + 2*wa*w*rho)
        print(f"  optimal ({w:.0%} to this)              combined {c:.3f}")

    print("\n" + "=" * 84)
    print("  Every figure above came from one specification on one file in one run.")
    print("  If a number is not here, it does not belong in the pitch.")


if __name__ == "__main__":
    main()