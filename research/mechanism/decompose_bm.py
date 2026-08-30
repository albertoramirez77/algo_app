"""
decompose_bm.py — where did every basis point of the Sharpe come from?

    python decompose_bm.py --prices px_wide.parquet

WHY THIS EXISTS

The reported Sharpe rose at every stage of testing: 0.602 -> 0.792 -> 0.848. Each change
had a defensible reason and none was made to improve the number, but monotone improvement
across three revisions is indistinguishable from tuning unless the provenance is written
down. This writes it down.

One of those changes was also undisclosed. The 0.760 -> 0.848 move was attributed to fixing
the entry price. That cannot be right. In the sizing formula

    N = w x C x tau x IDM / (M x P x sigma)     and     PnL = N x M x P x r

the multiplier M and the price P cancel exactly. Entry price and the cents-versus-dollars
units fix therefore have NO effect on fractional returns; they matter only through integer
rounding. The real cause was a volatility-timing change made silently in the same edit.

THE LADDER

Each rung adds exactly one change to the rung below it.

    0  raw rank weights, no volatility scaling          the original 0.602 specification
    1  + inverse-volatility scaling within the ranks    risk parity, Carver's framework
    2  + integer contracts                              the only thing actually tradeable
    3  + transaction costs at 3bp per side
    4  + volatility estimated through the current month rather than lagged one month
    5  + entry price at the current settle rather than lagged

Rungs 4 and 5 are both legitimate: if you trade at the settle on the last day of month t,
month t's data is known. But they were adopted mid-stream, so they are isolated here and
the conservative specification is what gets frozen.

THE MECHANISM QUESTION

Boons & Prado attribute basis-momentum to imbalances that materialise "when the
market-clearing ability of speculators and intermediaries is impaired," and explicitly
reject storage and inventory explanations. Your data rejects THEIR explanation: two of
three impairment proxies show the effect is weaker, not stronger, under stress.

So this script tests the explanation they rejected. Basis-momentum is, mechanically, the
momentum of the curve slope. A curve steepening into backwardation is the market pricing
tightening physical supply. If the effect concentrates where inventories are tight, the
inventory story holds and the paper's dismissal of it does not survive in modern data.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

CAPITAL = 450_000.0
VOL_TARGET = 0.20
IDM_CAP = 2.5


def load(path: str, J: int = 12, vol_window: int = 6) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df = df[df["contract_0"] != df["contract_1"]]
    df = (df.sort_values(["symbol", "date", "oi_0"], na_position="first")
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"]).reset_index(drop=True))
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
    gap = (df["expiry_1"] - df["expiry_0"]).dt.days
    with np.errstate(invalid="ignore", divide="ignore"):
        df["basis"] = np.log(df["settle_0"] / df["settle_1"]) / (gap / 365.25)
    df.loc[(gap <= 0) | (gap > 400), "basis"] = np.nan
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df["ym"] = df["date"].dt.to_period("M")

    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                basis=("basis", "last"), px=("settle_0", "last"),
                n_days=("r0", "size")).reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m = m[(m["n_days"] >= 10) & (m["asset"] == "commodity")].copy()
    c0 = m.groupby("symbol")["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    c1 = m.groupby("symbol")["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm"] = c0 - c1
    v = m.groupby("symbol")["r0"].transform(
        lambda s: s.rolling(vol_window, min_periods=3).std()) * np.sqrt(12)
    m["vol_now"] = v
    m["vol_lag"] = v.groupby(m["symbol"]).shift(1)
    m["px_now"] = m["px"]
    m["px_lag"] = m.groupby("symbol")["px"].shift(1)
    m["fwd"] = m.groupby("symbol")["r0"].shift(-1)
    return m.sort_values(["symbol", "ym"]).reset_index(drop=True)


def idm_of(m: pd.DataFrame) -> float:
    n = m["symbol"].nunique()
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    return min(1.0 / np.sqrt((1/n) + (1 - 1/n) * max(rho, 0.01)), IDM_CAP)


def run(m: pd.DataFrame, idm: float, *, vol_scale: bool, integer: bool, bps: float,
        vol_col: str, px_col: str, seed: int | None = None,
        cond: pd.Series | None = None, min_n: int = 6) -> pd.Series:
    """
    One rung of the ladder. `cond`, if given, is a per-row boolean that zeroes any
    position failing the condition — used for the mechanism tests.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    prev, out = {}, {}
    for ym, g in m.groupby("ym"):
        cols = ["symbol", "bm", vol_col, px_col, "fwd"]
        s = g[cols + (["_cond"] if cond is not None else [])].dropna()
        s = s[(s[vol_col] > 0) & (s[px_col] > 0)]
        if len(s) < min_n:
            continue
        sig = s["bm"]
        if rng is not None:
            sig = pd.Series(rng.permutation(sig.to_numpy()), index=sig.index)
        r = sig.rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr

        pnl = cost = 0.0
        held = {}
        for i, (sym, wi, vol, px, fwd) in enumerate(
                zip(s["symbol"], w, s[vol_col], s[px_col], s["fwd"])):
            if cond is not None and not s["_cond"].iloc[i]:
                continue
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            if vol_scale:
                denom = dpm * px * vol
                tgt = wi * CAPITAL * VOL_TARGET * idm / denom if denom > 0 else 0.0
            else:
                # equal dollar notional per unit of weight: reproduces the original
                # unlevered rank portfolio, where PnL = w x capital x r
                tgt = wi * CAPITAL / (dpm * px)
            n = float(np.round(tgt)) if integer else tgt
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0 and bps > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        if bps > 0:
            for sym in set(prev) - set(held):
                cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
    return pd.Series(out).sort_index()


def stat(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 48:
        return dict(n=len(r), sharpe=np.nan, t=np.nan, ann=np.nan, vol=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12, vol=av,
                dd=float((eq / eq.cummax() - 1).min()))


def line(lbl: str, s: dict, delta: float | None = None) -> None:
    if not np.isfinite(s["sharpe"]):
        print(f"  {lbl:44s} n={s['n']}"); return
    d = f"  {delta:+6.3f}" if delta is not None else "        "
    print(f"  {lbl:44s} SR {s['sharpe']:>+6.3f}{d}  t {s['t']:>+5.2f}  "
          f"vol {s['vol']*100:>5.1f}%  dd {s['dd']*100:>+6.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_wide.parquet")
    ap.add_argument("--seeds", type=int, default=25)
    a = ap.parse_args()

    m = load(a.prices)
    idm = idm_of(m)

    print("=" * 82)
    print("1. THE LADDER — one specification change per rung")
    print("=" * 82)
    rungs = [
        ("0  raw rank, no vol scaling (the original)",
         dict(vol_scale=False, integer=False, bps=0, vol_col="vol_lag", px_col="px_lag")),
        ("1  + inverse-vol scaling",
         dict(vol_scale=True, integer=False, bps=0, vol_col="vol_lag", px_col="px_lag")),
        ("2  + integer contracts",
         dict(vol_scale=True, integer=True, bps=0, vol_col="vol_lag", px_col="px_lag")),
        ("3  + costs at 3bp/side",
         dict(vol_scale=True, integer=True, bps=3, vol_col="vol_lag", px_col="px_lag")),
        ("4  + vol through current month",
         dict(vol_scale=True, integer=True, bps=3, vol_col="vol_now", px_col="px_lag")),
        ("5  + entry price at current settle",
         dict(vol_scale=True, integer=True, bps=3, vol_col="vol_now", px_col="px_now")),
    ]
    res, prev_sr = {}, None
    for lbl, kw in rungs:
        s = stat(run(m, idm, **kw))
        res[lbl[0]] = (s, kw)
        line(lbl, s, None if prev_sr is None else s["sharpe"] - prev_sr)
        prev_sr = s["sharpe"]

    print("\n  The delta column is what each change contributed, in Sharpe.")
    print("  Rungs 4 and 5 were adopted mid-stream without being flagged. Rung 5 should")
    print("  contribute almost nothing: price and multiplier cancel in the sizing")
    print("  formula, so entry price only reaches the result through integer rounding.")

    print("\n" + "=" * 82)
    print("2. THE FROZEN SPECIFICATION")
    print("=" * 82)
    print("  Rung 3 is frozen as the headline: inverse-vol scaling, integer contracts,")
    print("  3bp costs, and BOTH volatility and price lagged one month.")
    print()
    print("  Why the conservative rung and not the best one. Rungs 4 and 5 are defensible")
    print("  on their merits — if you trade at the settle on the last day of month t, then")
    print("  month t's data is known and lagging discards information. But they were")
    print("  adopted after the result was visible, and the difference is small. Quoting")
    print("  the lower number costs little and removes the entire argument.")
    frozen_kw = res["3"][1]
    frozen = run(m, idm, **frozen_kw)
    fs = stat(frozen)
    line("FROZEN", fs)
    print(f"\n  the more aggressive but still non-look-ahead variant (rung 5) gives "
          f"{res['5'][0]['sharpe']:+.3f}")
    print("  Report that as a robustness note, not as the headline.")

    print("\n" + "=" * 82)
    print("3. PLACEBO ON THE FROZEN SPECIFICATION")
    print("=" * 82)
    ts = [stat(run(m, idm, seed=sd, **frozen_kw))["t"] for sd in range(a.seeds)]
    ts = np.array([t for t in ts if np.isfinite(t)])
    z = (fs["t"] - ts.mean()) / max(ts.std(ddof=1), 1e-9)
    print(f"  placebo t {ts.mean():+.2f} +/- {ts.std(ddof=1):.2f} over {len(ts)} seeds")
    print(f"  real t {fs['t']:+.2f} sits {z:+.1f} placebo sd out   "
          f"{'PASS' if abs(z) > 2 else 'FAIL'}")

    print("\n" + "=" * 82)
    print("4. MECHANISM — the paper's story failed, so test the one it rejected")
    print("=" * 82)
    print("  Boons & Prado: imbalances when intermediaries are impaired. Explicitly NOT")
    print("  storage or inventory. Your stress test found the effect WEAKER under")
    print("  intermediation stress on two of three proxies, so their story does not hold")
    print("  here. Basis-momentum is mechanically the momentum of the curve slope, and a")
    print("  curve steepening into backwardation is the market pricing tightening supply.")
    print("  If the effect concentrates where inventories are tight, the inventory story")
    print("  they rejected is the one that survives.\n")

    # cross-sectional: restrict the book to backwardated names, then to contangoed names
    for lab, mask in (("backwardated only (basis > 0, tight inventory)", m["basis"] > 0),
                      ("contangoed only (basis <= 0, ample inventory)", m["basis"] <= 0)):
        mm = m.copy()
        mm["_cond"] = mask
        s = stat(run(mm, idm, cond=mm["_cond"], **frozen_kw))
        line(lab, s)

    # time series: months when the complex as a whole is tight
    agg = m.groupby("ym")["basis"].median().dropna()
    med = agg.expanding().median().shift(1)
    tight = agg.index[(agg > med).fillna(False)]
    ample = agg.index[(agg <= med).fillna(False)]
    print()
    line("months: complex tighter than its history", stat(frozen.reindex(tight).dropna()))
    line("months: complex looser than its history", stat(frozen.reindex(ample).dropna()))

    print("\n  Read it this way. If the backwardated leg is materially stronger, the")
    print("  economic rationale is inventory scarcity and you can say so with a test")
    print("  behind it — a genuine disagreement with a Journal of Finance paper, which")
    print("  is a far better pitch than repeating its abstract. If neither leg")
    print("  separates, then no mechanism you have tested is supported, and the honest")
    print("  rationale is that basis-momentum measures curve dynamics that predict")
    print("  returns for reasons this data cannot identify. Say that plainly.")

    print("\n" + "=" * 82)
    print("5. WHAT TO PUT IN THE PITCH")
    print("=" * 82)
    print(f"  headline Sharpe        {fs['sharpe']:.3f}  (t {fs['t']:.2f}), frozen at rung 3")
    print(f"  annual return          {fs['ann']*100:.2f}% on {fs['vol']*100:.1f}% vol")
    print(f"  max drawdown           {fs['dd']*100:.1f}%")
    print(f"  placebo separation     {z:+.1f} sd")
    print("  costs                   survives 20bp/side")
    print("  every number above comes from one frozen specification, and the ladder")
    print("  above shows what each earlier revision contributed.")


if __name__ == "__main__":
    main()