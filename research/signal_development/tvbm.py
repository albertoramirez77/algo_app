"""
tvbm.py — validate the time-varying units correction, adversarially.

    python tvbm.py --prices px_wide.parquet

WHAT SURVIVED THE LAST ROUND, AND WHY THE LAST VERDICT WAS WRONG

The gap weighting decomposes exactly:

    365.25/gap_t  =  (365.25/gap_const)  x  (gap_const/gap_t)
                      STATIC                 TIME-VARYING

Measured on real data:

    raw BM              0.760
    STATIC only         0.733   (-0.027)   the per-instrument tilt HURTS
    full flow           0.812   (+0.052)   dragged down by the static piece
    TIME-VARYING only   0.944   (+0.184)   the actual correction

The previous script concluded "it is a sector tilt." That was wrong, and it was wrong for
a structural reason: its random-constant placebo and its bootstrap were both run on FULL
FLOW, which contains the harmful static component. Worse, a random-constant placebo cannot
even apply to the time-varying leg — random per-instrument constants ARE the static
component, and the time-varying multiplier has a mean of exactly 1 within each instrument,
so it cannot express a per-instrument tilt at all.

This script tests the right object with the right placebos.

THE SHARPEST TEST AVAILABLE

If the correction is genuinely about maturity gaps, then instruments whose gaps never move
should contribute nothing to the gain. Metals sit at 30 days every single roll. Grains cycle
61, 61, 62, 91, 90. So:

    per-instrument gain from the correction  SHOULD track  that instrument's gap variability

No sector story predicts that. A tilt toward short-gap instruments would help energy and
metals most - and those are precisely the sectors with no gap variation, where the last run
already measured gains of -0.030 and +0.000.

THE RIGHT PLACEBO FOR A TIMING CLAIM

Permute each instrument's gap sequence THROUGH TIME. Same multipliers, same distribution,
attached to the wrong months. If the gain survives that, the timing of gap variation is
irrelevant and the units story is dead.
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
J = 12
VOL_WINDOW = 6


def load(path: str) -> pd.DataFrame:
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
    df["gap"] = gap.where((gap > 0) & (gap <= 400))
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df["ym"] = df["date"].dt.to_period("M")

    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                gap=("gap", "median"), px=("settle_0", "last"),
                n_days=("r0", "size")).reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m["sector"] = m["symbol"].map(lambda s: BY_SYMBOL[s].sector if s in BY_SYMBOL else "?")
    m = m[(m["n_days"] >= 10) & (m["asset"] == "commodity")].copy()
    m = m.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["spread"] = m["r0"] - m["r1"]
    v = g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
    m["vol"] = v.groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    return m


def add_signals(m: pd.DataFrame, expanding_const: bool = False) -> pd.DataFrame:
    m = m.copy()
    g = m.groupby("symbol")
    if expanding_const:
        # strictly ex ante: the instrument's typical gap as known at each point
        gc = g["gap"].transform(lambda s: s.expanding().median().shift(1))
        gc = gc.fillna(g["gap"].transform("median"))
    else:
        # the listing calendar is published years ahead, so an instrument's typical gap is
        # known ex ante; a full-sample median uses no information a trader lacks
        gc = g["gap"].transform("median")
    m["gap_const"] = gc
    m["s_tv"] = m["spread"] * (gc / m["gap"])
    m["s_static"] = m["spread"] * (365.25 / gc)
    m["s_flow"] = m["spread"] * (365.25 / m["gap"])
    m["bm_raw"] = g["spread"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    for col, nm in (("s_tv", "bm_tv"), ("s_static", "bm_static"), ("s_flow", "bm_flow")):
        m[nm] = m.groupby("symbol")[col].transform(
            lambda s: s.rolling(J, min_periods=J).sum())
    return m


def idm_of(m: pd.DataFrame) -> float:
    n = max(m["symbol"].nunique(), 2)
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    if not np.isfinite(rho):
        rho = 0.2
    return min(1.0 / np.sqrt((1 / n) + (1 - 1 / n) * max(rho, 0.01)), IDM_CAP)


def portfolio(m: pd.DataFrame, idm: float, sig: str, bps: float = 3.0,
              seed: int | None = None, min_n: int = 6,
              neutral_by: str | None = None) -> pd.Series:
    rng = np.random.default_rng(seed) if seed is not None else None
    prev, out = {}, {}
    for ym, g in m.groupby("ym"):
        cols = ["symbol", sig, "vol", "px_entry", "fwd"]
        if neutral_by:
            cols.append(neutral_by)
        s = g[cols].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        sv = s[sig].copy()
        if neutral_by:
            sv = sv - sv.groupby(s[neutral_by]).transform("mean")
        if rng is not None:
            sv = pd.Series(rng.permutation(sv.to_numpy()), index=sv.index)
        r = sv.rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        pnl = cost = 0.0
        held = {}
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            den = dpm * px * vol
            if den <= 0:
                continue
            n = float(np.round(wi * CAPITAL * VOL_TARGET * idm / den))
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


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 48:
        return np.nan
    av = r.std(ddof=1) * np.sqrt(12)
    return (r.mean() * 12) / av if av > 0 else np.nan


def stat(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 48:
        return dict(n=len(r), sharpe=np.nan, t=np.nan, ann=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12,
                dd=float((eq / eq.cummax() - 1).min()))


def line(lbl: str, s: dict, base: float | None = None) -> None:
    if not np.isfinite(s["sharpe"]):
        print(f"  {lbl:40s} n={s['n']}"); return
    d = f"  {s['sharpe']-base:+6.3f}" if base is not None else "        "
    star = " *" if abs(s["t"]) > 2 else ""
    print(f"  {lbl:40s} SR {s['sharpe']:>+6.3f}{d}  t {s['t']:>+5.2f}  "
          f"ret {s['ann']*100:>+6.2f}%  dd {s['dd']*100:>+6.1f}%{star}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_wide.parquet")
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()

    base = load(a.prices)
    m = add_signals(base)
    idm = idm_of(m)
    p_raw = portfolio(m, idm, "bm_raw")
    p_tv = portfolio(m, idm, "bm_tv")
    sr_raw, sr_tv = sharpe(p_raw), sharpe(p_tv)

    print("=" * 80)
    print("1. THE FOUR SIGNALS")
    print("=" * 80)
    line("raw BM (published)", stat(p_raw))
    line("STATIC only (the tilt)", stat(portfolio(m, idm, "bm_static")), sr_raw)
    line("full flow (static + TV)", stat(portfolio(m, idm, "bm_flow")), sr_raw)
    line("TIME-VARYING only  <-- the claim", stat(p_tv), sr_raw)
    print("\n  The static leg has an average multiplier that differs across instruments,")
    print("  so it CAN express a sector tilt. The time-varying leg has a mean multiplier")
    print("  of exactly 1 within every instrument, so it cannot. Any gain it produces is")
    print("  therefore not a cross-sectional bet by construction.")
    chk = m.groupby("symbol").apply(
        lambda g: (g["gap_const"] / g["gap"]).mean(), include_groups=False)
    print(f"  verification: mean TV multiplier per instrument ranges "
          f"{chk.min():.3f} to {chk.max():.3f}")

    print("\n" + "=" * 80)
    print("2. THE SHARPEST TEST — does the gain track GAP VARIABILITY?")
    print("=" * 80)
    print("  If this is really about units, instruments whose maturity gap never moves")
    print("  should contribute nothing. Leave-one-out: drop each instrument and see how")
    print("  much of the TV-minus-raw gain disappears with it.\n")
    gv = m.groupby("symbol")["gap"].agg(
        cv=lambda s: s.std() / s.mean() if s.mean() else np.nan,
        lo="min", hi="max", med="median")
    rows = []
    for sym in sorted(m["symbol"].unique()):
        sub = m[m["symbol"] != sym]
        i2 = idm_of(sub)
        gain_wo = sharpe(portfolio(sub, i2, "bm_tv")) - sharpe(portfolio(sub, i2, "bm_raw"))
        rows.append(dict(symbol=sym, cv=gv.at[sym, "cv"],
                         gap_lo=gv.at[sym, "lo"], gap_hi=gv.at[sym, "hi"],
                         gain_without=gain_wo,
                         contribution=(sr_tv - sr_raw) - gain_wo,
                         sector=BY_SYMBOL[sym].sector))
    tab = pd.DataFrame(rows).sort_values("cv", ascending=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    ok = tab.dropna(subset=["cv", "contribution"])
    if len(ok) > 5:
        rho = ok["cv"].corr(ok["contribution"])
        rho_s = ok["cv"].corr(ok["contribution"], method="spearman")
        print(f"\n  correlation of gap variability with contribution to the gain:")
        print(f"    Pearson {rho:+.3f}   Spearman {rho_s:+.3f}")
        flat = ok[ok["cv"] < 0.02]
        vary = ok[ok["cv"] >= 0.02]
        if len(flat) and len(vary):
            print(f"    mean contribution, gaps essentially FIXED (cv<0.02, n={len(flat)}): "
                  f"{flat['contribution'].mean():+.4f}")
            print(f"    mean contribution, gaps VARYING  (cv>=0.02, n={len(vary)}): "
                  f"{vary['contribution'].mean():+.4f}")
        track_ok = rho_s > 0.25
        print(f"  {'PASS' if track_ok else 'FAIL'} — no sector story predicts this pattern.")
    else:
        track_ok = False

    print("\n" + "=" * 80)
    print("3. THE RIGHT PLACEBO — permute each instrument's GAP SEQUENCE in time")
    print("=" * 80)
    print("  Same multipliers, same distribution, attached to the wrong months. If the")
    print("  gain survives, the TIMING of gap variation is irrelevant and the claim dies.")
    print("  A random-constant placebo cannot be used here: random per-instrument")
    print("  constants ARE the static component, which the time-varying leg excludes.\n")
    rng = np.random.default_rng(0)
    srs = []
    for _ in range(a.seeds):
        mm = m.copy()
        mm["gap"] = mm.groupby("symbol")["gap"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        mm = add_signals(mm)
        srs.append(sharpe(portfolio(mm, idm, "bm_tv")))
    srs = np.array([x for x in srs if np.isfinite(x)])
    z = (sr_tv - srs.mean()) / max(srs.std(ddof=1), 1e-9)
    print(f"  shuffled-gap-timing: mean {srs.mean():+.3f}  sd {srs.std(ddof=1):.3f}  "
          f"max {srs.max():+.3f}")
    print(f"  real TV {sr_tv:+.3f} sits {z:+.1f} sd out; beats "
          f"{(srs < sr_tv).mean():.0%} of draws")
    gapplacebo_ok = z > 2
    print(f"  {'PASS' if gapplacebo_ok else 'FAIL'}")

    print("\n  standard placebo — shuffle the TV signal across instruments each month:")
    ts = [stat(portfolio(m, idm, "bm_tv", seed=sd))["t"] for sd in range(a.seeds)]
    ts = np.array([t for t in ts if np.isfinite(t)])
    zt = (stat(p_tv)["t"] - ts.mean()) / max(ts.std(ddof=1), 1e-9)
    print(f"    placebo t {ts.mean():+.2f} +/- {ts.std(ddof=1):.2f}   "
          f"real {stat(p_tv)['t']:+.2f}   {zt:+.1f} sd   "
          f"{'PASS' if zt > 2 else 'FAIL'}")
    placebo_ok = zt > 2

    print("\n" + "=" * 80)
    print("4. BOOTSTRAP ON TV MINUS RAW")
    print("=" * 80)
    j = pd.concat([p_tv.rename("tv"), p_raw.rename("raw")], axis=1).dropna()
    d = (j["tv"] - j["raw"]).to_numpy()
    n = len(d)
    rng2 = np.random.default_rng(1)
    boots = []
    for _ in range(3000):
        idx = []
        while len(idx) < n:
            st = rng2.integers(0, max(n - 12, 1))
            idx.extend(range(st, min(st + 12, n)))
        boots.append(d[np.array(idx[:n])].mean() * 12)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  mean annual advantage {d.mean()*12*100:+.2f}%   "
          f"95% CI [{lo*100:+.2f}%, {hi*100:+.2f}%]")
    print(f"  share of resamples where TV loses: {(boots < 0).mean():.1%}")
    boot_ok = lo > 0
    print(f"  {'PASS' if boot_ok else 'FAIL'} — the interval must exclude zero")

    print("\n" + "=" * 80)
    print("5. ROBUSTNESS OF THE TIME-VARYING SIGNAL")
    print("=" * 80)
    print("  sector-neutral (all cross-sector tilt removed by construction):")
    line("  raw", stat(portfolio(m, idm, "bm_raw", neutral_by="sector")))
    line("  TV", stat(portfolio(m, idm, "bm_tv", neutral_by="sector")))

    print("\n  strictly ex-ante gap_const (expanding median, no full-sample input):")
    m2 = add_signals(base, expanding_const=True)
    line("  TV, expanding gap_const", stat(portfolio(m2, idm, "bm_tv")), sr_raw)

    print("\n  jackknife:")
    jk = tab["gain_without"] + sr_raw
    print(f"    TV Sharpe without each instrument: min {jk.min():+.3f}  "
          f"max {jk.max():+.3f}  (full {sr_tv:+.3f})")

    print("\n  P&L concentration and subperiods:")
    tot = p_tv.sum()
    for k in (3, 6, 12):
        print(f"    best {k:>2d} months = {p_tv.nlargest(k).sum()/tot*100:>6.1f}%")
    n3 = len(p_tv) // 3
    thirds = []
    for i, lab in enumerate(("first third ", "second third", "final third ")):
        seg = p_tv.iloc[i*n3:(i+1)*n3] if i < 2 else p_tv.iloc[2*n3:]
        thirds.append(sharpe(seg))
        line("  " + lab, stat(seg))
    yr = p_tv.groupby(p_tv.index.year).sum() * 100
    print("    annual %:", "  ".join(f"{y}:{v:+.0f}" for y, v in yr.items()))
    print(f"    positive years {int((yr > 0).sum())} of {len(yr)}")

    print("\n  cost sensitivity:")
    for bps in (3, 10, 20, 40):
        line(f"  {bps}bp per side", stat(portfolio(m, idm, "bm_tv", bps=bps)))

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    conc = p_tv.nlargest(6).sum() / tot if tot else np.nan
    checks = [
        ("TV beats raw", np.isfinite(sr_tv) and sr_tv > sr_raw),
        ("gain tracks gap variability", track_ok),
        ("survives gap-timing placebo", gapplacebo_ok),
        ("survives cross-sectional placebo", placebo_ok),
        ("bootstrap CI excludes zero", boot_ok),
        ("P&L not concentrated", np.isfinite(conc) and conc < 0.60),
        ("all thirds positive", all(np.isfinite(x) and x > 0 for x in thirds)),
    ]
    for k, v in checks:
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    n_pass = sum(v for _, v in checks)
    print()
    if n_pass == len(checks):
        print("  THE TIME-VARYING UNITS CORRECTION HOLDS. Headline the TV signal, state")
        print("  that the static component HURTS (-0.027) and was removed on the evidence,")
        print("  and report the multiple-testing burden alongside: roughly 30 looks, so a")
        print("  Bonferroni bound wants t>3.4 and the incremental alpha does not clear it.")
    elif not track_ok or not gapplacebo_ok:
        print("  THE UNITS STORY IS NOT SUPPORTED. The gain does not track gap variability")
        print("  or does not depend on the timing of gap changes. Fall back to raw BM at")
        print("  0.760, which has already survived everything, and report the units work")
        print("  as a tested-and-rejected extension. That is still a real finding.")
    else:
        print("  MIXED. Report every check exactly as printed and headline raw BM. A")
        print("  refinement that fails any structural check is worth less than the base")
        print("  signal that failed none.")


if __name__ == "__main__":
    main()