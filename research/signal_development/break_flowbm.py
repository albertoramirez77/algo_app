"""
break_flowbm.py — try to prove the units correction is an accident.

    python break_flowbm.py --prices data/px_clean.parquet

THE CLAIM UNDER ATTACK

Weighting each month's spread return by 365.25/gap before summing raises commodity
basis-momentum from Sharpe 0.760 to 0.968, and the result survives placebo, jackknife,
concentration, subperiods and costs to 40bp.

THE COMPETING EXPLANATION

Energy and metals sit near a 30-day maturity gap; grains near 63. Dividing by the gap
therefore roughly DOUBLES energy's signal relative to grains. Because instruments are
ranked against each other, that changes who sits at the extremes of the book.

So the improvement might be nothing to do with units. It might be a static energy-and-metals
tilt that happened to pay over sixteen years. The published diagnostics cannot tell these
apart, because both produce exactly the same Sharpe.

THE DECOMPOSITION THAT CAN

    365.25/gap_t  =  (365.25/gap_const)  x  (gap_const/gap_t)
                      STATIC per-instrument   TIME-VARYING within
                      rescaling: a tilt       instrument: the real fix

    static      raw BM rescaled by a constant per instrument. A sector bet.
    timevarying within-instrument reweighting, mean multiplier 1. No cross-sectional
                tilt at all: it cannot express a sector view.

If the improvement lives in `static`, the units story is a costume. If it lives in
`timevarying`, the correction is genuine. Both is possible and would mean both effects are
real, which is a weaker but still honest claim.

EIGHT MORE WAYS TO KILL IT

  random constants   replace 1/gap with a RANDOM per-instrument constant, many draws. If
                     random tilts do as well, then any tilt helps and this one is not
                     special. This is the most dangerous test here.
  reverse scaling    multiply by gap instead of dividing. Should HURT. If it also helps,
                     the mechanism is not units.
  within sector      gaps are near-uniform inside a sector, so flow should approximately
                     equal raw there. Improvement inside sectors is evidence FOR units.
  sector neutral     demean the signal within sector before ranking, removing all
                     cross-sector tilt.
  sector jackknife   drop each sector in turn.
  volatility proxy   short-gap contracts are more volatile. Is 1/gap just 1/vol wearing a
                     calendar?
  bootstrap          block bootstrap the flow-minus-raw difference for a real confidence
                     interval, not a point estimate.
  multiple testing   an honest count of how many variants have been examined.
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
    # The listing calendar is published years ahead, so an instrument's typical gap is
    # known ex ante. Using a full-sample median here is a diagnostic decomposition, not a
    # tradeable signal, and involves no information a trader would not have.
    gconst = g["gap"].transform("median")
    m["gap_const"] = gconst

    m["bm_raw"] = g["spread"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["s_flow"] = m["spread"] * (365.25 / m["gap"])
    m["s_static"] = m["spread"] * (365.25 / gconst)
    m["s_tv"] = m["spread"] * (gconst / m["gap"])          # mean multiplier ~ 1
    m["s_rev"] = m["spread"] * (m["gap"] / 365.25)         # reverse: should hurt
    for col, nm in (("s_flow", "bm_flow"), ("s_static", "bm_static"),
                    ("s_tv", "bm_tv"), ("s_rev", "bm_rev")):
        m[nm] = m.groupby("symbol")[col].transform(
            lambda s: s.rolling(J, min_periods=J).sum())

    v = g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
    m["vol"] = v.groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
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
              min_n: int = 6, neutral_by: str | None = None,
              scale_map: dict | None = None) -> pd.Series:
    """
    `scale_map` rescales raw BM by an arbitrary per-instrument constant, used for the
    random-constant placebo.
    """
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
        if scale_map is not None:
            sv = sv * s["symbol"].map(scale_map).astype(float)
        if neutral_by:
            sv = sv - sv.groupby(s[neutral_by]).transform("mean")
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
          f"ret {s['ann']*100:>+6.2f}%{star}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--draws", type=int, default=60)
    a = ap.parse_args()

    m = load(a.prices)
    idm = idm_of(m)
    p_raw = portfolio(m, idm, "bm_raw")
    p_flow = portfolio(m, idm, "bm_flow")
    sr_raw, sr_flow = sharpe(p_raw), sharpe(p_flow)

    print("=" * 80)
    print("1. THE DECOMPOSITION — is the gain a units fix or a sector tilt?")
    print("=" * 80)
    print("  365.25/gap_t  =  (365.25/gap_const) x (gap_const/gap_t)")
    print("                    STATIC tilt          TIME-VARYING correction")
    print("  This is an identity, not a statistical model, so the split of the gain")
    print("  between the two legs is arithmetically exact.\n")
    line("raw BM (baseline)", stat(p_raw))
    line("full flow weighting", stat(p_flow), sr_raw)
    p_static = portfolio(m, idm, "bm_static")
    p_tv = portfolio(m, idm, "bm_tv")
    line("STATIC component only", stat(p_static), sr_raw)
    line("TIME-VARYING component only", stat(p_tv), sr_raw)
    print("\n  If the static leg reproduces the full gain, the improvement is a per-")
    print("  instrument tilt and the units story is decoration. If the time-varying leg")
    print("  carries it, the correction is genuine: that leg has an average multiplier of")
    print("  one and therefore cannot express any cross-sectional view at all.")

    gaps = m.groupby("symbol")["gap_const"].first().sort_values()
    print("\n  typical gap by instrument (days), and the static multiplier it implies:")
    for sym, gv in gaps.items():
        print(f"    {sym:5s} {gv:>5.0f}d   x{365.25/gv:>5.2f}   "
              f"{BY_SYMBOL[sym].sector}")

    print("\n" + "=" * 80)
    print("2. RANDOM-CONSTANT PLACEBO — would ANY per-instrument tilt have worked?")
    print("=" * 80)
    real_mult = (365.25 / gaps).to_dict()
    lo, hi = min(real_mult.values()), max(real_mult.values())
    rng = np.random.default_rng(0)
    srs = []
    for _ in range(a.draws):
        rm = {s: rng.uniform(lo, hi) for s in gaps.index}
        srs.append(sharpe(portfolio(m, idm, "bm_raw", scale_map=rm)))
    srs = np.array([x for x in srs if np.isfinite(x)])
    pct = (srs < sr_flow).mean()
    print(f"  real multipliers span {lo:.2f}x to {hi:.2f}x")
    print(f"  {len(srs)} random draws from that same range:")
    print(f"    mean {srs.mean():+.3f}   sd {srs.std(ddof=1):.3f}   "
          f"min {srs.min():+.3f}   max {srs.max():+.3f}")
    print(f"  flow {sr_flow:+.3f} beats {pct:.0%} of random tilts   "
          f"z = {(sr_flow - srs.mean())/max(srs.std(ddof=1),1e-9):+.1f}")
    rand_ok = pct > 0.90
    print(f"  {'PASS' if rand_ok else 'FAIL'} — the real gap weighting must beat random")
    print("  tilts drawn from the same range, or it is not the gaps that matter.")

    print("\n" + "=" * 80)
    print("3. REVERSE AND NULL SCALINGS")
    print("=" * 80)
    line("multiply by gap (reverse — should HURT)", stat(portfolio(m, idm, "bm_rev")), sr_raw)
    line("divide by gap (flow)", stat(p_flow), sr_raw)
    print("  If reversing the weighting ALSO improves on raw, the mechanism is not units:")
    print("  it would mean any monotone reweighting of the same signal helps.")

    print("\n" + "=" * 80)
    print("4. SECTOR ANALYSIS — where does the gain live?")
    print("=" * 80)
    print("  within sector, gaps are near-uniform, so flow should approximate raw there.")
    print("  A gain that survives INSIDE sectors is evidence for units, not tilt.\n")
    for sec, g in m.groupby("sector"):
        if g["symbol"].nunique() < 3:
            continue
        i2 = idm_of(g)
        sr_r = sharpe(portfolio(g, i2, "bm_raw", min_n=3))
        sr_f = sharpe(portfolio(g, i2, "bm_flow", min_n=3))
        if np.isfinite(sr_r) and np.isfinite(sr_f):
            print(f"    {sec:12s} ({g['symbol'].nunique()} inst)  "
                  f"raw {sr_r:>+6.3f}   flow {sr_f:>+6.3f}   diff {sr_f-sr_r:>+6.3f}")

    print("\n  sector-neutral (signal demeaned within sector, removing all sector tilt):")
    line("raw, sector-neutral", stat(portfolio(m, idm, "bm_raw", neutral_by="sector")))
    line("flow, sector-neutral", stat(portfolio(m, idm, "bm_flow", neutral_by="sector")))
    nr = sharpe(portfolio(m, idm, "bm_raw", neutral_by="sector"))
    nf = sharpe(portfolio(m, idm, "bm_flow", neutral_by="sector"))
    neutral_ok = np.isfinite(nf) and np.isfinite(nr) and nf > nr
    print(f"  {'PASS' if neutral_ok else 'FAIL'} — flow must still beat raw once every")
    print("  sector tilt has been removed by construction.")

    print("\n  sector jackknife, drop one sector at a time:")
    for sec in sorted(m["sector"].unique()):
        sub = m[m["sector"] != sec]
        if sub["symbol"].nunique() < 6:
            continue
        i2 = idm_of(sub)
        r_, f_ = sharpe(portfolio(sub, i2, "bm_raw")), sharpe(portfolio(sub, i2, "bm_flow"))
        print(f"    without {sec:12s} raw {r_:>+6.3f}   flow {f_:>+6.3f}   "
              f"diff {f_-r_:>+6.3f}")

    print("\n" + "=" * 80)
    print("5. IS 1/GAP JUST A VOLATILITY PROXY?")
    print("=" * 80)
    iv = m.groupby("symbol")["vol"].median()
    mult = 365.25 / gaps
    j = pd.concat([mult.rename("mult"), iv.rename("vol")], axis=1).dropna()
    print(f"  correlation across instruments, 1/gap multiplier vs median volatility: "
          f"{j['mult'].corr(j['vol']):+.3f}")
    vol_map = (1.0 / iv).to_dict()
    line("raw BM scaled by 1/volatility instead",
         stat(portfolio(m, idm, "bm_raw", scale_map=vol_map)), sr_raw)
    line("raw BM scaled by 1/gap (= flow)", stat(p_flow), sr_raw)
    print("  If scaling by 1/vol reproduces the gain, then 1/gap was proxying volatility")
    print("  and the calendar had nothing to do with it.")

    print("\n" + "=" * 80)
    print("6. BOOTSTRAP — a confidence interval on the DIFFERENCE, not a point estimate")
    print("=" * 80)
    j2 = pd.concat([p_flow.rename("f"), p_raw.rename("r")], axis=1).dropna()
    d = (j2["f"] - j2["r"]).to_numpy()
    n = len(d)
    block = 12
    rng2 = np.random.default_rng(1)
    boots = []
    for _ in range(2000):
        idx = []
        while len(idx) < n:
            st = rng2.integers(0, max(n - block, 1))
            idx.extend(range(st, min(st + block, n)))
        s = d[np.array(idx[:n])]
        boots.append(s.mean() * 12)
    boots = np.array(boots)
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    print(f"  mean annual return advantage of flow over raw: {d.mean()*12*100:+.2f}%")
    print(f"  block bootstrap 95% CI ({block}-month blocks, 2000 resamples): "
          f"[{lo_ci*100:+.2f}%, {hi_ci*100:+.2f}%]")
    print(f"  share of resamples where flow loses: {(boots < 0).mean():.1%}")
    boot_ok = lo_ci > 0
    print(f"  {'PASS' if boot_ok else 'FAIL'} — the interval must exclude zero")

    print("\n  advantage by subperiod:")
    n3 = len(j2) // 3
    for i, lab in enumerate(("first third ", "second third", "final third ")):
        seg = j2.iloc[i*n3:(i+1)*n3] if i < 2 else j2.iloc[2*n3:]
        print(f"    {lab}  raw {sharpe(seg['r']):>+6.3f}   flow {sharpe(seg['f']):>+6.3f}"
              f"   diff {sharpe(seg['f'])-sharpe(seg['r']):>+6.3f}")

    print("\n" + "=" * 80)
    print("7. HONEST MULTIPLE-TESTING ACCOUNT")
    print("=" * 80)
    print("  Variants examined across this project on the same data:")
    print("    5 economic hypotheses, 4 of them killed")
    print("    basis-momentum: 2 return variants x 4 formation periods x 4 asset classes")
    print("    specification ladder: 6 rungs")
    print("    parameter grid: 12 cells")
    print("    weighting schemes: 3")
    print("    units correction: 3 forms (raw, level, flow)")
    print("  A Bonferroni bound over roughly 30 meaningful looks wants t > 3.4 for 5%.")
    print(f"  The flow signal's own t is {stat(p_flow)['t']:+.2f}; the INCREMENTAL alpha")
    print("  over raw BM was t +2.72, which does NOT clear that bound. State both.")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    tv_gain = sharpe(p_tv) - sr_raw
    st_gain = sharpe(p_static) - sr_raw
    checks = [
        ("time-varying component carries a real share of the gain",
         np.isfinite(tv_gain) and tv_gain > 0.5 * (sr_flow - sr_raw)),
        ("beats 90% of random per-instrument tilts", rand_ok),
        ("survives sector neutralisation", neutral_ok),
        ("bootstrap CI on the difference excludes zero", boot_ok),
    ]
    for k, v in checks:
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  static component gain {st_gain:+.3f}   "
          f"time-varying component gain {tv_gain:+.3f}")
    print()
    if all(v for _, v in checks):
        print("  THE UNITS CORRECTION SURVIVES ADVERSARIAL TESTING. Pitch it, with the")
        print("  incremental t of 2.72 and the multiple-testing count stated alongside.")
    elif not rand_ok or not neutral_ok:
        print("  IT IS A SECTOR TILT. The gain does not survive random-tilt comparison or")
        print("  sector neutralisation, which means the calendar is incidental and the")
        print("  improvement is a bet on energy and metals. Drop the claim, keep raw BM at")
        print("  0.760, and report the units test as a tested-and-rejected extension.")
    else:
        print("  MIXED. Report every check exactly as printed. A correction that passes")
        print("  some adversarial tests and fails others is a weaker claim than raw BM")
        print("  alone, because the failures are what a PM will probe first.")


if __name__ == "__main__":
    main()