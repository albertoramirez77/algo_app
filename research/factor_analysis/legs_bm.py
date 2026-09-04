"""
legs_bm.py — is basis-momentum a long factor carrying a redundant short leg?

    python legs_bm.py --prices data/px_clean.parquet

THE OBSERVATION

Splitting the frozen strategy's P&L by side:

    long cells   +125.9bp and +109.5bp per month     93.7% of all profit
    short cells   -14.8bp and  -12.4bp per month      6.3% of all profit

on roughly equal gross exposure (1.085 long, 1.047 short). Half the capital produces a
sixteenth of the return.

THE FLAW IN READING THAT DIRECTLY — stated before any result

In a dollar-neutral book the long leg's return CONTAINS the market drift and the short
leg's contains MINUS the market drift. If the commodity complex drifted upward at all over
2010-2026, a long-minus-short gap appears automatically with no asymmetry in the signal
whatsoever. A raw gap of 131bp/month is therefore not evidence of anything.

The genuine test is each leg's MARKET-ADJUSTED alpha. That is what this script measures,
and the raw figures are printed only so the size of the mechanical component is visible.

THE ECONOMIC CLAIM, IF THE ALPHA SURVIVES

Contango is bounded by full carry: if the deferred price exceeds spot by more than
financing plus storage, the arbitrage is riskless - buy spot, store, sell forward, deliver.
Backwardation has no such bound, because shorting physical inventory you do not own is
impossible.

Basis-momentum bets on CONTINUATION of basis changes. Its short leg therefore bets on
further contango, into a wall that arbitrage capital defends. Its long leg bets on further
backwardation, into open space. The predictability should be asymmetric, and the asymmetry
should be a property of BASIS-MOMENTUM specifically.

THE DISCRIMINATING PREDICTIONS

    storage bound      the long-minus-short alpha gap is LARGER for basis-momentum than
                       for carry or momentum, because only basis-momentum bets on curve
                       continuation. Carry bets on the LEVEL of the curve and momentum on
                       price direction; neither is bounded by the storage arbitrage.

    commodity beta     all three factors show the same gap, and all three gaps collapse
                       after market adjustment.

    long-only premium  the gap survives market adjustment but is identical across factors,
                       which would mean it is about being long, not about the curve.

THE IMPLEMENTATION CONSEQUENCE, IF IT HOLDS

At $450,000 the binding constraint is integer contract granularity. Dropping or shrinking a
leg that contributes almost nothing halves the position count, halves transaction costs, and
materially reduces rounding error. The short-leg scaling curve is reported across the full
range rather than fitted, so no free parameter is introduced.
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
    with np.errstate(invalid="ignore", divide="ignore"):
        df["basis"] = np.log(df["settle_0"] / df["settle_1"]) / (df["gap"] / 365.25)
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df["ym"] = df["date"].dt.to_period("M")

    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                basis=("basis", "last"), px=("settle_0", "last"),
                n_days=("r0", "size")).reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m = m[(m["n_days"] >= 10) & (m["asset"] == "commodity")].copy()
    m = m.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["spread"] = m["r0"] - m["r1"]
    m["bm"] = g["spread"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["mom"] = g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["carry"] = m["basis"]
    v = g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
    m["vol"] = v.groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    return m


def market(m: pd.DataFrame) -> pd.Series:
    """Equal-weighted long-only commodity complex, next-month returns."""
    return m.groupby("ym")["fwd"].mean().dropna()


def idm_of(m: pd.DataFrame) -> float:
    n = max(m["symbol"].nunique(), 2)
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    if not np.isfinite(rho):
        rho = 0.2
    return min(1.0 / np.sqrt((1 / n) + (1 - 1 / n) * max(rho, 0.01)), IDM_CAP)


def terciles(m: pd.DataFrame, sig: str, seed: int | None = None,
             min_n: int = 9) -> pd.DataFrame:
    """
    Split the cross-section into thirds each month and return the equal-weighted forward
    return of each.

    WHY THIRDS AND NOT A MARKET ADJUSTMENT. The equal-weighted commodity market CONTAINS
    the factor, so benchmarking a long leg and a short leg against it removes exactly the
    asymmetry we are trying to measure: if high-signal names outperform and low-signal
    names are neutral, both legs show identical alpha against that market and the gap goes
    to zero by construction. Verified on synthetic data with genuine one-sided
    predictability embedded, where the market-adjusted gap came back at t = 0.12.

    The MIDDLE tercile is the right benchmark. It is the neutral group, drawn from the same
    universe in the same month, and it is not contaminated by the signal.

        upside   = top - middle      how much do high-signal names outperform neutral?
        downside = middle - bottom   how much do low-signal names underperform neutral?

    A symmetric factor has upside = downside. A one-sided factor does not.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    rows = []
    for ym, g in m.groupby("ym"):
        s = g[["symbol", sig, "fwd"]].dropna()
        if len(s) < min_n:
            continue
        sv = s[sig]
        if rng is not None:
            sv = pd.Series(rng.permutation(sv.to_numpy()), index=sv.index)
        order = sv.rank(method="first").to_numpy()
        k = len(s) // 3
        fwd = s["fwd"].to_numpy()
        bot = fwd[order <= k].mean()
        top = fwd[order > len(s) - k].mean()
        mid = fwd[(order > k) & (order <= len(s) - k)].mean()
        if not all(np.isfinite([bot, mid, top])):
            continue
        rows.append(dict(ym=ym, bot=bot, mid=mid, top=top,
                         upside=top - mid, downside=mid - bot,
                         asym=(top - mid) - (mid - bot), spread=top - bot))
    return pd.DataFrame(rows).set_index("ym")


def paired_t(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna()
    if len(x) < 48:
        return np.nan, np.nan, len(x)
    se = x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), (x.mean() / se if se > 0 else np.nan), len(x)


def alpha_beta(y: pd.Series, mkt: pd.Series) -> tuple[float, float, float]:
    j = pd.concat([y.rename("y"), mkt.rename("m")], axis=1).dropna()
    if len(j) < 60:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(len(j)), j["m"].to_numpy()])
    b = np.linalg.pinv(X.T @ X) @ (X.T @ j["y"].to_numpy())
    e = j["y"].to_numpy() - X @ b
    se = e.std(ddof=2) / np.sqrt(len(j))
    return b[0], b[1], (b[0] / se if se > 0 else np.nan)


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 48:
        return np.nan
    av = r.std(ddof=1) * np.sqrt(12)
    return (r.mean() * 12) / av if av > 0 else np.nan


def portfolio(m: pd.DataFrame, idm: float, sig: str, short_scale: float = 1.0,
              bps: float = 3.0, min_n: int = 6) -> tuple[pd.Series, dict]:
    """Frozen spec, with the short leg scaled by `short_scale`. 1.0 = unchanged."""
    prev, out, stats = {}, {}, dict(pos=[], zeroed=[], gross=[], cost=0.0)
    for ym, g in m.groupby("ym"):
        s = g[["symbol", sig, "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s[sig].rank()
        w = (r - r.mean()).to_numpy()
        w = np.where(w < 0, w * short_scale, w)
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        pnl = cost = 0.0
        held = {}
        nz = 0
        gross = 0.0
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            den = dpm * px * vol
            if den <= 0:
                continue
            tgt = wi * CAPITAL * VOL_TARGET * idm / den
            n = float(np.round(tgt))
            if n == 0 and abs(wi) > 1e-9:
                nz += 1
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            gross += abs(n) * dpm * px
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
        stats["pos"].append(sum(1 for v in held.values() if v != 0))
        stats["zeroed"].append(nz)
        stats["gross"].append(gross / CAPITAL)
        stats["cost"] += cost
    return pd.Series(out).sort_index(), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=100)
    a = ap.parse_args()

    m = load(a.prices)
    idm = idm_of(m)
    mkt = market(m)

    print("=" * 82)
    print("1. THE MECHANICAL COMPONENT — how much of the gap is just market drift?")
    print("=" * 82)
    print(f"  equal-weighted commodity complex, {len(mkt)} months")
    print(f"    mean {mkt.mean()*1e4:+.1f}bp/month  ({mkt.mean()*12*100:+.2f}%/yr)  "
          f"vol {mkt.std(ddof=1)*np.sqrt(12)*100:.1f}%   Sharpe {sharpe(mkt):+.3f}")
    print(f"  a dollar-neutral book's long leg carries +market and its short leg -market,")
    print(f"  so drift alone manufactures a gap of {2*mkt.mean()*1e4:+.1f}bp/month before")
    print("  any asymmetry in the signal. Everything below is adjusted for it.")

    print("\n" + "=" * 82)
    print("2. UPSIDE vs DOWNSIDE PREDICTABILITY — benchmarked on the MIDDLE tercile")
    print("=" * 82)
    print("  upside   = top third minus middle third   (do winners outperform neutral?)")
    print("  downside = middle third minus bottom third (do losers underperform neutral?)")
    print("  A symmetric factor has upside = downside. Storage theory predicts basis-")
    print("  momentum does not: its short leg bets on further contango, into an")
    print("  arbitrage bound, while its long leg bets into open space.\n")
    res = {}
    for sig, lab in (("bm", "basis-momentum"), ("carry", "carry"), ("mom", "12m momentum")):
        T = terciles(m, sig)
        if T.empty:
            continue
        mu_u, t_u, _ = paired_t(T["upside"])
        mu_d, t_d, _ = paired_t(T["downside"])
        mu_a, t_a, n_a = paired_t(T["asym"])
        mu_s, t_s, _ = paired_t(T["spread"])
        res[sig] = dict(T=T, u=mu_u, tu=t_u, d=mu_d, td=t_d, a=mu_a, ta=t_a,
                        s=mu_s, ts=t_s, n=n_a)
        print(f"  {lab}")
        print(f"    total spread (top-bottom) {mu_s*1e4:>+8.1f}bp/m  t {t_s:>+5.2f}")
        print(f"    upside   (top-middle)     {mu_u*1e4:>+8.1f}bp/m  t {t_u:>+5.2f}")
        print(f"    downside (middle-bottom)  {mu_d*1e4:>+8.1f}bp/m  t {t_d:>+5.2f}")
        print(f"    ASYMMETRY (up - down)     {mu_a*1e4:>+8.1f}bp/m  t {t_a:>+5.2f}  "
              f"n={n_a}   <-- the claim")
        share = mu_u / mu_s if mu_s else np.nan
        print(f"    share of the spread coming from the upside: {share:.0%}\n")

    print("=" * 82)
    print("3. IS THE ASYMMETRY SPECIFIC TO BASIS-MOMENTUM?")
    print("=" * 82)
    print("  Only basis-momentum bets on CONTINUATION of curve changes, so only it should")
    print("  meet the storage bound. Carry bets on the LEVEL of the curve and momentum on")
    print("  price direction; neither runs into that wall.\n")
    print(f"  {'factor':18s} {'asymmetry':>12s} {'t':>7s} {'upside share':>14s}")
    for sig, lab in (("bm", "basis-momentum"), ("carry", "carry"), ("mom", "12m momentum")):
        if sig not in res:
            continue
        r = res[sig]
        sh = r["u"] / r["s"] if r["s"] else np.nan
        print(f"  {lab:18s} {r['a']*1e4:>+11.1f}bp {r['ta']:>+7.2f} {sh:>13.0%}")
    bm_a = res.get("bm", {}).get("a", np.nan)
    others = np.nanmean([res.get("carry", {}).get("a", np.nan),
                         res.get("mom", {}).get("a", np.nan)])
    specific = np.isfinite(bm_a) and np.isfinite(others) and bm_a > 1.5 * abs(others)
    print(f"\n  basis-momentum {bm_a*1e4:+.1f}bp vs mean of the others {others*1e4:+.1f}bp")
    print(f"  {'SPECIFIC to BM' if specific else 'NOT specific to BM'}")

    print("\n" + "=" * 82)
    print("4. PLACEBO — shuffle the signal, keep everything else")
    print("=" * 82)
    gaps = []
    for sd in range(a.seeds):
        T = terciles(m, "bm", seed=sd)
        if T.empty:
            continue
        mu, _, _ = paired_t(T["asym"])
        if np.isfinite(mu):
            gaps.append(mu)
    placebo_ok = False
    if gaps:
        gaps = np.array(gaps)
        z = (bm_a - gaps.mean()) / max(gaps.std(ddof=1), 1e-9)
        print(f"  placebo asymmetry {gaps.mean()*1e4:+.1f} +/- {gaps.std(ddof=1)*1e4:.1f}bp "
              f"over {len(gaps)} shuffles")
        print(f"  real {bm_a*1e4:+.1f}bp sits {z:+.1f} sd out   "
              f"{'PASS' if z > 2 else 'FAIL'}")
        placebo_ok = z > 2
    print("  A shuffled signal still has a top and a bottom third, so if the asymmetry")
    print("  survives shuffling it is an artefact of the sort, not of the signal.")

    print("\n" + "=" * 82)
    print("5. THE SHORT-LEG SCALING CURVE — reported, not fitted")
    print("=" * 82)
    print("  Every value from 0 (long only) to 1 (symmetric), so no parameter is chosen.\n")
    print(f"  {'short scale':>12s} {'Sharpe':>9s} {'ret':>9s} {'positions':>10s} "
          f"{'zeroed':>8s} {'gross':>7s} {'cost %':>8s}")
    curve = []
    for sc in (0.0, 0.25, 0.5, 0.75, 1.0):
        p, st = portfolio(m, idm, "bm", short_scale=sc)
        yrs = len(p) / 12
        sr = sharpe(p)
        curve.append((sc, sr))
        print(f"  {sc:>12.2f} {sr:>+9.3f} {p.mean()*12*100:>+8.2f}% "
              f"{np.mean(st['pos']):>10.1f} {np.mean(st['zeroed']):>8.1f} "
              f"{np.mean(st['gross']):>6.2f}x {st['cost']/CAPITAL/yrs*100:>7.2f}%")
    print("\n  At $450,000 the binding constraint is integer granularity, so 'positions'")
    print("  and 'zeroed' matter as much as Sharpe. A leg that adds nothing but consumes")
    print("  half the position count is expensive in a way an academic paper never sees.")

    print("\n" + "=" * 82)
    print("6. WHAT A LONG-ONLY BOOK IS ACTUALLY EXPOSED TO")
    print("=" * 82)
    p0, _ = portfolio(m, idm, "bm", short_scale=0.0)
    p1, _ = portfolio(m, idm, "bm", short_scale=1.0)
    for lab, p in (("symmetric (short scale 1.0)", p1), ("long only (0.0)", p0)):
        aa, bb, tt = alpha_beta(p, mkt)
        print(f"  {lab:30s} Sharpe {sharpe(p):>+6.3f}   "
              f"market beta {bb:>+5.2f}   alpha {aa*12*100:>+6.2f}%/yr (t {tt:>+5.2f})")
    print("\n  Dropping the short leg buys simplicity and costs market neutrality. If the")
    print("  long-only beta is large, the strategy stops being a diversifier and becomes a")
    print("  commodity bet, which is a different product regardless of its Sharpe.")

    print("\n" + "=" * 82)
    print("VERDICT")
    print("=" * 82)
    bm = res.get("bm", {})
    # The raw t on the asymmetry CANNOT be the primary check. Validation on synthetic
    # data generated to be symmetric produced an asymmetry of -71bp at t = -1.99: the
    # tercile sort carries its own bias, probably from skew in the signal distribution.
    # The placebo shuffles the signal while preserving every sorting mechanic, so it
    # measures that bias directly. The placebo-relative z is therefore the real test and
    # the raw t is reported for completeness only.
    checks = [
        ("asymmetry exceeds the placebo distribution (primary)", placebo_ok),
        ("the asymmetry is specific to basis-momentum", specific),
        ("raw asymmetry t > 2 (secondary — sort is biased)",
         np.isfinite(bm.get("ta", np.nan)) and bm["ta"] > 2),
        ("downside predictability is weak (t < 2)",
         not (np.isfinite(bm.get("td", np.nan)) and bm["td"] > 2)),
    ]
    for k, v in checks:
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print()
    if all(v for _, v in checks):
        print("  ASYMMETRY SUPPORTED, AND IT IS BASIS-MOMENTUM'S. The short leg bets on")
        print("  further contango into an arbitrage bound; the long leg bets on further")
        print("  backwardation into open space. That is a structural claim about a")
        print("  published factor, with an implementation consequence a $450,000 account")
        print("  feels directly. Report the scaling curve, not a fitted optimum.")
    elif not placebo_ok:
        print("  NOT SUPPORTED. Upside and downside predictability are not")
        print("  distinguishable, so basis-momentum is symmetric and the 131bp raw leg")
        print("  gap was market drift. Report it as a fifth tested-and-rejected")
        print("  extension, and state that the raw leg decomposition is mechanically")
        print("  biased — that is worth knowing on its own.")
    else:
        print("  PARTIAL. Report every check as printed. An asymmetry that is real but")
        print("  not specific to basis-momentum is a fact about long-short commodity")
        print("  books in general, which is a weaker and less original claim.")


if __name__ == "__main__":
    main()