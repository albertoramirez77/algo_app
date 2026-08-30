"""
test_squeeze.py — the convergence squeeze, tested against three pre-registered predictions.

    python test_squeeze.py --prices px_wide.parquet

Read 08_preregistration.md first. The predictions were written before this data existed.

THE HYPOTHESIS

A short in a physically-delivered futures contract must deliver the commodity. A financial
participant cannot — no warehouse, no transport, no licence. They must buy back before
First Notice Day, on a date published years in advance, regardless of price. Arbitrage
capital faces the same constraint, which is why the effect survives: a fund that sees the
front rich cannot sell it and deliver either.

WHAT MAKES THIS TESTABLE RATHER THAN A STORY

The universe contains its own control group, fixed by contract specification:

    treatment  17 commodities        delivery needs a warehouse
    control     4 equity index       cash-settled, no delivery obligation exists
                6 rates              deliverable Treasuries, any fund can hold a note
                8 FX                 deliverable currency, costless

If the effect appears in both, it is a convergence or microstructure artefact of any
contract's last days and the delivery story is wrong. The control cannot be
reverse-engineered — it was set by the exchange, not by us.

WHAT IS AND IS NOT EVIDENCE HERE

The unconditional magnitude was observed on the narrow universe BEFORE the hypothesis was
formed. It is not evidence. The evidence is P1 (delivery separates it), P2 (inventory
scarcity conditions it), and P3 (it is concentrated pre-FND). Those the earlier observation
does not determine.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

BUCKETS = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 45), (46, 90)]
NEAR, FAR = 10, (31, 90)


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    n0 = len(df)
    df = df[df["contract_0"] != df["contract_1"]]
    # keep the leg WITH open interest: pandas sorts NaN last, so na_position="first"
    # is what puts the populated row at the end for keep="last".
    df = (df.sort_values(["symbol", "date", "oi_0"], na_position="first")
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"]).reset_index(drop=True))

    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        df[f"ret_{leg}"] = df[f"settle_{leg}"] / prev - 1.0

    df["spread_ret"] = df["ret_1"] - df["ret_0"]
    df["dte"] = (df["expiry_0"] - df["date"]).dt.days
    gap = (df["expiry_1"] - df["expiry_0"]).dt.days
    df["basis"] = np.log(df["settle_0"] / df["settle_1"]) / (gap / 365.25)
    df.loc[(gap <= 0) | (gap > 400), "basis"] = np.nan

    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df["physical"] = df["symbol"].map(
        lambda s: BY_SYMBOL[s].physical if s in BY_SYMBOL else False)

    print("=" * 78)
    print("0. DATA")
    print("=" * 78)
    print(f"  {n0:,} rows in -> {len(df):,} out, {df['symbol'].nunique()} instruments")
    span = df.groupby("symbol")["date"].agg(
        lambda d: d.nunique() / max((d.max() - d.min()).days / 365.25, 1e-9))
    print(f"  sessions per year: {span.min():.0f}-{span.max():.0f}  (must be ~250)")
    print(f"  weekend rows: {(df['date'].dt.dayofweek >= 5).mean():.2%}  (must be ~0%)")
    inv = (gap <= 0).mean()
    print(f"  inverted legs: {inv:.2%}  (must be ~0%)")
    for a, g in df.groupby("asset"):
        print(f"    {a:10s} {g['symbol'].nunique():>2d} instruments  "
              f"{len(g):>7,} rows  median dte {g['dte'].median():.0f}d")
    if span.min() < 235 or span.max() > 265 or inv > 0.02:
        print("\n  DATA IS NOT CLEAN. Stop and fix before reading anything below.")
    return df


def residualise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the mechanical part. The front converges toward the second because of the
    basis; only what survives that is a candidate for forced flow. Residualised WITHIN
    asset class, because the basis-return relationship differs across them.
    """
    d = df.dropna(subset=["spread_ret", "basis", "dte"]).copy()
    d = d[(d["dte"] >= 0) & (d["dte"] <= 90)]
    d["resid"] = np.nan
    for a, g in d.groupby("asset"):
        X = np.column_stack([np.ones(len(g)), g["basis"].to_numpy()])
        y = g["spread_ret"].to_numpy()
        beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        d.loc[g.index, "resid"] = y - X @ beta
        print(f"    {a:10s} basis slope {beta[1]:+.5f}  removed")
    return d


# ----------------------------------------------------------------------------------
# inference — date-clustered throughout
# ----------------------------------------------------------------------------------

def clustered_mean(s: pd.DataFrame, col: str) -> tuple[float, float, int]:
    """Mean of daily cross-sectional averages, with a t-stat on the date series."""
    g = s.groupby("date")[col].mean().dropna()
    if len(g) < 30:
        return np.nan, np.nan, len(g)
    return g.mean(), g.mean() / (g.std(ddof=1) / np.sqrt(len(g))), len(g)


def near_minus_far(s: pd.DataFrame, col: str = "resid") -> tuple[float, float]:
    n = s[s["dte"] <= NEAR].groupby("date")[col].mean().dropna()
    f = s[(s["dte"] >= FAR[0]) & (s["dte"] <= FAR[1])].groupby("date")[col].mean().dropna()
    if len(n) < 30 or len(f) < 30:
        return np.nan, np.nan
    d = n.mean() - f.mean()
    se = np.sqrt(n.var(ddof=1) / len(n) + f.var(ddof=1) / len(f))
    return d, (d / se if se > 0 else np.nan)


def diff_of_diffs(a: pd.DataFrame, b: pd.DataFrame, col: str = "resid") -> tuple:
    """Near-minus-far in group A versus group B, with an independent-sample t."""
    def arm(s):
        n = s[s["dte"] <= NEAR].groupby("date")[col].mean().dropna()
        f = s[(s["dte"] >= FAR[0]) & (s["dte"] <= FAR[1])].groupby("date")[col].mean().dropna()
        return n, f
    an, af = arm(a); bn, bf = arm(b)
    if min(len(an), len(af), len(bn), len(bf)) < 30:
        return np.nan, np.nan, np.nan, np.nan
    da = an.mean() - af.mean()
    db = bn.mean() - bf.mean()
    va = an.var(ddof=1) / len(an) + af.var(ddof=1) / len(af)
    vb = bn.var(ddof=1) / len(bn) + bf.var(ddof=1) / len(bf)
    se = np.sqrt(va + vb)
    return da, db, da - db, ((da - db) / se if se > 0 else np.nan)


def profile(s: pd.DataFrame, label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'days to expiry':>16s} {'raw':>21s} {'basis-residual':>21s}     n")
    for lo, hi in BUCKETS:
        sub = s[(s["dte"] >= lo) & (s["dte"] <= hi)]
        if len(sub) < 100:
            continue
        rm, rt, _ = clustered_mean(sub, "spread_ret")
        em, et, nd = clustered_mean(sub, "resid")
        print(f"  {lo:>6d}-{hi:<3d}    {rm*100:>+9.4f}%/d t={rt:>+5.2f}   "
              f"{em*100:>+9.4f}%/d t={et:>+5.2f}  {nd:>6,}")


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_wide.parquet")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()

    df = load(a.prices)
    print("\n  removing the mechanical basis component, within asset class:")
    d = residualise(df)

    comm = d[d["physical"]]
    fin = d[~d["physical"]]

    # ---------------- P1 ----------------
    print("\n" + "=" * 78)
    print("P1 — DOES DELIVERY CAPABILITY SEPARATE THE EFFECT?")
    print("=" * 78)
    profile(comm, f"PHYSICALLY DELIVERED  ({comm['symbol'].nunique()} instruments)")
    profile(fin, f"FINANCIAL / CASH OR COSTLESS DELIVERY  ({fin['symbol'].nunique()})")

    cd, ct = near_minus_far(comm)
    fd, ft = near_minus_far(fin)
    da, db, dd, dt = diff_of_diffs(comm, fin)
    print(f"\n  near-minus-far, commodities  {cd*100:+.4f}%/day  t={ct:+.2f}")
    print(f"  near-minus-far, financials   {fd*100:+.4f}%/day  t={ft:+.2f}")
    print(f"  DIFFERENCE                   {dd*100:+.4f}%/day  t={dt:+.2f}")
    p1 = np.isfinite(dt) and abs(dt) > 2 and abs(ct) > 2
    print(f"  P1 {'PASS' if p1 else 'FAIL'} — predicted: commodities significant, "
          f"financials flat, difference significant")
    if np.isfinite(ft) and abs(ft) > 2 and np.isfinite(dt) and abs(dt) < 2:
        print("  ^^ financials show it TOO and the difference is not significant. That is")
        print("     a convergence or microstructure artefact of any contract's last days,")
        print("     not a delivery constraint. The hypothesis is falsified.")

    # ---------------- P2 ----------------
    print("\n" + "=" * 78)
    print("P2 — DOES INVENTORY SCARCITY CONDITION IT?")
    print("=" * 78)
    # Split on the instrument's OWN basis history, not the cross-sectional median of the
    # day. A cross-sectional split mostly sorts instruments against each other — gold is
    # structurally contangoed, cattle structurally backwardated — so it compares gold to
    # cattle rather than scarce-inventory states to abundant ones. The within-instrument
    # split isolates the state, which is what the hypothesis is about.
    #
    # Both arms are also restricted to the NEAR window and differenced directly. Building
    # a near-minus-far inside each arm and then differencing those adds the noise of two
    # far windows to a comparison that does not need them, which cost roughly a factor of
    # three in power on synthetic data with a known 4x conditioning effect embedded.
    comm = comm.copy()
    comm["basis_z"] = comm.groupby("symbol")["basis"].transform(
        lambda s: (s - s.expanding().mean()) / s.expanding().std())
    near = comm[comm["dte"] <= NEAR]
    back = near[near["basis_z"] > 0]
    cont = near[near["basis_z"] <= 0]
    bd, bt, _ = clustered_mean(back, "resid")
    kd, kt, _ = clustered_mean(cont, "resid")
    gb = back.groupby("date")["resid"].mean().dropna()
    gc = cont.groupby("date")["resid"].mean().dropna()
    if len(gb) > 30 and len(gc) > 30:
        d2 = gb.mean() - gc.mean()
        se2 = np.sqrt(gb.var(ddof=1) / len(gb) + gc.var(ddof=1) / len(gc))
        t2 = d2 / se2 if se2 > 0 else np.nan
    else:
        d2 = t2 = np.nan
    print(f"  (within-instrument basis state, near window only)")
    print(f"  backwardated  {bd*100:+.4f}%/day  t={bt:+.2f}   n={len(gb):,} days")
    print(f"  contangoed    {kd*100:+.4f}%/day  t={kt:+.2f}   n={len(gc):,} days")
    print(f"  DIFFERENCE         {d2*100:+.4f}%/day  t={t2:+.2f}")
    p2 = np.isfinite(t2) and abs(t2) > 2
    print(f"  P2 {'PASS' if p2 else 'FAIL'} — predicted: stronger where inventory is scarce")

    # ---------------- P3 ----------------
    print("\n" + "=" * 78)
    print("P3 — IS IT CONCENTRATED PRE-EXPIRY RATHER THAN SPREAD ACROSS THE CYCLE?")
    print("=" * 78)
    mids = [(lo + hi) / 2 for lo, hi in BUCKETS]
    means = [clustered_mean(comm[(comm["dte"] >= lo) & (comm["dte"] <= hi)], "resid")[0]
             for lo, hi in BUCKETS]
    ok = [(m, v) for m, v in zip(mids, means) if np.isfinite(v)]
    if len(ok) >= 5:
        x = np.array([m for m, _ in ok]); y = np.array([v for _, v in ok])
        slope = np.polyfit(x, y, 1)[0]
        near_v = np.mean([v for m, v in ok if m <= 12])
        far_v = np.mean([v for m, v in ok if m >= 25])
        conc = abs(near_v) / max(abs(far_v), 1e-12)
        print(f"  mean residual, buckets under 12 days: {near_v*100:+.4f}%/day")
        print(f"  mean residual, buckets over 25 days:  {far_v*100:+.4f}%/day")
        print(f"  concentration ratio: {conc:.1f}x")
        print(f"  linear slope across the cycle: {slope*1e4:+.4f} bp/day per day")
        p3 = conc > 3
        print(f"  P3 {'PASS' if p3 else 'FAIL'} — predicted: concentrated near expiry, "
              f"not a smooth ramp")
    else:
        p3 = False
        print("  too few populated buckets")

    # ---------------- placebo ----------------
    print("\n" + "=" * 78)
    print("PLACEBO — shuffle days-to-expiry within instrument")
    print("=" * 78)
    rng = np.random.default_rng(0)
    ts = []
    for _ in range(a.seeds):
        p = comm.copy()
        p["dte"] = p.groupby("symbol")["dte"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        _, t = near_minus_far(p)
        if np.isfinite(t):
            ts.append(t)
    ts = np.array(ts)
    if len(ts):
        z = (ct - ts.mean()) / max(ts.std(ddof=1), 1e-9)
        print(f"  placebo t: {ts.mean():+.2f} ± {ts.std(ddof=1):.2f} over {len(ts)} seeds")
        print(f"  real result sits {z:+.1f} placebo sd from the placebo mean")
        pl = abs(z) > 2
    else:
        pl = False
        print("  placebo could not run")

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for k, v in (("P1 delivery separates", p1), ("P2 inventory conditions", p2),
                 ("P3 concentrated pre-expiry", p3), ("placebo distinguishes", pl)):
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    n = sum([p1, p2, p3, pl])
    print()
    if n == 4:
        print("  All four hold. The mechanism is supported on evidence the earlier")
        print("  observation did not determine. Specify the strategy, then answer the")
        print("  capacity question separately — the economics can be right and the trade")
        print("  still too small to express at $450,000.")
    elif n >= 2 and p1:
        print("  Partial. P1 is the load-bearing one and it holds. Report exactly which")
        print("  predictions failed; do not reparameterise to rescue them.")
    else:
        print("  The hypothesis is dead as specified. Report it as dead. A pre-registered")
        print("  prediction that failed is a finding — the fund's guidelines ask for")
        print("  contingencies when the economic hypothesis does not hold.")


if __name__ == "__main__":
    main()