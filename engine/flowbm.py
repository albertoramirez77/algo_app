"""
flowbm.py — the units correction works, but only when it never crosses a roll.

    python flowbm.py --prices data/px_clean.parquet

WHAT HAPPENED

Basis-momentum is measured in inconsistent units: it sums spread returns across maturity
gaps that range from 25 to 91 days in this data, then ranks the results against each other.
Two corrections were tested.

    LEVEL   NBM = b_t - b_{t-12},  b = log(F0/F1) / (gap/365.25)
            SR -0.055 in commodities. Destroyed the signal.

    FLOW    FBM = sum over 12 months of (r0 - r1) x 365.25/gap
            SR +0.968 (t 3.77) against raw BM's +0.760 (t 2.96). Improved it, in all
            three universe tiers.

Same idea, opposite outcomes, and their cross-sectional correlation is only +0.195. They
are not variants of one signal; one of them is contaminated.

WHY THE LEVEL VERSION FAILS

The annualised basis jumps mechanically at every roll, because the new contract pair has a
different maturity gap. Corn's front-to-second gap cycles through 61, 61, 62, 91 and 90 days
as the listing calendar turns. Nothing about scarcity changed; the denominator did.

    b_t - b_{t-12}  compares two endpoint LEVELS and therefore absorbs every roll jump
                    that occurred in between
    FBM             accumulates WITHIN-CONTRACT monthly moves and never crosses a roll

This script quantifies that directly: it measures how much of the level version's variance
is roll jumps rather than price movement. If the roll term dominates, the explanation is
established rather than asserted.

WHAT ELSE IS TESTED

Everything the raw signal has already survived, applied to the flow version, because a
0.968 that has not been stress-tested is worth less than a 0.760 that has: placebo,
spanning against raw BM and against carry and momentum, jackknife, subperiods, P&L
concentration, cost sensitivity, and the combination of both signals.
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
        df[f"blk{leg}"] = blk
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
    df["is_roll"] = ((df["blk0"] != df.groupby("symbol")["blk0"].shift(1)) |
                     (df["blk1"] != df.groupby("symbol")["blk1"].shift(1))).fillna(False)

    gap = (df["expiry_1"] - df["expiry_0"]).dt.days
    df["gap"] = gap.where((gap > 0) & (gap <= 400))
    with np.errstate(invalid="ignore", divide="ignore"):
        df["basis_ann"] = np.log(df["settle_0"] / df["settle_1"]) / (df["gap"] / 365.25)
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df["ym"] = df["date"].dt.to_period("M")
    return df


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                basis_ann=("basis_ann", "last"), gap=("gap", "last"),
                px=("settle_0", "last"), n_days=("r0", "size"),
                rolls=("is_roll", "sum")).reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m = m[(m["n_days"] >= 10) & (m["asset"] == "commodity")].copy()
    m = m.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")

    m["spread"] = m["r0"] - m["r1"]
    m["spread_ann"] = m["spread"] * (365.25 / m["gap"])
    m["bm_raw"] = g["spread"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm_flow"] = g["spread_ann"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm_level"] = m["basis_ann"] - g["basis_ann"].shift(J)
    m["mom"] = g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["carry"] = m["basis_ann"]

    v = g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
    m["vol"] = v.groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    return m


def idm_of(m: pd.DataFrame) -> float:
    n = m["symbol"].nunique()
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    return min(1.0 / np.sqrt((1 / n) + (1 - 1 / n) * max(rho, 0.01)), IDM_CAP)


def portfolio(m: pd.DataFrame, idm: float, sig: str, bps: float = 3.0,
              seed: int | None = None, min_n: int = 6) -> pd.Series:
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


def line(lbl: str, s: dict) -> None:
    if not np.isfinite(s["sharpe"]):
        print(f"  {lbl:36s} n={s['n']}"); return
    star = " *" if abs(s["t"]) > 2 else ""
    print(f"  {lbl:36s} SR {s['sharpe']:>+6.3f}  t {s['t']:>+5.2f}  "
          f"ret {s['ann']*100:>+6.2f}%  dd {s['dd']*100:>+6.1f}%{star}")


def spanning(y: pd.Series, x: pd.Series, ylab: str, xlab: str) -> None:
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(j) < 60:
        return
    X = np.column_stack([np.ones(len(j)), j["x"].to_numpy()])
    b = np.linalg.pinv(X.T @ X) @ (X.T @ j["y"].to_numpy())
    e = j["y"].to_numpy() - X @ b
    se = e.std(ddof=2) / np.sqrt(len(j))
    star = " *" if abs(b[0] / se) > 2 else ""
    print(f"    {ylab:16s} on {xlab:16s} alpha {b[0]*12*100:>+6.2f}%/yr  "
          f"t {b[0]/se:>+5.2f}  beta {b[1]:>+5.2f}  rho {j['y'].corr(j['x']):>+5.2f}{star}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=25)
    a = ap.parse_args()

    df = load(a.prices)
    m = monthly(df)
    idm = idm_of(m)

    print("=" * 80)
    print("1. WHY THE LEVEL VERSION FAILS — roll contamination, measured")
    print("=" * 80)
    d = df[df["asset"] == "commodity"].copy()
    d["db"] = d.groupby("symbol")["basis_ann"].diff()
    roll_var = d.loc[d["is_roll"], "db"].pow(2).sum()
    tot_var = d["db"].pow(2).sum()
    print(f"  daily changes in the annualised basis: {d['db'].notna().sum():,} observations")
    print(f"  share of those days that are rolls: {d['is_roll'].mean():.1%}")
    print(f"  share of TOTAL squared variation occurring on roll days: "
          f"{roll_var/tot_var:.1%}")
    print(f"  mean |change| on roll days   {d.loc[d['is_roll'],'db'].abs().mean():.4f}")
    print(f"  mean |change| on other days  {d.loc[~d['is_roll'],'db'].abs().mean():.4f}")
    ratio = (d.loc[d["is_roll"], "db"].abs().mean() /
             max(d.loc[~d["is_roll"], "db"].abs().mean(), 1e-12))
    print(f"  ratio: {ratio:.1f}x")
    print("\n  b_t - b_{t-12} absorbs every one of those jumps. The gap in the denominator")
    print("  changes at each roll while nothing about scarcity does. The flow version")
    print("  accumulates within-contract moves only and never crosses a roll.")

    gp = m.groupby("symbol")["gap"].agg(["median", "min", "max"])
    print(f"\n  maturity gaps, days: median {gp['median'].min():.0f} to "
          f"{gp['median'].max():.0f} across instruments")
    print(f"  within-instrument variation: max/min gap ratio up to "
          f"{(gp['max']/gp['min']).max():.1f}x")

    print("\n" + "=" * 80)
    print("2. THE THREE SIGNALS, HEAD TO HEAD")
    print("=" * 80)
    p_raw = portfolio(m, idm, "bm_raw")
    p_flow = portfolio(m, idm, "bm_flow")
    p_lvl = portfolio(m, idm, "bm_level")
    s_raw, s_flow, s_lvl = stat(p_raw), stat(p_flow), stat(p_lvl)
    line("raw BM (as published)", s_raw)
    line("FLOW BM (gap-weighted)", s_flow)
    line("level BM (roll-contaminated)", s_lvl)
    cs = []
    for _, g in m.groupby("ym"):
        s = g[["bm_raw", "bm_flow"]].dropna()
        if len(s) >= 6 and s["bm_raw"].std() > 0 and s["bm_flow"].std() > 0:
            cs.append(s["bm_raw"].corr(s["bm_flow"]))
    print(f"\n  cross-sectional correlation, raw vs flow: {np.mean(cs):+.3f}")

    print("\n" + "=" * 80)
    print("3. PLACEBO ON THE FLOW SIGNAL")
    print("=" * 80)
    ts = []
    for sd in range(a.seeds):
        s = stat(portfolio(m, idm, "bm_flow", seed=sd))
        if np.isfinite(s["t"]):
            ts.append(s["t"])
    ts = np.array(ts)
    z = (s_flow["t"] - ts.mean()) / max(ts.std(ddof=1), 1e-9)
    print(f"  placebo t {ts.mean():+.2f} +/- {ts.std(ddof=1):.2f} over {len(ts)} seeds")
    print(f"  real t {s_flow['t']:+.2f} sits {z:+.1f} placebo sd out   "
          f"{'PASS' if abs(z) > 2 else 'FAIL'}")
    placebo_ok = abs(z) > 2

    print("\n" + "=" * 80)
    print("4. SPANNING — is the gap weighting adding anything?")
    print("=" * 80)
    spanning(p_flow, p_raw, "FLOW BM", "raw BM")
    spanning(p_raw, p_flow, "raw BM", "FLOW BM")
    print()
    p_carry = portfolio(m, idm, "carry")
    p_mom = portfolio(m, idm, "mom")
    line("carry", stat(p_carry))
    line("12m momentum", stat(p_mom))
    j = pd.concat([p_flow.rename("f"), p_carry.rename("c"), p_mom.rename("m")],
                  axis=1).dropna()
    if len(j) > 60:
        X = np.column_stack([np.ones(len(j)), j["c"].to_numpy(), j["m"].to_numpy()])
        b = np.linalg.pinv(X.T @ X) @ (X.T @ j["f"].to_numpy())
        e = j["f"].to_numpy() - X @ b
        se = e.std(ddof=3) / np.sqrt(len(j))
        print(f"\n    FLOW BM on carry and momentum: alpha {b[0]*12*100:+.2f}%/yr  "
              f"t {b[0]/se:+.2f}   beta carry {b[1]:+.3f}  beta mom {b[2]:+.3f}")

    print("\n" + "=" * 80)
    print("5. COMBINING BOTH SIGNALS")
    print("=" * 80)
    mm = m.copy()
    for c in ("bm_raw", "bm_flow"):
        mm[f"{c}_r"] = mm.groupby("ym")[c].rank(pct=True)
    mm["bm_both"] = mm[["bm_raw_r", "bm_flow_r"]].mean(axis=1)
    line("raw only", s_raw)
    line("flow only", s_flow)
    line("average of the two ranks", stat(portfolio(mm, idm, "bm_both")))
    print("  If the blend beats both, the two carry independent information and the")
    print("  honest strategy holds both rather than declaring a winner.")

    print("\n" + "=" * 80)
    print("6. ROBUSTNESS OF THE FLOW SIGNAL")
    print("=" * 80)
    print("  jackknife, drop one instrument at a time:")
    rows = []
    for sym in sorted(m["symbol"].unique()):
        sub = m[m["symbol"] != sym]
        rows.append(dict(dropped=sym, sharpe=stat(portfolio(sub, idm_of(sub), "bm_flow"))["sharpe"]))
    jk = pd.DataFrame(rows).sort_values("sharpe")
    print(f"    full {s_flow['sharpe']:+.3f}   worst {jk['sharpe'].min():+.3f} "
          f"({jk.iloc[0]['dropped']})   best {jk['sharpe'].max():+.3f}")
    jk_ok = jk["sharpe"].min() > 0.35

    print("\n  P&L concentration:")
    tot = p_flow.sum()
    conc = p_flow.nlargest(6).sum() / tot if tot != 0 else np.nan
    for k in (3, 6, 12):
        print(f"    best {k:>2d} months = {p_flow.nlargest(k).sum()/tot*100:>6.1f}%")

    print("\n  subperiods:")
    n3 = len(p_flow) // 3
    thirds = []
    for i, lab in enumerate(("first third ", "second third", "final third ")):
        seg = p_flow.iloc[i*n3:(i+1)*n3] if i < 2 else p_flow.iloc[2*n3:]
        st = stat(seg)
        thirds.append(st["sharpe"])
        line("  " + lab, st)
    yr = p_flow.groupby(p_flow.index.year).sum() * 100
    print("    annual %:", "  ".join(f"{y}:{v:+.0f}" for y, v in yr.items()))
    print(f"    positive years {int((yr > 0).sum())} of {len(yr)}")

    print("\n  cost sensitivity:")
    for bps in (3, 5, 10, 20, 40):
        line(f"  {bps}bp per side", stat(portfolio(m, idm, "bm_flow", bps=bps)))

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    checks = [
        ("flow beats raw", s_flow["sharpe"] > s_raw["sharpe"]),
        ("flow survives placebo", placebo_ok),
        ("no single instrument carries it", jk_ok),
        ("P&L not concentrated (best 6 < 60%)", np.isfinite(conc) and conc < 0.60),
        ("all thirds positive", all(np.isfinite(x) and x > 0 for x in thirds)),
        ("survives 20bp/side",
         stat(portfolio(m, idm, "bm_flow", bps=20))["sharpe"] > 0.35),
    ]
    for k, v in checks:
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print()
    if all(v for _, v in checks):
        print("  THE UNITS CORRECTION HOLDS, in the form that never crosses a roll.")
        print("  This is a defect in a published Journal of Finance factor, identified")
        print("  from first principles, corrected with no free parameter, and validated")
        print("  on the same diagnostics the original signal survived. The level-version")
        print("  failure is part of the finding, not an embarrassment: it isolates WHY")
        print("  the correction has to be applied to flows rather than levels.")
    else:
        print("  Report exactly which checks failed. A correction that improves the")
        print("  headline Sharpe but fails a structural check is worse than no")
        print("  correction, because it looks like tuning.")


if __name__ == "__main__":
    main()