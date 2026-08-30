"""
normbm.py — basis-momentum is measured in inconsistent units. Does fixing it help?

    python normbm.py --prices px_wide.parquet

THE DEFECT

Basis-momentum, as published, is

    BM = sum over 12 months of (r_front - r_second)
       = the 12-month change in log(F0 / F1)

That log ratio spans a maturity gap which VARIES enormously. Measured in this data:
energy and metals sit near a 30-day gap, grains near 63, soybeans 60, livestock 59-61, and
the financial contracts are quarterly at roughly 90.

A 1% spread move across a 30-day gap is a FOUR TIMES larger change in the annualised carry
rate than the same 1% move across a 120-day gap. Basis-momentum adds those as though they
were the same quantity and then ranks them against each other cross-sectionally.

The carry literature fixed this decades ago: that is precisely why the basis is annualised
by gap/365 before anything is compared. Basis-momentum is the CHANGE in that same quantity
and never received the correction.

THE FIX

    b_t   = log(F0 / F1) / (gap / 365.25)        the annualised basis, i.e. the carry rate
    NBM   = b_t - b_{t-12}                        its 12-month change

One line of algebra, no free parameter, and comparing the same calendar month a year apart
controls for agricultural seasonality at no cost. A flow variant - the sum of gap-weighted
monthly spread moves - runs as a robustness check, since the two differ whenever the gap
itself changes within an instrument.

WHY IT MATTERS MOST CROSS-ASSET

Basis-momentum has been documented in commodities (Boons & Prado, JF 2019) and separately
in currencies (Fan, EFM 2025). It has never been tested as one unified cross-asset factor,
and inconsistent units are exactly why such a test would have been incoherent. So the
prediction is directional and falsifiable: the correction should help LITTLE where gaps are
uniform and MORE as the universe widens. Three tiers, reported separately.

    tier 1   17 commodities                 published domain
    tier 2   + 8 currencies                 independently replicated
    tier 3   + 6 rates, 4 equity index      extension, no published support

Everything runs on the frozen specification: inverse-volatility scaled ranks, integer
contracts, 3bp per side, and both sizing inputs lagged one month.
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

TIERS = {
    "commodities (17)": ["commodity"],
    "+ currencies (25)": ["commodity", "fx"],
    "all 35 [extension]": ["commodity", "fx", "rates", "equity"],
}


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
        df["logbasis"] = np.log(df["settle_0"] / df["settle_1"])
        df["basis_ann"] = df["logbasis"] / (df["gap"] / 365.25)
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df["ym"] = df["date"].dt.to_period("M")

    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                basis_ann=("basis_ann", "last"), gap=("gap", "last"),
                px=("settle_0", "last"), n_days=("r0", "size")).reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m = m[m["n_days"] >= 10].copy()
    m = m.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")

    m["spread"] = m["r0"] - m["r1"]
    # published signal: unnormalised sum of spread returns
    m["bm_raw"] = g["spread"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    # corrected signal: 12-month change in the ANNUALISED basis
    m["bm_norm"] = m["basis_ann"] - g["basis_ann"].shift(J)
    # flow variant: each month's move converted to annualised units before summing
    m["spread_ann"] = m["spread"] * (365.25 / m["gap"])
    m["bm_flow"] = g["spread_ann"].transform(lambda s: s.rolling(J, min_periods=J).sum())

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
              seed: int | None = None, min_n: int = 6,
              neutral: bool = False) -> pd.Series:
    """Frozen spec. `neutral` demeans the signal within asset class before ranking."""
    rng = np.random.default_rng(seed) if seed is not None else None
    prev, out = {}, {}
    for ym, g in m.groupby("ym"):
        s = g[["symbol", sig, "vol", "px_entry", "fwd", "asset"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        sig_v = s[sig]
        if neutral:
            sig_v = sig_v - s.groupby("asset")[sig].transform("mean")
        if rng is not None:
            sig_v = pd.Series(rng.permutation(sig_v.to_numpy()), index=sig_v.index)
        r = sig_v.rank()
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
        print(f"    {lbl:38s} n={s['n']}"); return
    star = " *" if abs(s["t"]) > 2 else ""
    print(f"    {lbl:38s} SR {s['sharpe']:>+6.3f}  t {s['t']:>+5.2f}  "
          f"ret {s['ann']*100:>+6.2f}%  dd {s['dd']*100:>+6.1f}%{star}")


def spanning(y: pd.Series, x: pd.Series, ylab: str, xlab: str) -> None:
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(j) < 60:
        print(f"    {ylab} vs {xlab}: too few months"); return
    X = np.column_stack([np.ones(len(j)), j["x"].to_numpy()])
    b = np.linalg.pinv(X.T @ X) @ (X.T @ j["y"].to_numpy())
    e = j["y"].to_numpy() - X @ b
    se = e.std(ddof=2) / np.sqrt(len(j))
    print(f"    {ylab:22s} on {xlab:22s} alpha {b[0]*12*100:>+6.2f}%/yr  "
          f"t {b[0]/se:>+5.2f}   beta {b[1]:>+5.2f}   rho {j['y'].corr(j['x']):>+5.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_wide.parquet")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()

    m = load(a.prices)

    print("=" * 82)
    print("1. THE UNITS PROBLEM, QUANTIFIED")
    print("=" * 82)
    gp = m.groupby("asset")["gap"].agg(["median", "min", "max"])
    print("  maturity gap between the two legs, days:")
    print(gp.to_string(float_format=lambda x: f"{x:7.0f}"))
    for lbl, assets in TIERS.items():
        sub = m[m["asset"].isin(assets)]
        med = sub.groupby("symbol")["gap"].median()
        print(f"\n  {lbl}: gap ranges {med.min():.0f} to {med.max():.0f} days "
              f"across instruments, a {med.max()/med.min():.1f}x spread")
    print("\n  Ranking a 30-day spread move against a 90-day one without annualising")
    print("  compares quantities that are not in the same units. The correction should")
    print("  therefore help LITTLE in tier 1 and MORE as the universe widens.")

    print("\n" + "=" * 82)
    print("2. HOW DIFFERENT ARE THE TWO SIGNALS?")
    print("=" * 82)
    cs = []
    for _, g in m.groupby("ym"):
        s = g[["bm_raw", "bm_norm"]].dropna()
        if len(s) >= 6 and s["bm_raw"].std() > 0 and s["bm_norm"].std() > 0:
            cs.append(s["bm_raw"].corr(s["bm_norm"]))
    print(f"  mean cross-sectional correlation, raw vs normalised: {np.mean(cs):+.3f}")
    print("  If this were near 1.0 the correction could not matter. Below about 0.9 it is")
    print("  a materially different signal, not a cosmetic adjustment.")

    print("\n" + "=" * 82)
    print("3. HEAD TO HEAD, BY TIER")
    print("=" * 82)
    results = {}
    for lbl, assets in TIERS.items():
        sub = m[m["asset"].isin(assets)].copy()
        idm = idm_of(sub)
        neutral = len(assets) > 1
        print(f"\n  {lbl}"
              f"{'   (signal demeaned within asset class)' if neutral else ''}")
        raw = portfolio(sub, idm, "bm_raw", neutral=neutral)
        nrm = portfolio(sub, idm, "bm_norm", neutral=neutral)
        flw = portfolio(sub, idm, "bm_flow", neutral=neutral)
        sr, sn, sf = stat(raw), stat(nrm), stat(flw)
        line("raw BM (as published)", sr)
        line("normalised BM (the fix)", sn)
        line("flow variant (robustness)", sf)
        if np.isfinite(sr["sharpe"]) and np.isfinite(sn["sharpe"]):
            print(f"      improvement from normalising: "
                  f"{sn['sharpe'] - sr['sharpe']:+.3f} Sharpe")
        results[lbl] = dict(raw=raw, norm=nrm, flow=flw, sub=sub, idm=idm,
                            neutral=neutral)

    print("\n" + "=" * 82)
    print("4. SPANNING — does either signal survive the other?")
    print("=" * 82)
    for lbl, r in results.items():
        print(f"\n  {lbl}")
        spanning(r["norm"], r["raw"], "normalised BM", "raw BM")
        spanning(r["raw"], r["norm"], "raw BM", "normalised BM")
    print("\n  If normalised BM has alpha over raw and raw has none over normalised, the")
    print("  correction strictly dominates. If BOTH carry alpha, the unit error was")
    print("  capturing something independent and the honest answer is to hold both.")

    print("\n" + "=" * 82)
    print("5. PLACEBO ON THE NORMALISED SIGNAL")
    print("=" * 82)
    for lbl, r in results.items():
        real = stat(r["norm"])["t"]
        if not np.isfinite(real):
            continue
        ts = []
        for sd in range(a.seeds):
            s = stat(portfolio(r["sub"], r["idm"], "bm_norm", seed=sd,
                               neutral=r["neutral"]))
            if np.isfinite(s["t"]):
                ts.append(s["t"])
        if ts:
            ts = np.array(ts)
            z = (real - ts.mean()) / max(ts.std(ddof=1), 1e-9)
            print(f"  {lbl:22s} real t {real:+.2f}   placebo {ts.mean():+.2f} "
                  f"+/- {ts.std(ddof=1):.2f}   {z:+.1f} sd   "
                  f"{'PASS' if abs(z) > 2 else 'FAIL'}")

    print("\n" + "=" * 82)
    print("6. ROBUSTNESS OF THE BEST TIER")
    print("=" * 82)
    best = max(results.items(),
               key=lambda kv: stat(kv[1]["norm"])["sharpe"]
               if np.isfinite(stat(kv[1]["norm"])["sharpe"]) else -9)
    lbl, r = best
    p = r["norm"].dropna()
    print(f"  best tier by normalised Sharpe: {lbl}")
    line("normalised BM", stat(p))
    tot = p.sum()
    if tot != 0:
        for k in (3, 6, 12):
            print(f"    best {k:>2d} months = {p.nlargest(k).sum()/tot*100:>6.1f}% of P&L")
    n3 = len(p) // 3
    for i, t in enumerate(("first third ", "second third", "final third ")):
        seg = p.iloc[i*n3:(i+1)*n3] if i < 2 else p.iloc[2*n3:]
        line(t, stat(seg))
    yr = p.groupby(p.index.year).sum() * 100
    print("    annual %:", "  ".join(f"{y}:{v:+.0f}" for y, v in yr.items()))
    print(f"    positive years {int((yr > 0).sum())} of {len(yr)}")

    print("\n" + "=" * 82)
    print("VERDICT")
    print("=" * 82)
    print(f"  {'tier':24s} {'raw':>8s} {'norm':>8s} {'delta':>8s}")
    deltas = []
    for lbl, r in results.items():
        sr, sn = stat(r["raw"])["sharpe"], stat(r["norm"])["sharpe"]
        if np.isfinite(sr) and np.isfinite(sn):
            deltas.append(sn - sr)
            print(f"  {lbl:24s} {sr:>+8.3f} {sn:>+8.3f} {sn-sr:>+8.3f}")
    print()
    if deltas and all(d > 0 for d in deltas):
        rising = len(deltas) > 1 and deltas[-1] > deltas[0]
        print("  THE CORRECTION HELPS IN EVERY TIER.")
        if rising:
            print("  And the improvement GROWS with the universe, which is the directional")
            print("  prediction the units argument makes. That is the strongest possible")
            print("  form of this result: a defect identified from first principles, a")
            print("  correction with no free parameter, and a prediction about WHERE it")
            print("  should matter that the data confirms.")
        else:
            print("  The improvement does not grow with the universe, so the units argument")
            print("  is supported in level but not in its directional prediction. Report")
            print("  both facts.")
    elif deltas and any(d > 0 for d in deltas):
        print("  MIXED. The correction helps in some tiers and not others. Report the")
        print("  tier table exactly as printed and do not select the best one after the")
        print("  fact — that is the failure mode this project has spent five hypotheses")
        print("  learning to avoid.")
    else:
        print("  THE CORRECTION DOES NOT HELP. A plausible defect in a published factor,")
        print("  identified from first principles and tested honestly, turned out not to")
        print("  matter. That is a legitimate finding and it belongs in the pitch: it")
        print("  shows the units were checked rather than assumed.")


if __name__ == "__main__":
    main()