"""
validate_bm.py — is the surviving basis-momentum result real, and can it be traded?

    python validate_bm.py --prices px_wide.parquet

WHAT SURVIVED

Commodity basis-momentum, nearby returns, 12-month formation:
    Sharpe +0.602, t = +2.34
    placebo -0.16 +/- 1.10, real sits +2.3 placebo sd out          PASS
    alpha over 12m momentum +5.59%/yr, t = +2.42, beta 0.163       PASS
    12m momentum itself: SR -0.048.  Carry: SR +0.175 (t=0.70).

    Partial replication: published Sharpe 0.9, expected t 3.62, measured 2.34. The
    spreading variant, which Boons & Prado report alongside nearby, failed both its
    t-test and its placebo.

WHAT THIS SCRIPT ADDS

Three things the first run never tested, two of which killed earlier hypotheses:

  1  TURNOVER, CORRECTLY MEASURED. The first run reported 99.5% monthly one-way turnover
     for a signal formed over twelve months, which is not credible. The cause was a bug:
     weights were indexed by dataframe row number rather than by symbol, so each month's
     weights landed in different columns and every position registered as opened and
     closed. Fixed here by indexing on symbol.

  2  P&L CONCENTRATION. Hypothesis 1 produced a positive Sharpe in which the best 20 of
     4,900 days generated 109% of total P&L — the other 4,880 lost money collectively. A
     premium is earned steadily; a lottery ticket is not.

  3  SUBPERIOD STABILITY. A result driven by one regime is not a premium. Reported by
     third and year by year, with no smoothing.

Then the capacity question: integer contracts at $450,000, which is the binding constraint
in this account and has excluded contracts before on granularity rather than liquidity.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL, UNIVERSE
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

J = 12
CAPITAL = 450_000.0
VOL_TARGET = 0.20


# ----------------------------------------------------------------------------------

def load_monthly(path: str) -> pd.DataFrame:
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
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df["ym"] = df["date"].dt.to_period("M")

    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                date=("date", "last"), n_days=("r0", "size"),
                px=("settle_0", "last"), dvol=("r0", "std"))
           .reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m = m[m["n_days"] >= 10]
    c0 = m.groupby("symbol")["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    c1 = m.groupby("symbol")["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm"] = c0 - c1
    m["mom"] = c0
    m["fwd"] = m.groupby("symbol")["r0"].shift(-1)
    return m.sort_values(["symbol", "ym"]).reset_index(drop=True)


def portfolio(m: pd.DataFrame, sig: str, ret: str = "fwd", min_n: int = 6):
    """Weights indexed BY SYMBOL so turnover is measurable."""
    rets, W = {}, {}
    for ym, g in m.groupby("ym"):
        s = g[["symbol", sig, ret]].dropna()
        if len(s) < min_n:
            continue
        r = s[sig].rank()
        w = r - r.mean()
        gross = w.abs().sum()
        if gross <= 0:
            continue
        w = (w / gross).to_numpy()
        rets[ym] = float((w * s[ret].to_numpy()).sum())
        W[ym] = pd.Series(w, index=s["symbol"].to_numpy())
    if not rets:
        return pd.Series(dtype=float), pd.DataFrame()
    r = pd.Series(rets).sort_index()
    wdf = pd.DataFrame(W).T.sort_index().fillna(0.0)
    return r, wdf


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 60:
        return dict(n=len(r))
    yrs = len(r) / 12
    ar, av = r.mean() * 12, r.std(ddof=1) * np.sqrt(12)
    sr = ar / av if av > 0 else np.nan
    eq = np.exp(r.cumsum())
    return dict(n=len(r), years=yrs, ann_ret=ar, ann_vol=av, sharpe=sr,
                t=sr * np.sqrt(yrs), max_dd=float((eq / eq.cummax() - 1).min()),
                hit=float((r > 0).mean()), skew=float(r.skew()))


def line(label: str, s: dict) -> None:
    if "sharpe" not in s:
        print(f"  {label:30s} too few months ({s.get('n', 0)})")
        return
    print(f"  {label:30s} SR {s['sharpe']:>+6.3f}  t {s['t']:>+6.2f}  "
          f"ret {s['ann_ret']*100:>+6.2f}%  vol {s['ann_vol']*100:>5.2f}%  "
          f"dd {s['max_dd']*100:>+6.1f}%  hit {s['hit']:.0%}")


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_wide.parquet")
    a = ap.parse_args()

    m = load_monthly(a.prices)
    comm = m[m["asset"] == "commodity"]
    r, W = portfolio(comm, "bm")
    st = stats(r)

    print("=" * 78)
    print("1. THE SURVIVOR, RESTATED")
    print("=" * 78)
    line("BM nearby, commodities", st)

    print("\n" + "=" * 78)
    print("2. TURNOVER — corrected")
    print("=" * 78)
    d = W.diff().abs().sum(axis=1) / 2
    to = d.iloc[1:].mean()
    print(f"  monthly one-way turnover {to:.1%} of gross   annualised {to*12:.1f}x")
    print(f"  the first run reported 99.5% — that was a symbol-indexing bug")
    print(f"  mean absolute weight {W.abs().mean().mean():.4f}   "
          f"instruments per month {(W != 0).sum(axis=1).mean():.1f}")
    print("\n  net Sharpe after cost, ONE leg per position:")
    for bps in (1, 2, 3, 5, 10):
        drag = to * 12 * bps / 1e4
        print(f"    {bps:>2d}bp/side -> drag {drag*100:>5.2f}%/yr   "
              f"net SR {(st['ann_ret'] - drag)/st['ann_vol']:>+6.3f}")

    print("\n" + "=" * 78)
    print("3. P&L CONCENTRATION — the test that killed hypothesis 1")
    print("=" * 78)
    tot = r.sum()
    for k in (1, 3, 6, 12):
        print(f"  best {k:>2d} months = {r.nlargest(k).sum()/tot*100:>6.1f}% of total P&L")
    print(f"  months positive: {(r > 0).mean():.1%} of {len(r)}")
    med_pos = r[r > 0].median()
    med_neg = r[r < 0].median()
    print(f"  median winning month {med_pos*100:+.2f}%   "
          f"median losing month {med_neg*100:+.2f}%")
    print("\n  Hypothesis 1's best 20 of 4,900 days produced 109% of total P&L — the rest")
    print("  lost money collectively. If the best 6 months here exceed ~60%, this is the")
    print("  same pathology and the Sharpe is not a premium.")

    print("\n" + "=" * 78)
    print("4. SUBPERIOD STABILITY")
    print("=" * 78)
    n3 = len(r) // 3
    for i, lab in enumerate(("first third ", "second third", "final third ")):
        seg = r.iloc[i*n3:(i+1)*n3] if i < 2 else r.iloc[2*n3:]
        line(lab, stats(seg) if len(seg) >= 60 else dict(n=len(seg)))
        if len(seg) < 60:
            s2 = seg.dropna()
            yrs = len(s2)/12
            sr = (s2.mean()*12)/(s2.std(ddof=1)*np.sqrt(12)) if s2.std() > 0 else np.nan
            print(f"  {lab:30s} SR {sr:>+6.3f}  t {sr*np.sqrt(yrs):>+6.2f}  "
                  f"({len(s2)} months)")
    print("\n  annual returns, %:")
    yr = r.groupby(r.index.year).sum() * 100
    print("   ", "  ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))
    print(f"  positive years: {(yr > 0).sum()} of {len(yr)}")

    print("\n" + "=" * 78)
    print("5. IS IT A TREND STRATEGY IN DISGUISE?")
    print("=" * 78)
    rm, _ = portfolio(comm, "mom")
    ts_rows = {}
    for ym, g in comm.groupby("ym"):
        s = g[["symbol", "mom", "fwd"]].dropna()
        if len(s) < 6:
            continue
        w = np.sign(s["mom"].to_numpy())
        gross = np.abs(w).sum()
        if gross > 0:
            ts_rows[ym] = float((w / gross * s["fwd"].to_numpy()).sum())
    ts = pd.Series(ts_rows).sort_index()
    line("cross-sectional momentum", stats(rm))
    line("time-series momentum (trend)", stats(ts))
    j = pd.concat([r.rename("bm"), ts.rename("trend")], axis=1).dropna()
    if len(j) > 60:
        rho = j["bm"].corr(j["trend"])
        worst = j[j["trend"] <= j["trend"].quantile(0.20)]
        print(f"\n  correlation to time-series momentum: {rho:+.3f}")
        print(f"  mean BM month, unconditional          {j['bm'].mean()*100:+.3f}%")
        print(f"  mean BM month, worst quintile of trend {worst['bm'].mean()*100:+.3f}%")
        print("  The fund's stated edge is uncorrelated risk premia. Trend is the")
        print("  dominant premium in systematic macro, so a low correlation here is the")
        print("  diversification argument — and it must be structural, not incidental.")

    print("\n" + "=" * 78)
    print("6. CAPACITY — integer contracts at $450,000")
    print("=" * 78)
    last = m[m["ym"] == m["ym"].max()].set_index("symbol")
    n_inst = comm["symbol"].nunique()
    # IDM from realised correlation of the commodity book
    piv = comm.pivot_table(index="ym", columns="symbol", values="r0").dropna(axis=1, how="all")
    cm = piv.corr().to_numpy()
    rho_bar = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    idm = min(1.0 / np.sqrt((1/n_inst) + (1 - 1/n_inst) * max(rho_bar, 0.01)), 2.5)
    budget = CAPITAL * VOL_TARGET * idm / n_inst
    print(f"  average pairwise correlation {rho_bar:+.3f}   IDM {idm:.2f}")
    print(f"  per-instrument risk budget ${budget:,.0f} of annualised dollar vol\n")
    rows = []
    for inst in UNIVERSE:
        if inst.asset != "commodity" or inst.symbol not in last.index:
            continue
        px = last.at[inst.symbol, "px"]
        dv = last.at[inst.symbol, "dvol"]
        if not np.isfinite(px) or not np.isfinite(dv):
            continue
        ann_vol = dv * np.sqrt(252)
        contract_dv = inst.multiplier * px * ann_vol
        n = budget / contract_dv if contract_dv > 0 else np.nan
        rows.append(dict(symbol=inst.symbol, price=px, ann_vol=ann_vol,
                         contract_dvol=contract_dv, contracts=n,
                         rounding_err=abs(round(n) - n) / n if n > 0 else np.nan))
    cap = pd.DataFrame(rows).sort_values("contracts")
    print(cap.to_string(index=False, float_format=lambda x: f"{x:10.2f}"))
    viable = (cap["contracts"] >= 2).sum()
    print(f"\n  instruments with a median position of 2+ contracts: {viable} of {len(cap)}")
    print(f"  mean integer rounding error: {cap['rounding_err'].mean():.1%} of intended size")
    print("  Below two contracts the position cannot express the signal. That is the")
    print("  binding constraint at this account size — granularity, not liquidity.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    conc6 = r.nlargest(6).sum() / tot
    thirds = []
    for i in range(3):
        seg = r.iloc[i*n3:(i+1)*n3] if i < 2 else r.iloc[2*n3:]
        s2 = seg.dropna()
        if len(s2) > 24 and s2.std() > 0:
            thirds.append((s2.mean()*12)/(s2.std(ddof=1)*np.sqrt(12)))
    checks = [
        ("P&L not concentrated (best 6 months < 60%)", conc6 < 0.60),
        ("all three subperiods positive", len(thirds) == 3 and all(x > 0 for x in thirds)),
        ("majority of years positive", (yr > 0).mean() > 0.5),
        ("survives 3bp/side cost", (st["ann_ret"] - to*12*3/1e4)/st["ann_vol"] > 0.4),
        (f"{viable} instruments tradeable at 2+ contracts", viable >= 10),
    ]
    for k, v in checks:
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    n_pass = sum(v for _, v in checks)
    print()
    if n_pass == len(checks):
        print("  Holds up. It remains a PARTIAL replication — t 2.34 against an expected")
        print("  3.62, and the spreading variant failed. Pitch it as partial, with the")
        print("  power arithmetic beside it. Do not round 2.34 up to a confirmation.")
    else:
        print("  One or more structural checks failed. Report which, and do not")
        print("  reparameterise. A survivor that fails concentration or stability is the")
        print("  same pathology that killed hypothesis 1.")


if __name__ == "__main__":
    main()