"""
regenerate.py — every number in the pitch, from one file, in one run.

    python regenerate.py --prices px_clean.parquet > PITCH_NUMBERS.txt

WHY THIS EXISTS

Two inconsistencies were found days before submission, and both had the same cause.

    the headline was 0.756 on px_wide.parquet and 0.586 on px_clean.parquet, because the
    repair removed roll-date rows whose stale settlements were producing returns that never
    happened

    the pitch quoted 13.07% on 17.2% volatility while the same specification elsewhere gave
    15.54% on 20.6%, because one script computed the diversification multiplier from the
    correlation matrix and another hardcoded 2.5. Sharpe is scale-invariant so the ratios
    agreed and the discrepancy hid

Neither was a strategy error. Both came from assembling a document out of seven scripts run
at different times against different files. This script exists so that never happens again:
ONE specification, ONE file, ONE run, and every figure the pitch needs printed with the
label it appears under.

If a number is not in this output, it does not go in the pitch.
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
BETA_WINDOW, MIN_BETA = 60, 36


# ----------------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
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
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    gap = (df["expiry_1"] - df["expiry_0"]).dt.days
    with np.errstate(invalid="ignore", divide="ignore"):
        df["basis"] = np.log(df["settle_0"] / df["settle_1"]) / (gap / 365.25)
    df.loc[(gap <= 0) | (gap > 400), "basis"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                basis=("basis", "last"), px=("settle_0", "last"),
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
    return m


def book(m, sig="bm", bps=3.0, seed=None, drop=None, J_=None, vw=None, min_n=6):
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
    prev, out, zeroed, npos = {}, {}, [], []
    W = {}
    for ym, g in m.groupby("ym"):
        s = g[["symbol", sig, "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        sv = s[sig]
        if rng is not None:
            sv = pd.Series(rng.permutation(sv.to_numpy()), index=sv.index)
        r = sv.rank(); w = (r - r.mean()).to_numpy(); gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        pnl = cost = 0.0; held = {}; z = 0
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]; dpm = inst.dollar_price_mult
            den = dpm * px * vol
            if den <= 0:
                continue
            tgt = wi * CAPITAL * VOL_TARGET * IDM / den
            n = float(np.round(tgt))
            if n == 0 and abs(wi) > 1e-9:
                z += 1
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
        zeroed.append(z); npos.append(len(s))
        W[ym] = pd.Series(w, index=s["symbol"].to_numpy())
    return (pd.Series(out).sort_index(),
            dict(zero_share=np.sum(zeroed) / max(np.sum(npos), 1),
                 W=pd.DataFrame(W).T.sort_index().fillna(0.0)))


def st(r):
    r = r.dropna()
    if len(r) < 48:
        return dict(n=len(r), sharpe=np.nan, t=np.nan, ann=np.nan, vol=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), yrs=yrs, sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12,
                vol=av, dd=float((eq / eq.cummax() - 1).min()))


def ab(y, x):
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(j) < 60:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(len(j)), j["x"].to_numpy()])
    b = np.linalg.pinv(X.T @ X) @ (X.T @ j["y"].to_numpy())
    e = j["y"].to_numpy() - X @ b
    se = e.std(ddof=2) / np.sqrt(len(j))
    return b[0] * 12, b[1], (b[0] / se if se > 0 else np.nan)


def r2(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    if len(y) < 250 or y.var() <= 0:
        return np.nan
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.pinv(A.T @ A) @ (A.T @ y)
    return float(1.0 - (y - A @ b).var() / y.var())


def channels(path, m):
    """Curve / sector / market / PCA residual momentum, all on trailing loadings."""
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["contract_0"] != raw["contract_1"]]
    raw = (raw.sort_values(["symbol", "date", "oi_0"], na_position="first")
              .drop_duplicates(["date", "symbol"], keep="last"))
    raw["asset"] = raw["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    raw = raw[raw["asset"] == "commodity"].sort_values(["symbol", "date"])
    for leg in ("0", "1"):
        blk = raw.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = raw.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw[f"r{leg}"] = np.log(raw[f"settle_{leg}"] / prev)
        raw.loc[~np.isfinite(raw[f"r{leg}"]), f"r{leg}"] = np.nan
    p0 = raw.pivot_table(index="date", columns="symbol", values="r0").sort_index()
    p1 = raw.pivot_table(index="date", columns="symbol", values="r1").sort_index()
    syms = [s for s in p0.columns if p0[s].notna().sum() > 500]
    out = {}
    for s in syms:
        y = p0[s].to_numpy()
        out.setdefault("curve", []).append(
            r2(y, p1[s].to_numpy().reshape(-1, 1)) if s in p1.columns else np.nan)
        others = [o for o in syms if o != s]
        mkt = p0[others].mean(axis=1).to_numpy()
        out.setdefault("market", []).append(r2(y, mkt.reshape(-1, 1)))
        sec = [o for o in others if BY_SYMBOL[o].sector == BY_SYMBOL[s].sector]
        if sec:
            out.setdefault("sector", []).append(r2(y, p0[sec].mean(axis=1).to_numpy().reshape(-1, 1)))
        A = p0[others].fillna(0.0).to_numpy()
        Ac = A - A.mean(axis=0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(Ac, full_matrices=False)
            PC = U * S
            for k in (5, 8):
                out.setdefault(f"pca{k}", []).append(r2(y, PC[:, :k]))
        except np.linalg.LinAlgError:
            pass
    return {k: float(np.nanmean(v)) for k, v in out.items()}


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=40)
    a = ap.parse_args()

    m = load(a.prices)
    r, aux = book(m)
    s = st(r)
    mkt = m.groupby("ym")["fwd"].mean().dropna()

    print("=" * 78)
    print(f"PITCH NUMBERS — generated from {a.prices}")
    print("=" * 78)
    print(f"  instruments {m['symbol'].nunique()}   months {s['n']}   "
          f"{m['ym'].min()} to {m['ym'].max()}   IDM {IDM}   vol target {VOL_TARGET:.0%}")

    print("\n--- HEADLINE ------------------------------------------------------------")
    print(f"  Sharpe                          {s['sharpe']:.3f}")
    print(f"  t-statistic                     {s['t']:.2f}")
    print(f"  annualised return               {s['ann']*100:.2f}%")
    print(f"  annualised volatility           {s['vol']*100:.1f}%")
    print(f"  maximum drawdown                {s['dd']*100:.1f}%")
    aa, bb, at = ab(r, mkt)
    print(f"  market beta                     {bb:+.2f}")
    print(f"  market-adjusted alpha           {aa*100:+.2f}%/yr  (t {at:+.2f})")

    print("\n--- BENCHMARKS ON THE SAME UNIVERSE -------------------------------------")
    for sig, lab in (("mom0", "front momentum"), ("carry", "carry")):
        rr, _ = book(m, sig=sig)
        ss = st(rr)
        print(f"  {lab:30s} {ss['sharpe']:.3f}")
    ts = {}
    for ym, g in m.groupby("ym"):
        x = g[["symbol", "mom0", "fwd"]].dropna()
        if len(x) < 6:
            continue
        w = np.sign(x["mom0"].to_numpy())
        if np.abs(w).sum() > 0:
            ts[ym] = float((w / np.abs(w).sum() * x["fwd"].to_numpy()).sum())
    tsm = pd.Series(ts).sort_index()
    j = pd.concat([r.rename("bm"), tsm.rename("trend")], axis=1).dropna()
    print(f"  correlation to time-series momentum   {j['bm'].corr(j['trend']):+.3f}")

    print("\n--- ECONOMIC RATIONALE --------------------------------------------------")
    cs = []
    for _, g in m.groupby("ym"):
        x = g[["mom0", "mom1"]].dropna()
        if len(x) >= 6 and x["mom0"].std() > 0 and x["mom1"].std() > 0:
            cs.append(x["mom0"].corr(x["mom1"]))
    print(f"  cross-sectional correlation of the two legs   {np.mean(cs)*100:.1f}%")
    print(f"  variance of BM as a share of front momentum   "
          f"{m['bm'].var()/m['mom0'].var()*100:.1f}%")
    for sig, lab in (("mom0", "front momentum"), ("mom1", "second momentum"), ("bm", "basis-momentum")):
        rr, _ = book(m, sig=sig)
        _, b_, _ = ab(rr, mkt)
        print(f"  market beta, {lab:24s} {b_:+.3f}")
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    print(f"  average pairwise correlation across commodities "
          f"{np.nanmean(cm[np.triu_indices_from(cm,k=1)]):.3f}")

    print("\n  CHANNELS (variance of the common component removed)")
    ch = channels(a.prices, m)
    order = [("curve", "deferred contract", 1), ("pca8", "8 principal components", 8),
             ("pca5", "5 principal components", 5), ("sector", "sector peers", 1),
             ("market", "equal-weighted market", 1)]
    for k, lab, nreg in order:
        if k in ch and np.isfinite(ch[k]):
            print(f"    {lab:26s} {ch[k]*100:5.1f}%   ({nreg} regressor"
                  f"{'s' if nreg > 1 else ''})")

    print("\n--- ROBUSTNESS ----------------------------------------------------------")
    pt = []
    for sd in range(a.seeds):
        v = st(book(m, seed=sd)[0])["t"]
        if np.isfinite(v):
            pt.append(v)
    pt = np.array(pt)
    z = (s["t"] - pt.mean()) / max(pt.std(ddof=1), 1e-9)
    print(f"  placebo, {len(pt)} shuffles      t {pt.mean():+.2f} ± {pt.std(ddof=1):.2f}"
          f"   real {s['t']:+.2f}   {z:+.1f} sd")
    jk = [st(book(m, drop=x)[0])["sharpe"] for x in sorted(m["symbol"].unique())]
    jk = [v for v in jk if np.isfinite(v)]
    print(f"  jackknife                        worst {min(jk):.3f}   best {max(jk):.3f}")
    grid = {}
    for k in (6, 9, 12, 15):
        for vw in (3, 6, 12):
            grid[(k, vw)] = st(book(m, J_=k, vw=vw)[0])["sharpe"]
    ok = sum(1 for v in grid.values() if np.isfinite(v) and v > 0.35)
    print(f"  parameter grid                   {ok} of {len(grid)} cells above 0.35   "
          f"(min {min(grid.values()):.3f}, max {max(grid.values()):.3f})")
    tot = r.sum()
    print(f"  best 6 of {len(r)} months           {r.nlargest(6).sum()/tot*100:.1f}% of P&L")
    yr = r.groupby(r.index.year).sum()
    print(f"  positive calendar years          {int((yr>0).sum())} of {len(yr)}")
    print(f"  worst calendar year              {yr.idxmin()} at {yr.min()*100:+.1f}%")

    print("\n--- COSTS AND CAPACITY --------------------------------------------------")
    Wd = aux["W"]
    to = (Wd.diff().abs().sum(axis=1) / 2).iloc[1:].mean()
    print(f"  monthly one-way turnover         {to*100:.0f}% of gross")
    print(f"  positions rounding to zero       {aux['zero_share']*100:.1f}%")
    for bps in (3, 10, 20, 40):
        print(f"  Sharpe at {bps:>2d}bp per side        {st(book(m, bps=bps)[0])['sharpe']:.3f}")

    print("\n--- PORTFOLIO COMBINATION ------------------------------------------------")
    rho = j["bm"].corr(j["trend"])
    for tr in (0.6,):
        for wb in (0.20, 0.30, 0.40):
            wa = 1 - wb
            c = (wa*tr + wb*s["sharpe"]) / np.sqrt(wa**2 + wb**2 + 2*wa*wb*rho)
            print(f"  {wb:.0%} risk weight beside a {tr} trend book   combined {c:.3f}")
        na, nb = tr - rho*s["sharpe"], s["sharpe"] - rho*tr
        tt = na + nb
        if tt > 0:
            wa_, wb_ = na/tt, nb/tt
            c = (wa_*tr + wb_*s["sharpe"]) / np.sqrt(wa_**2 + wb_**2 + 2*wa_*wb_*rho)
            print(f"  optimal weight ({wb_:.0%} to this)          combined {c:.3f}")

    print("\n--- SAMPLE LIMITS --------------------------------------------------------")
    print(f"  minimum detectable Sharpe difference   {2/np.sqrt(s['yrs']):.2f}")
    print(f"  Bonferroni bound over ~30 looks        t > 3.4  "
          f"(headline t {s['t']:.2f})")

    print("\n" + "=" * 78)
    print("  Every figure above came from one specification on one file in one run.")
    print("  If a number is not here, it does not belong in the pitch.")


if __name__ == "__main__":
    main()