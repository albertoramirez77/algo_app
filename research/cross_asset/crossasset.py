"""
crossasset.py — does curve-residual momentum work outside commodities?

    python crossasset.py --prices px_clean.parquet

WHY THIS IS NOT A REPEAT

An earlier run reported basis-momentum by asset class and found it commodity-specific. That
run used the SPREADING variant (long deferred, short front). The variant that works, and the
one being pitched, is NEARBY — trading the front contract outright on the same signal — and
nearby has only ever been tested on commodities. The expansion is genuinely untested for the
specification in the pitch.

THE TENSION THIS RESOLVES

The pitch currently contains two different mechanisms, and they make opposite predictions.

    THE CURVE IS A UNIQUELY EFFICIENT HEDGE. The deferred contract is the only tradeable
    instrument sharing the underlying exactly, so it removes the common price component with
    one regressor where eight principal components need many and still fall short. Nothing
    in that argument is specific to commodities. It should hold for any futures curve.

    THE SIGNAL IS PHYSICAL SCARCITY. A tightening curve means inventories are drawing down.
    Equity, rate and currency curves reflect financing costs and have no inventory to draw
    down, so the effect should be absent there.

If the strategy works outside commodities, the second story is wrong and the rationale must
be rebuilt around the first. If it does not, commodity-specificity becomes a PREDICTION the
data confirms rather than a limitation the pitch apologises for. Both are defensible. Only
one is true, and the pitch cannot assert both.

WHAT IS MEASURED

  1  each asset class as its own sleeve, identical construction throughout
  2  the correlation between sleeves, which is where diversification comes from
  3  a combined book, ranked WITHIN asset class so a corn signal is never compared with a
     bond signal, then sized and volatility-targeted as one portfolio
  4  the channels test per asset class — is the curve a uniquely efficient hedge everywhere,
     or only where there is something physical to store?

Point 4 is the one that decides which mechanism the pitch should claim.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

CAPITAL, VOL_TARGET, IDM_CAP, J, VOL_WINDOW = 450_000.0, 0.20, 2.5, 12, 6

# minimum names needed to form a cross-section. Equity index has only four instruments,
# so a long-short book there is two against two — thin, and labelled as such.
MIN_N = {"commodity": 6, "fx": 4, "rates": 3, "equity": 3}


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
    df = df[df["asset"] != "?"].copy()
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                px=("settle_0", "last"), nd=("r0", "size")).reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m = m[m["nd"] >= 10].sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["mom0"] = g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["mom1"] = g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm"] = m["mom0"] - m["mom1"]
    m["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(m["symbol"]).shift(1)
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


def book(m: pd.DataFrame, idm: float, bps: float = 3.0, seed: int | None = None,
         within: str | None = None, min_n: int = 6):
    """
    `within` ranks the signal inside each group before pooling, so a corn signal is never
    compared with a bond signal. Without it, ranks would mix incommensurable quantities.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    prev, out, zero, tot = {}, {}, 0, 0
    for ym, g in m.groupby("ym"):
        cols = ["symbol", "bm", "vol", "px_entry", "fwd"] + ([within] if within else [])
        s = g[cols].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        if within:
            # rank inside each group, then centre so groups contribute symmetrically
            r = s.groupby(within)["bm"].rank(pct=True) - 0.5
        else:
            rk = s["bm"].rank()
            r = rk - rk.mean()
        w = r.to_numpy()
        if rng is not None:
            w = rng.permutation(w)
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
            tgt = wi * CAPITAL * VOL_TARGET * idm / den
            n = float(np.round(tgt))
            tot += 1
            if n == 0 and abs(tgt) > 1e-9:
                zero += 1
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
    return pd.Series(out).sort_index(), dict(zero=zero / max(tot, 1))


def st(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 36:
        return dict(n=len(r), sharpe=np.nan, t=np.nan, ann=np.nan, vol=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), yrs=yrs, sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12,
                vol=av, dd=float((eq / eq.cummax() - 1).min()))


def line(lbl: str, s: dict, extra: str = "") -> None:
    if not np.isfinite(s["sharpe"]):
        print(f"  {lbl:30s} insufficient months ({s['n']})"); return
    star = " *" if abs(s["t"]) > 2 else "  "
    print(f"  {lbl:30s} SR {s['sharpe']:>+6.3f}  t {s['t']:>+5.2f}{star} "
          f"ret {s['ann']*100:>+6.2f}%  vol {s['vol']*100:>5.1f}%  "
          f"dd {s['dd']*100:>+6.1f}%  n {s['n']:>3d} {extra}")


def r2(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    if len(y) < 250 or y.var() <= 0:
        return np.nan
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.pinv(A.T @ A) @ (A.T @ y)
    return float(1.0 - (y - A @ b).var() / y.var())


def channels_by_asset(path: str) -> pd.DataFrame:
    """Is the curve a uniquely efficient hedge everywhere, or only in commodities?"""
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["contract_0"] != raw["contract_1"]]
    raw = (raw.sort_values(["symbol", "date", "oi_0"], na_position="first")
              .drop_duplicates(["date", "symbol"], keep="last"))
    raw["asset"] = raw["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    raw = raw[raw["asset"] != "?"].sort_values(["symbol", "date"])
    for leg in ("0", "1"):
        blk = raw.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = raw.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw[f"r{leg}"] = np.log(raw[f"settle_{leg}"] / prev)
        raw.loc[~np.isfinite(raw[f"r{leg}"]), f"r{leg}"] = np.nan
    rows = []
    for asset, g in raw.groupby("asset"):
        p0 = g.pivot_table(index="date", columns="symbol", values="r0").sort_index()
        p1 = g.pivot_table(index="date", columns="symbol", values="r1").sort_index()
        # pivot_table drops any date where every value is NaN, so the two legs can end up
        # with different indices and the regression fails on a length mismatch. Align them
        # on the union of dates before anything is regressed.
        idx = p0.index.union(p1.index)
        p0 = p0.reindex(idx)
        p1 = p1.reindex(idx)
        syms = [s for s in p0.columns if p0[s].notna().sum() > 400]
        if len(syms) < 3:
            continue
        cur, peer, pcs = [], [], []
        for s in syms:
            y = p0[s].to_numpy()
            if s in p1.columns:
                v = r2(y, p1[s].to_numpy().reshape(-1, 1))
                if np.isfinite(v):
                    cur.append(v)
            others = [o for o in syms if o != s]
            if len(others) >= 2:
                best = max((r2(y, p0[o].to_numpy().reshape(-1, 1)) for o in others),
                           default=np.nan)
                if np.isfinite(best):
                    peer.append(best)
                A = p0[others].fillna(0.0).to_numpy()
                Ac = A - A.mean(axis=0, keepdims=True)
                try:
                    U, S, _ = np.linalg.svd(Ac, full_matrices=False)
                    k = min(5, U.shape[1])
                    v = r2(y, (U * S)[:, :k])
                    if np.isfinite(v):
                        pcs.append(v)
                except np.linalg.LinAlgError:
                    pass
        rows.append(dict(asset=asset, n=len(syms),
                         curve=np.mean(cur) if cur else np.nan,
                         best_peer=np.mean(peer) if peer else np.nan,
                         pca5=np.mean(pcs) if pcs else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=25)
    a = ap.parse_args()

    m = load(a.prices)

    print("=" * 82)
    print("1. COVERAGE BY ASSET CLASS")
    print("=" * 82)
    for asset, g in m.groupby("asset"):
        print(f"  {asset:10s} {g['symbol'].nunique():>2d} instruments   "
              f"{g['ym'].nunique():>3d} months   "
              f"{g['ym'].min()} to {g['ym'].max()}   "
              f"{', '.join(sorted(g['symbol'].unique())[:9])}"
              f"{'...' if g['symbol'].nunique() > 9 else ''}")

    print("\n" + "=" * 82)
    print("2. EACH ASSET CLASS AS ITS OWN SLEEVE")
    print("=" * 82)
    print("  Identical construction throughout: same signal, same 12-month formation, same")
    print("  inverse-volatility sizing, integer contracts, 3bp per side.\n")
    sleeves = {}
    for asset in ("commodity", "fx", "rates", "equity"):
        sub = m[m["asset"] == asset]
        if sub.empty or sub["symbol"].nunique() < 3:
            continue
        r, aux = book(sub, idm_of(sub), min_n=MIN_N.get(asset, 4))
        s = st(r)
        if np.isfinite(s["sharpe"]):
            sleeves[asset] = r
        thin = "  [thin cross-section]" if sub["symbol"].nunique() < 6 else ""
        line(f"{asset} ({sub['symbol'].nunique()} inst)", s,
             f"zeroed {aux['zero']*100:.0f}%{thin}")

    print("\n  Commodity is the published domain. Currencies have an independent")
    print("  replication in the literature. Rates and equity index have neither, so a")
    print("  positive result there is a genuine extension and a null is not a failure.")

    if len(sleeves) >= 2:
        print("\n" + "=" * 82)
        print("3. ARE THE SLEEVES UNCORRELATED? — where diversification would come from")
        print("=" * 82)
        S = pd.DataFrame(sleeves).dropna(how="all")
        C = S.corr()
        print(f"  {'':12s}" + "".join(f"{c:>12s}" for c in C.columns))
        for i in C.index:
            print(f"  {i:12s}" + "".join(f"{C.at[i,j]:>12.3f}" for j in C.columns))
        off = C.to_numpy()[np.triu_indices_from(C.to_numpy(), k=1)]
        print(f"\n  mean pairwise correlation between sleeves: {np.nanmean(off):+.3f}")
        print("  Low correlation is what makes a multi-asset version worth more than the")
        print("  sum of its parts, and it is the whole argument for expanding at all.")

        print("\n" + "=" * 82)
        print("4. COMBINING THE SLEEVES")
        print("=" * 82)
        eq = S.mean(axis=1).dropna()
        line("equal risk weight", st(eq))
        srs = {k: st(v)["sharpe"] for k, v in sleeves.items()}
        pos = {k: v for k, v in srs.items() if np.isfinite(v) and v > 0}
        if len(pos) >= 2:
            wts = pd.Series(pos) / sum(pos.values())
            wtd = (S[list(wts.index)] * wts).sum(axis=1).dropna()
            line("Sharpe-weighted", st(wtd),
                 "  " + " ".join(f"{k}:{v:.0%}" for k, v in wts.items()))
        base = st(sleeves.get("commodity", pd.Series(dtype=float)))
        if np.isfinite(base["sharpe"]):
            print(f"\n  commodity alone {base['sharpe']:+.3f}   "
                  f"equal-weight combination {st(eq)['sharpe']:+.3f}   "
                  f"gain {st(eq)['sharpe']-base['sharpe']:+.3f}")

    print("\n" + "=" * 82)
    print("5. ONE INTEGRATED BOOK — ranked WITHIN asset class")
    print("=" * 82)
    print("  A pooled rank would compare a corn signal with a bond signal, which are not")
    print("  the same quantity. Ranking inside each class first makes them commensurable.\n")
    r_all, aux_all = book(m, idm_of(m), within="asset", min_n=12)
    line("all 35 instruments", st(r_all), f"zeroed {aux_all['zero']*100:.0f}%")
    comm = m[m["asset"] == "commodity"]
    r_c, aux_c = book(comm, idm_of(comm), min_n=6)
    line("commodities only (17)", st(r_c), f"zeroed {aux_c['zero']*100:.0f}%")
    print("\n  Watch the 'zeroed' column. Thirty-five instruments on $450,000 means each")
    print("  position is smaller, so more of them round away to nothing. Breadth bought")
    print("  with granularity is not breadth.")

    print("\n" + "=" * 82)
    print("6. PLACEBO ON THE INTEGRATED BOOK")
    print("=" * 82)
    ts = []
    for sd in range(a.seeds):
        v = st(book(m, idm_of(m), within="asset", min_n=12, seed=sd)[0])["t"]
        if np.isfinite(v):
            ts.append(v)
    if ts:
        ts = np.array(ts)
        real = st(r_all)["t"]
        z = (real - ts.mean()) / max(ts.std(ddof=1), 1e-9)
        print(f"  placebo t {ts.mean():+.2f} ± {ts.std(ddof=1):.2f} over {len(ts)} shuffles")
        print(f"  real t {real:+.2f} sits {z:+.1f} sd out   "
              f"{'PASS' if z > 2 else 'FAIL'}")

    print("\n" + "=" * 82)
    print("7. THE MECHANISM TEST — is the curve special everywhere?")
    print("=" * 82)
    print("  This decides which story the pitch should tell. If the deferred contract")
    print("  dominates in every asset class, the mechanism is the hedge and the rationale")
    print("  must be rewritten. If it dominates only in commodities, physical scarcity is")
    print("  the story and commodity-specificity is a prediction, not a limitation.\n")
    ch = channels_by_asset(a.prices)
    if not ch.empty:
        print(f"  {'asset':12s} {'n':>3s} {'curve R2':>10s} {'best peer':>11s} "
              f"{'5 PCs':>8s} {'curve edge':>12s}")
        for _, r in ch.iterrows():
            edge = r["curve"] - r["best_peer"] if np.isfinite(r["best_peer"]) else np.nan
            print(f"  {r['asset']:12s} {int(r['n']):>3d} {r['curve']:>10.3f} "
                  f"{r['best_peer']:>11.3f} {r['pca5']:>8.3f} {edge:>+12.3f}")
        wins = int((ch["curve"] > ch["best_peer"]).sum())
        print(f"\n  the curve is the better hedge in {wins} of {len(ch)} asset classes")

    print("\n" + "=" * 82)
    print("WHAT TO DO WITH THIS")
    print("=" * 82)
    base_sr = st(sleeves.get("commodity", pd.Series(dtype=float)))["sharpe"] \
        if "commodity" in sleeves else np.nan
    all_sr = st(r_all)["sharpe"]
    print(f"  commodities only      {base_sr:+.3f}")
    print(f"  all 35 integrated     {all_sr:+.3f}")
    print()
    if np.isfinite(all_sr) and np.isfinite(base_sr) and all_sr > base_sr + 0.10:
        print("  EXPAND. The strategy is not commodity-specific, which means the economic")
        print("  rationale in the pitch is wrong as written. Rebuild it around the hedge:")
        print("  the deferred contract is the only instrument sharing the underlying")
        print("  exactly, and that is true of any futures curve. This is a stronger claim")
        print("  than the scarcity story, and it satisfies the guideline about applying")
        print("  across asset classes directly rather than by argument.")
    elif np.isfinite(all_sr) and np.isfinite(base_sr) and all_sr > base_sr - 0.05:
        print("  MIXED. Expansion neither helps nor hurts materially. Report the")
        print("  asset-class table as evidence the strategy was TESTED across markets, keep")
        print("  commodities as the traded universe on capacity grounds, and let the")
        print("  breadth question be answered with data rather than assertion.")
    else:
        print("  DO NOT EXPAND. The effect is commodity-specific, which CONFIRMS the")
        print("  rationale already in the pitch: commodity curves reflect storage and")
        print("  scarcity, financial curves reflect financing and have no inventory to")
        print("  draw down. State that you tested all four asset classes and the result")
        print("  matched the prediction. A tested boundary is stronger than an untested")
        print("  claim of generality.")


if __name__ == "__main__":
    main()