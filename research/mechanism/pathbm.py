"""
pathbm.py — two questions: why does basis-momentum work, and does HOW the curve moved matter?

    python pathbm.py --prices data/px_clean.parquet

PART A — WHY IT WORKS, WITHOUT INVENTING A STORY

The mechanism tests all failed: no state variable explains WHEN basis-momentum pays. But
there is a structural question that has not been asked and does not need a state variable.

    BM = momentum(front) - momentum(second)

In this sample momentum on the front is dead (SR 0.110) and carry is dead (SR 0.117), yet
their combination pays 0.760. Subtracting the deferred contract's momentum from the front's
must therefore ADD information rather than dilute it. The reason is structural: both legs
share the same spot price, so the difference cancels the common spot component and leaves
what is specific to the front — the curve. That is a claim about variance decomposition,
not about anyone's behaviour, and it is directly testable:

    if BM purges spot direction, it should carry near-zero beta to the commodity market
    factor while front momentum carries meaningful beta

Part A tests exactly that, and decomposes BM into its two legs to show that neither leg
alone works while the difference does.

PART B — THE TWIST: PATH CONSISTENCY

Basis-momentum treats two curves identically if they arrive at the same place. A curve that
steepens steadily for twelve months is pricing persistent physical tightening. A curve that
arrives at the same point through one violent month is pricing a shock, and shocks revert.
Same BM, different information.

The natural measure is Kaufman's efficiency ratio over the formation window:

    ER = |sum of monthly spread returns| / sum of |monthly spread returns|

which runs from near 0 (violent back-and-forth) to 1 (perfectly monotone).

THE TRAP, AND THE FIX. The numerator of ER *is* BM. So ER is mechanically correlated with
|BM| and an interaction between them repeats the error that made the basis interaction
uninterpretable: a signal cannot cleanly condition on a transform of itself. This script
therefore reports the raw correlation between ER and |BM|, and builds a residualised version
- ER orthogonalised against |BM| cross-sectionally each month - which is the version the
interaction and portfolio tests use.
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
    m = m.sort_values(["symbol", "ym"]).reset_index(drop=True)

    m["spread"] = m["r0"] - m["r1"]
    g = m.groupby("symbol")
    m["mom0"] = g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["mom1"] = g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm"] = m["mom0"] - m["mom1"]
    # path: total absolute travel of the spread over the same window
    m["travel"] = g["spread"].transform(
        lambda s: s.abs().rolling(J, min_periods=J).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        m["er"] = m["bm"].abs() / m["travel"]
    m.loc[~np.isfinite(m["er"]), "er"] = np.nan

    # LURCH: the share of the window's total spread movement that happened in its single
    # largest month. Near 1/J for a steady path, near 1 for a single violent move.
    #
    # This is the measure the interaction actually uses. The efficiency ratio above is
    # |BM| / travel, so orthogonalising it against |BM| — which is required, since a
    # signal cannot condition on a transform of itself — leaves little more than inverse
    # spread volatility. Lurch is scale-invariant by construction: multiply every monthly
    # move by any constant and it does not change. It is a property of the SHAPE of the
    # path, not its size, so it needs no orthogonalisation to be interpretable.
    m["lurch"] = g["spread"].transform(
        lambda s: s.abs().rolling(J, min_periods=J).max()) / m["travel"]
    m.loc[~np.isfinite(m["lurch"]), "lurch"] = np.nan
    m["steady"] = -m["lurch"]          # sign it so that higher = steadier

    v = g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
    m["vol"] = v.groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    m["fwd_spread"] = g["spread"].shift(-1)
    return m


def zc(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def orthogonalise(df: pd.DataFrame, target: str, against: str) -> pd.Series:
    """Cross-sectional residual of `target` on `against`, month by month."""
    out = pd.Series(np.nan, index=df.index)
    for ym, g in df.groupby("ym"):
        s = g[[target, against]].dropna()
        if len(s) < 6 or s[against].std() == 0:
            continue
        X = np.column_stack([np.ones(len(s)), s[against].to_numpy()])
        y = s[target].to_numpy()
        b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        out.loc[s.index] = y - X @ b
    return out


def fm(panel: pd.DataFrame, y: str, xs: list[str], min_n: int = 8) -> pd.DataFrame:
    coefs = []
    for _, g in panel.groupby("ym"):
        s = g[[y] + xs].dropna()
        if len(s) < min_n:
            continue
        X = np.column_stack([np.ones(len(s))] + [s[x].to_numpy() for x in xs])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        coefs.append(np.linalg.pinv(X.T @ X) @ (X.T @ s[y].to_numpy()))
    if len(coefs) < 60:
        return pd.DataFrame()
    C = np.array(coefs)
    rows = []
    for i, nm in enumerate(["const"] + xs):
        c = C[:, i]
        se = c.std(ddof=1) / np.sqrt(len(c))
        rows.append(dict(term=nm, coef=c.mean(), t=c.mean() / se if se > 0 else np.nan))
    return pd.DataFrame(rows)


def show_fm(title: str, tab: pd.DataFrame, focus: str | None = None) -> None:
    if tab.empty:
        print(f"  {title}: too few cross-sections"); return
    print(f"\n  {title}")
    for _, r in tab.iterrows():
        star = " *" if abs(r["t"]) > 2 else ""
        mark = "  <--" if focus and r["term"] == focus else ""
        print(f"    {r['term']:20s} {r['coef']*100:>+8.4f}%  t {r['t']:>+6.2f}{star}{mark}")


def idm_of(m: pd.DataFrame) -> float:
    n = m["symbol"].nunique()
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    return min(1.0 / np.sqrt((1/n) + (1 - 1/n) * max(rho, 0.01)), IDM_CAP)


def portfolio(m: pd.DataFrame, idm: float, sig: str, tilt: str | None = None,
              bps: float = 3.0, min_n: int = 6) -> pd.Series:
    """Frozen specification: inverse-vol scaled ranks, integer contracts, lagged inputs."""
    prev, out = {}, {}
    for ym, g in m.groupby("ym"):
        cols = ["symbol", sig, "vol", "px_entry", "fwd"] + ([tilt] if tilt else [])
        s = g[cols].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s[sig].rank()
        w = (r - r.mean()).to_numpy()
        if tilt:
            w = w * np.clip(1.0 + zc(s[tilt]).to_numpy(), 0.0, None)
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
        print(f"  {lbl:40s} n={s['n']}"); return
    star = " *" if abs(s["t"]) > 2 else ""
    print(f"  {lbl:40s} SR {s['sharpe']:>+6.3f}  t {s['t']:>+5.2f}  "
          f"ret {s['ann']*100:>+6.2f}%  dd {s['dd']*100:>+6.1f}%{star}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()

    m = load(a.prices)
    idm = idm_of(m)

    print("=" * 80)
    print("PART A — WHY DOES THE DIFFERENCE WORK WHEN NEITHER LEG DOES?")
    print("=" * 80)
    for c in ("bm", "mom0", "mom1"):
        m[f"{c}_z"] = m.groupby("ym")[c].transform(zc)
    print("\n  1. Each leg on its own, and their difference:")
    p_bm = portfolio(m, idm, "bm")
    p_m0 = portfolio(m, idm, "mom0")
    p_m1 = portfolio(m, idm, "mom1")
    line("front momentum alone", stat(p_m0))
    line("second-contract momentum alone", stat(p_m1))
    line("their difference = basis-momentum", stat(p_bm))

    cs = []
    for _, g in m.groupby("ym"):
        s = g[["mom0", "mom1"]].dropna()
        if len(s) >= 6 and s["mom0"].std() > 0 and s["mom1"].std() > 0:
            cs.append(s["mom0"].corr(s["mom1"]))
    print(f"\n  2. Cross-sectional correlation of the two legs: {np.mean(cs):+.3f}")
    vr = m["bm"].var() / m["mom0"].var()
    print(f"     variance of the difference as a share of front momentum: {vr:.1%}")
    print("     The legs move almost identically because they share a spot price. The")
    print("     difference is the small residual that does NOT, which is the curve.")

    mkt = m.groupby("ym")["r0"].mean().rename("mkt")
    print("\n  3. Beta to the commodity market factor (equal-weighted long-only):")
    for lbl, p in (("front momentum", p_m0), ("second momentum", p_m1),
                   ("basis-momentum", p_bm)):
        j = pd.concat([p.rename("p"), mkt], axis=1).dropna()
        if len(j) < 60:
            continue
        X = np.column_stack([np.ones(len(j)), j["mkt"].to_numpy()])
        y = j["p"].to_numpy()
        b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        e = y - X @ b
        se_a = e.std(ddof=2) / np.sqrt(len(j))
        se_b = e.std(ddof=2) / (j["mkt"].std(ddof=1) * np.sqrt(len(j)))
        print(f"    {lbl:22s} beta {b[1]:>+6.3f} (t {b[1]/se_b:>+5.2f})   "
              f"alpha {b[0]*12*100:>+6.2f}%/yr (t {b[0]/se_a:>+5.2f})")
    print("\n     PREDICTION: if the difference purges spot direction, basis-momentum")
    print("     should carry a materially smaller market beta than front momentum. That")
    print("     is a claim about variance decomposition, not about anyone's behaviour,")
    print("     and it is the honest answer to 'why does this work'.")

    print("\n" + "=" * 80)
    print("PART B — DOES THE PATH MATTER, NOT JUST THE DESTINATION?")
    print("=" * 80)
    print("  Two curves with identical basis-momentum can arrive very differently. One")
    print("  steepens steadily for twelve months; the other lurches there in a single")
    print("  month. The first is pricing persistent tightening, the second a shock.\n")
    er = m["er"].dropna()
    print(f"  efficiency ratio: median {er.median():.3f}  "
          f"p10 {er.quantile(.10):.3f}  p90 {er.quantile(.90):.3f}")
    corr_raw = m[["er", "bm"]].dropna().assign(ab=lambda d: d["bm"].abs())[["er", "ab"]] \
                .corr().iloc[0, 1]
    print(f"  correlation of ER with |BM|: {corr_raw:+.3f}")
    print("  ER's numerator IS |BM|, so raw ER cannot cleanly condition the signal — the")
    print("  same trap that made the basis interaction uninterpretable. Residualised:")

    m["abs_bm"] = m["bm"].abs()
    m["er_resid"] = orthogonalise(m, "er", "abs_bm")
    chk = m[["er_resid", "abs_bm"]].dropna().corr().iloc[0, 1]
    print(f"  correlation of residualised ER with |BM|: {chk:+.3f}   (should be ~0)")

    m["er_z"] = m.groupby("ym")["er_resid"].transform(zc)
    m["bm_x_er"] = m["bm_z"] * m["er_z"]

    lc = m[["lurch", "abs_bm"]].dropna().corr().iloc[0, 1]
    print(f"\n  LURCH — share of the window's total spread movement in its single largest")
    print(f"  month. Scale-invariant, so it needs no orthogonalisation.")
    print(f"    median {m['lurch'].median():.3f}  p10 {m['lurch'].quantile(.10):.3f}  "
          f"p90 {m['lurch'].quantile(.90):.3f}   (1/12 = {1/12:.3f} is perfectly even)")
    print(f"    correlation with |BM|: {lc:+.3f}   "
          f"(compare {corr_raw:+.3f} for the raw efficiency ratio)")
    m["steady_z"] = m.groupby("ym")["steady"].transform(zc)
    m["bm_x_steady"] = m["bm_z"] * m["steady_z"]

    show_fm("BM x path STEADINESS (the clean measure)",
            fm(m, "fwd", ["bm_z", "steady_z", "bm_x_steady"]), "bm_x_steady")
    show_fm("BM x efficiency ratio, residualised (the contaminated measure)",
            fm(m, "fwd", ["bm_z", "er_z", "bm_x_er"]), "bm_x_er")
    print("    A positive interaction means basis-momentum is worth more when the curve")
    print("    got there steadily. That is the twist, and it is testable rather than told.")

    print("\n  Tradeable versions:")
    line("plain basis-momentum", stat(p_bm))
    line("conviction scaled by steadiness", stat(portfolio(m, idm, "bm", tilt="steady_z")))
    line("conviction scaled by residualised ER",
         stat(portfolio(m, idm, "bm", tilt="er_z")))
    hi = m.copy()
    hi.loc[hi["steady_z"] < 0, "bm"] = np.nan      # trade only the steadier half
    line("steady paths only", stat(portfolio(hi, idm, "bm")))

    print("\n  Placebo on the interaction — shuffle path consistency across instruments")
    print("  within each month, leaving basis-momentum untouched:")
    rng = np.random.default_rng(0)
    ts = []
    for _ in range(a.seeds):
        p = m.copy()
        p["steady_z"] = p.groupby("ym")["steady_z"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        s = stat(portfolio(p, idm, "bm", tilt="steady_z"))
        if np.isfinite(s["t"]):
            ts.append(s["t"])
    real_t = stat(portfolio(m, idm, "bm", tilt="steady_z"))["t"]
    if ts:
        ts = np.array(ts)
        z = (real_t - ts.mean()) / max(ts.std(ddof=1), 1e-9)
        print(f"    placebo t {ts.mean():+.2f} +/- {ts.std(ddof=1):.2f}   "
              f"real {real_t:+.2f}   {z:+.1f} sd   {'PASS' if abs(z) > 2 else 'FAIL'}")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    tab = fm(m, "fwd", ["bm_z", "steady_z", "bm_x_steady"])
    it = tab[tab["term"] == "bm_x_steady"]["t"].iloc[0] if not tab.empty else np.nan
    tilted = stat(portfolio(m, idm, "bm", tilt="steady_z"))
    print(f"  path interaction t {it:+.2f}")
    print(f"  plain SR {stat(p_bm)['sharpe']:+.3f}   "
          f"path-tilted SR {tilted['sharpe']:+.3f}")
    print()
    if np.isfinite(it) and abs(it) > 2:
        print("  THE PATH MATTERS. This is a genuine extension of a published factor with")
        print("  an economic rationale that can be stated in one sentence and tested in")
        print("  one regression. Report the interaction, the tilted portfolio, and the")
        print("  placebo together — the twist is only worth claiming if all three hold.")
    else:
        print("  THE PATH DOES NOT SEPARATE at conventional significance. Report it as a")
        print("  tested-and-rejected extension rather than dropping it: a hypothesis that")
        print("  was specific enough to fail is evidence of method. The Part A")
        print("  decomposition still answers 'why does this work' without inventing a")
        print("  mechanism, and that is the more important of the two results.")


if __name__ == "__main__":
    main()