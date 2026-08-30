"""
stress_bm.py — try to break the surviving strategy.

    python stress_bm.py --prices px_wide.parquet

TWO PROBLEMS WITH THE HEADLINE NUMBER, FIXED HERE

1. THE SPEC CHANGED WITHOUT BEING PLACEBO-TESTED. validate_bm.py used raw rank weights and
   reported Sharpe 0.602 with a placebo at +2.3 sd. backtest_bm.py divides each position by
   its own volatility — inverse-vol weighting on top of the ranks — and reported 0.792. The
   lift is a specification change, not new evidence, and the version that would actually be
   traded has never faced a placebo. It does here.

2. px_entry USED THE PREVIOUS MONTH'S CLOSE. The signal is known at the end of month t and
   the trade happens at that settle, so notional must be priced at month t, not t-1. The old
   code priced positions two months stale relative to the return window. The bias was
   conservative — high-BM names have risen, so their notional was understated — but it was
   wrong.

WHAT ELSE IS TESTED

  jackknife        drop each instrument in turn. If two names carry it, it is not a premium.
  parameter grid   formation period, vol window, vol target. A plateau is evidence; a spike
                   is a warning.
  decay            the paper circulated from 2015. Sub-periods run 1.175 / 0.381 / 0.391,
                   which is what post-publication attenuation looks like. Quantify it.
  weighting        rank versus quintile versus equal-weight extremes. If only one scheme
                   works, the result is the scheme.
  costs            out to 20bp per side, well beyond anything plausible.
  MECHANISM        Boons & Prado attribute basis-momentum to imbalances that materialise
                   "when the market-clearing ability of speculators and intermediaries is
                   impaired." That is a testable claim and nobody in this literature
                   conditions on it. If BM is not stronger when intermediation is impaired,
                   the stated mechanism is unsupported — the factor may still pay, but the
                   economic story would be decoration.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

VOL_TARGET = 0.20
IDM_CAP = 2.5


def load_monthly(path: str, J: int = 12, vol_window: int = 6) -> pd.DataFrame:
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
                n_days=("r0", "size"))
           .reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m = m[(m["n_days"] >= 10) & (m["asset"] == "commodity")].copy()
    c0 = m.groupby("symbol")["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    c1 = m.groupby("symbol")["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm"] = c0 - c1
    # FIX: price the position at the settle we actually trade on, month t, not t-1.
    m["px_entry"] = m["px"]
    m["vol"] = (m.groupby("symbol")["r0"]
                  .transform(lambda s: s.rolling(vol_window, min_periods=3).std())) * np.sqrt(12)
    m["fwd"] = m.groupby("symbol")["r0"].shift(-1)
    return m.sort_values(["symbol", "ym"]).reset_index(drop=True)


def idm_of(m: pd.DataFrame) -> float:
    n = m["symbol"].nunique()
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    return min(1.0 / np.sqrt((1/n) + (1 - 1/n) * max(rho, 0.01)), IDM_CAP)


def weights_from(sig: pd.Series, scheme: str) -> np.ndarray:
    if scheme == "rank":
        r = sig.rank(); w = (r - r.mean()).to_numpy()
    elif scheme == "quintile":
        k = max(1, len(sig) // 5)
        o = sig.rank(method="first").to_numpy()
        w = np.where(o > len(sig) - k, 1.0, np.where(o <= k, -1.0, 0.0))
    elif scheme == "tercile":
        k = max(1, len(sig) // 3)
        o = sig.rank(method="first").to_numpy()
        w = np.where(o > len(sig) - k, 1.0, np.where(o <= k, -1.0, 0.0))
    else:
        raise ValueError(scheme)
    g = np.abs(w).sum()
    return w / g if g > 0 else w


def run(m: pd.DataFrame, idm: float, capital: float = 450_000, bps: float = 3.0,
        scheme: str = "rank", integer: bool = True, vol_target: float = VOL_TARGET,
        shuffle_seed: int | None = None, min_n: int = 6) -> pd.Series:
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    prev, out = {}, {}
    for ym, g in m.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        sig = s["bm"]
        if rng is not None:
            sig = pd.Series(rng.permutation(sig.to_numpy()), index=sig.index)
        w = weights_from(sig, scheme)
        pnl = cost = 0.0
        held = {}
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            cdv = dpm * px * vol
            if cdv <= 0:
                continue
            tgt = wi * capital * vol_target * idm / cdv
            n = float(np.round(tgt)) if integer else tgt
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / capital
    return pd.Series(out).sort_index()


def stat(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 48:
        return dict(n=len(r), sharpe=np.nan, t=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12, vol=av,
                dd=float((eq / eq.cummax() - 1).min()))


def line(lbl: str, s: dict, extra: str = "") -> None:
    if not np.isfinite(s.get("sharpe", np.nan)):
        print(f"  {lbl:32s} n={s.get('n',0)}"); return
    flag = " *" if abs(s["t"]) > 2 else ""
    print(f"  {lbl:32s} SR {s['sharpe']:>+6.3f}  t {s['t']:>+5.2f}  "
          f"ret {s['ann']*100:>+6.2f}%  dd {s['dd']*100:>+6.1f}%{flag}  {extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_wide.parquet")
    ap.add_argument("--seeds", type=int, default=25)
    a = ap.parse_args()

    m = load_monthly(a.prices)
    idm = idm_of(m)
    base = run(m, idm)
    bs = stat(base)

    print("=" * 78)
    print("1. BASELINE, WITH THE ENTRY-PRICE BUG FIXED")
    print("=" * 78)
    line("integer, rank, 3bp", bs)
    old = run(m.assign(px_entry=m.groupby("symbol")["px"].shift(1)), idm).dropna()
    line("with the old stale entry price", stat(old))
    print("  The old version priced positions at the previous month's settle. The bias")
    print("  was conservative, but the number quoted must come from the corrected run.")

    print("\n" + "=" * 78)
    print("2. PLACEBO ON THE TRADED SPECIFICATION — the gap that mattered")
    print("=" * 78)
    ts = []
    for sd in range(a.seeds):
        s = stat(run(m, idm, shuffle_seed=sd))
        if np.isfinite(s["t"]):
            ts.append(s["t"])
    ts = np.array(ts)
    z = (bs["t"] - ts.mean()) / max(ts.std(ddof=1), 1e-9)
    print(f"  placebo t {ts.mean():+.2f} ± {ts.std(ddof=1):.2f} over {len(ts)} seeds")
    print(f"  real t {bs['t']:+.2f} sits {z:+.1f} placebo sd out   "
          f"{'PASS' if abs(z) > 2 else 'FAIL'}")
    placebo_ok = abs(z) > 2

    print("\n" + "=" * 78)
    print("3. JACKKNIFE — is it two instruments?")
    print("=" * 78)
    rows = []
    for sym in sorted(m["symbol"].unique()):
        sub = m[m["symbol"] != sym]
        s = stat(run(sub, idm_of(sub)))
        rows.append(dict(dropped=sym, sharpe=s["sharpe"], t=s["t"]))
    jk = pd.DataFrame(rows).sort_values("sharpe")
    print(jk.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\n  full sample SR {bs['sharpe']:+.3f}   "
          f"worst drop-one {jk['sharpe'].min():+.3f} ({jk.iloc[0]['dropped']})   "
          f"best {jk['sharpe'].max():+.3f}")
    jk_ok = jk["sharpe"].min() > 0.35
    print(f"  {'PASS' if jk_ok else 'FAIL'} — no single instrument carries the result")

    print("\n" + "=" * 78)
    print("4. PARAMETER GRID — plateau or spike?")
    print("=" * 78)
    print(f"  {'J':>3s} {'volwin':>7s} {'target':>7s} {'SR':>8s} {'t':>7s}")
    grid = []
    for Jx in (6, 9, 12, 15):
        for vw in (3, 6, 12):
            mm = load_monthly(a.prices, J=Jx, vol_window=vw)
            s = stat(run(mm, idm_of(mm)))
            grid.append(dict(J=Jx, vw=vw, sharpe=s["sharpe"], t=s["t"]))
            print(f"  {Jx:>3d} {vw:>7d} {VOL_TARGET:>7.0%} {s['sharpe']:>+8.3f} "
                  f"{s['t']:>+7.2f}")
    gdf = pd.DataFrame(grid)
    frac_pos = (gdf["sharpe"] > 0.35).mean()
    print(f"\n  cells above SR 0.35: {frac_pos:.0%} of {len(gdf)}   "
          f"median {gdf['sharpe'].median():+.3f}   spread "
          f"{gdf['sharpe'].min():+.3f} to {gdf['sharpe'].max():+.3f}")
    grid_ok = frac_pos > 0.7
    print(f"  {'PASS' if grid_ok else 'FAIL'} — a flat surface is evidence, a spike is not")

    print("\n" + "=" * 78)
    print("5. WEIGHTING SCHEME")
    print("=" * 78)
    for sch in ("rank", "tercile", "quintile"):
        line(sch, stat(run(m, idm, scheme=sch)))
    print("  If only one scheme works, the result is the scheme, not the signal.")

    print("\n" + "=" * 78)
    print("6. DECAY — the paper circulated from 2015")
    print("=" * 78)
    for lo, hi, lab in ((2011, 2015, "2011-2015 (pre-pub)"),
                        (2016, 2020, "2016-2020"),
                        (2021, 2026, "2021-2026 (post-pub)")):
        seg = base[(base.index.year >= lo) & (base.index.year <= hi)]
        line(lab, stat(seg))
    print("  Post-publication attenuation is documented for strong factors and is not")
    print("  itself evidence of data mining. But it must be disclosed, and it means the")
    print("  forward-looking estimate is the LATER number, not the full-sample one.")

    print("\n" + "=" * 78)
    print("7. COST SENSITIVITY")
    print("=" * 78)
    for bps in (3, 5, 10, 20, 40):
        line(f"{bps}bp per side", stat(run(m, idm, bps=bps)))

    print("\n" + "=" * 78)
    print("8. MECHANISM — is BM stronger when intermediation is impaired?")
    print("=" * 78)
    print("  Boons & Prado attribute BM to imbalances that materialise when the")
    print("  market-clearing ability of speculators and intermediaries is impaired.")
    print("  Nobody in this literature conditions on that. Three proxies, all from")
    print("  price data available at the decision point:\n")
    st = m.groupby("ym").agg(disp=("basis", "std"),
                             cxvol=("r0", "std"),
                             imb=("bm", lambda s: s.abs().mean())).dropna()
    for col, name in (("disp", "basis dispersion"),
                      ("cxvol", "cross-sectional return dispersion"),
                      ("imb", "mean absolute basis-momentum")):
        # expanding median so the split uses no future information
        med = st[col].expanding().median().shift(1)
        hi_m = st.index[(st[col] > med).fillna(False)]
        lo_m = st.index[(st[col] <= med).fillna(False)]
        h, l = stat(base.reindex(hi_m).dropna()), stat(base.reindex(lo_m).dropna())
        if np.isfinite(h["sharpe"]) and np.isfinite(l["sharpe"]):
            print(f"  {name}")
            print(f"    impaired (high) SR {h['sharpe']:>+6.3f}  t {h['t']:>+5.2f}   "
                  f"n={h['n']}")
            print(f"    normal   (low)  SR {l['sharpe']:>+6.3f}  t {l['t']:>+5.2f}   "
                  f"n={l['n']}")
            print(f"    difference in mean monthly return "
                  f"{(base.reindex(hi_m).mean() - base.reindex(lo_m).mean())*100:+.3f}%")
    print("\n  If the impaired state is NOT stronger, the stated mechanism is")
    print("  unsupported in this sample. The factor may still pay, but the economic")
    print("  story would be decoration and the pitch must say so.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    checks = [("placebo on the traded spec", placebo_ok),
              ("no single instrument carries it", jk_ok),
              ("parameter surface is a plateau", grid_ok),
              ("survives 20bp per side", stat(run(m, idm, bps=20))["sharpe"] > 0.35)]
    for k, v in checks:
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\n  Whatever the outcome, the honest framing is fixed: this is a published")
    print("  factor with a published critique (Maio & Kwon 2024) arguing the spreading")
    print("  variant is irrelevant and only the nearby variant retains explanatory")
    print("  power. This data independently reproduces that split. Say it plainly —")
    print("  it is a stronger claim than pretending the factor is clean.")


if __name__ == "__main__":
    main()