"""
reconcile.py — which file produced the headline, and what is the number now?

    python reconcile.py --before px_wide.parquet --after px_clean.parquet

WHY THIS EXISTS

sizing.py reported the fixed-IDM baseline at Sharpe 0.586. That is the frozen
specification — same signal, same formation window, same inverse-volatility scaling, same
integer contracts, same 3bp per side, same multiplier of 2.5 — and it should have returned
0.760. It did not.

The likely cause is the data repair. repair.py resolved 6,007 duplicate roll-date rows by
continuity rather than by open interest, and removed rows carrying non-positive settlements.
If the old rule was systematically selecting stale expiring contracts, those rows were
producing returns that never happened, and the earlier number was inflated by them.

This runs ONE implementation of the frozen specification against BOTH files and reports the
difference. There is no ambiguity afterwards: either the repair moved the headline or it did
not, and if it did, the pitch has to be rewritten before it goes anywhere.

The strategy code below is a single function used for both files, so any difference is the
data and nothing else.
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


def frozen(path: str, bps: float = 3.0):
    """The frozen specification, verbatim. One implementation, used on both files."""
    df = pd.read_parquet(path)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    n_raw = len(df)
    df = df[df["contract_0"] != df["contract_1"]]
    n_dup = df.duplicated(["date", "symbol"]).sum()
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
    df["ym"] = df["date"].dt.to_period("M")
    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                px=("settle_0", "last"), nd=("r0", "size")).reset_index())
    m = m[m["nd"] >= 10].sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["bm"] = (g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
               - g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum()))
    m["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)

    prev_pos, out = {}, {}
    for ym, gg in m.groupby("ym"):
        s = gg[["symbol", "bm", "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < 6:
            continue
        r = s["bm"].rank()
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
            n = float(np.round(wi * CAPITAL * VOL_TARGET * IDM / den))
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            tr = abs(n - prev_pos.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        for sym in set(prev_pos) - set(held):
            cost += abs(prev_pos[sym]) * BY_SYMBOL[sym].commission
        prev_pos = held
        out[ym] = (pnl - cost) / CAPITAL
    ser = pd.Series(out).sort_index()
    yrs = len(ser) / 12
    av = ser.std(ddof=1) * np.sqrt(12)
    sr = (ser.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + ser).cumprod()
    return dict(series=ser, sharpe=sr, t=sr * np.sqrt(yrs), ann=ser.mean() * 12,
                vol=av, dd=float((eq / eq.cummax() - 1).min()), months=len(ser),
                rows=n_raw, dups=n_dup, inst=m["symbol"].nunique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="px_wide.parquet")
    ap.add_argument("--after", default="px_clean.parquet")
    a = ap.parse_args()

    print("=" * 78)
    print("ONE IMPLEMENTATION, TWO FILES")
    print("=" * 78)
    res = {}
    for lab, path in (("before repair", a.before), ("after repair", a.after)):
        try:
            res[lab] = frozen(path)
        except FileNotFoundError:
            print(f"  {lab}: {path} not found")
    if len(res) < 2:
        raise SystemExit("\n  Need both files to reconcile.")

    print(f"  {'':14s} {'Sharpe':>8s} {'t':>7s} {'return':>9s} {'vol':>7s} "
          f"{'maxDD':>8s} {'months':>7s} {'rows':>9s} {'dups':>7s}")
    for lab, r in res.items():
        print(f"  {lab:14s} {r['sharpe']:>+8.3f} {r['t']:>+7.2f} "
              f"{r['ann']*100:>+8.2f}% {r['vol']*100:>6.1f}% {r['dd']*100:>+7.1f}% "
              f"{r['months']:>7d} {r['rows']:>9,} {r['dups']:>7,}")

    b, af = res["before repair"], res["after repair"]
    d = af["sharpe"] - b["sharpe"]
    print(f"\n  change in Sharpe from the repair: {d:+.3f}")

    j = pd.concat([b["series"].rename("before"), af["series"].rename("after")],
                  axis=1).dropna()
    if len(j) > 24:
        print(f"  correlation of the two monthly return series: {j['before'].corr(j['after']):+.4f}")
        diff = (j["before"] - j["after"])
        worst = diff.abs().nlargest(6)
        print(f"  months where they differ most:")
        for ym in worst.index:
            print(f"    {ym}   before {j.at[ym,'before']*100:>+7.2f}%   "
                  f"after {j.at[ym,'after']*100:>+7.2f}%   "
                  f"difference {diff[ym]*100:>+7.2f}%")
        share = (diff.abs() > 0.005).mean()
        print(f"  share of months differing by more than 0.5%: {share:.1%}")

    print("\n" + "=" * 78)
    print("WHAT THIS MEANS")
    print("=" * 78)
    if abs(d) < 0.03:
        print("  The repair did not move the headline. The 0.586 reported by sizing.py")
        print("  came from somewhere else in that script and needs a separate look.")
    elif d < 0:
        print(f"  THE REPAIR LOWERED THE HEADLINE by {abs(d):.3f}.")
        print()
        print("  The earlier number was inflated by the duplicate-resolution rule. On roll")
        print("  dates the old rule kept whichever record carried the higher open interest,")
        print("  which could be a stale expiring contract with a distorted settlement — and")
        print("  a distorted settle produces a return that never happened.")
        print()
        print(f"  THE CORRECT HEADLINE IS {af['sharpe']:.3f}, NOT {b['sharpe']:.3f}.")
        print()
        print("  Every figure in the pitch must be regenerated from the repaired file. The")
        print("  strategy is not broken — it is smaller than the earlier run suggested, and")
        print("  the earlier run was reading bad rows. Finding this before submission is")
        print("  the whole reason the verification suite exists.")
    else:
        print(f"  The repair RAISED the headline by {d:.3f}. Still regenerate everything")
        print("  from the repaired file: the number that ships must come from the data")
        print("  that is correct, not the data that flattered it.")

    print("\n  Regenerate, in this order:")
    print("    python verify.py         --prices px_clean.parquet")
    print("    python channels.py       --prices px_clean.parquet")
    print("    python make_exhibits.py  --prices px_clean.parquet")
    print("  then update the pitch from those outputs and nothing else.")


if __name__ == "__main__":
    main()