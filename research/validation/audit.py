"""
audit.py — an attempt to kill the strategy.

    python audit.py --prices px_clean.parquet

This script is written to FAIL the strategy, not to confirm it. Every check below is an
attack. A clean result only means the attack missed, and each one states what a failure
would look like so a pass cannot be read as vague reassurance.

  1  LISTING DATES. Micro contracts did not exist for most of this sample. Micro WTI
     listed in 2021, Micro Copper later still. If those symbols show data back to 2010,
     either the history is padded, or the vendor is returning full-size contract data
     under a micro symbol - in which case dollar P&L is wrong by a factor of ten for
     those names and the headline is meaningless.

  2  MULTIPLIER SANITY. Independent of listing dates: does each contract's implied daily
     dollar move look like the contract the universe file claims it is?

  3  IS 12 MONTHS THE BEST CELL? The parameter grid is reported as a plateau and a round
     value is claimed to have been chosen in advance. If twelve months with a six-month
     volatility window happens to be the single best cell in the grid, that claim is not
     credible however it was actually arrived at.

  4  IS THE RETURN JUST A PERSISTENT SECTOR TILT? The book runs a standing long in
     livestock and a standing short in energy. Over a period when energy fell hard, a
     static short would have made money on its own. If the strategy is that tilt wearing
     a signal, it is not a strategy.

  5  IS IT REALLY CONVEX TO STRESS? The pitch claims convexity to commodity stress on the
     strength of 2014-16. But 2020 was also stress and the strategy lost 17%. Slow
     declines and violent shocks are different things and the claim may be conflating
     them.

  6  DOES ANY SINGLE MONTH CARRY IT? Beyond the concentration figure already reported -
     what happens if the best few months are simply deleted.

  7  SUB-SAMPLE HONESTY. What the strategy looks like if only the most recent half of the
     data existed, which is the sample a sceptical reader will mentally construct.
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
COST_MULTIPLE = 3.0

# Publicly known first listing dates for the micro / E-mini contracts in this universe.
# Anything with history materially before these dates is a data problem, not a windfall.
KNOWN_LISTING = {
    "MCL": "2021-07",   # Micro WTI Crude
    "MHG": "2022-05",   # Micro Copper
    "MGC": "2010-10",   # Micro Gold
    "SIL": "2013-03",   # Micro Silver
    "M2K": "2019-05", "MES": "2019-05", "MNQ": "2019-05", "MYM": "2019-05",
    "QG":  "2000-01",   # E-mini Natural Gas, long-standing
}


def load(path: str):
    df = pd.read_parquet(path)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df = df[df["contract_0"] != df["contract_1"]]
    df = (df.sort_values(["symbol", "date", "oi_0"], na_position="first")
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"]).reset_index(drop=True))
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
    return df


def commodity_panel(df: pd.DataFrame):
    d = df[df["asset"] == "commodity"].copy()
    med = d.groupby("symbol")["settle_0"].median()
    cost = {}
    for s in med.index:
        i = BY_SYMBOL[s]
        n = med[s] * i.dollar_price_mult
        cost[s] = 1.5 * (i.tick_value / n * 1e4) + i.commission / n * 1e4
    cs = pd.Series(cost)
    drop = set(cs[cs > COST_MULTIPLE * cs.median()].index)
    d = d[~d["symbol"].isin(drop)].copy()
    m = (d.groupby(["symbol", "ym"])
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
    m["sector"] = m["symbol"].map(lambda s: BY_SYMBOL[s].sector)
    return m, drop


def book(m, J_=J, vw=VOL_WINDOW, sector_neutral=False, static=False, min_n=6):
    """`static` replaces the signal with a constant per instrument: the tilt, no signal."""
    d = m.copy()
    if J_ != J or vw != VOL_WINDOW:
        g = d.groupby("symbol")
        d["bm"] = (g["r0"].transform(lambda s: s.rolling(J_, min_periods=J_).sum())
                   - g["r1"].transform(lambda s: s.rolling(J_, min_periods=J_).sum()))
        d["vol"] = (g["r0"].transform(
            lambda s: s.rolling(vw, min_periods=3).std()) * np.sqrt(12)
            ).groupby(d["symbol"]).shift(1)
    if static:
        # each instrument's average rank over the whole sample, held fixed. This uses
        # future information deliberately - it is the strongest possible version of the
        # "it is only a standing tilt" attack, so beating it means something.
        avg = d.groupby("ym")["bm"].rank().groupby(d["symbol"]).transform("mean")
        d["bm"] = d["symbol"].map(d.assign(a=avg).groupby("symbol")["a"].mean())
    prev, out = {}, {}
    for ym, g in d.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd", "sector"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s["bm"].rank()
        w = (r - r.mean()).to_numpy().astype(float).copy()
        if sector_neutral:
            sec = s["sector"].to_numpy()
            for x in set(sec):
                msk = sec == x
                if msk.sum() > 1:
                    w[msk] -= w[msk].mean()
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
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * 3.0 / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
    return pd.Series(out).sort_index()


def sharpe(r):
    r = r.dropna()
    if len(r) < 24:
        return np.nan
    av = r.std(ddof=1) * np.sqrt(12)
    return (r.mean() * 12) / av if av > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_clean.parquet")
    a = ap.parse_args()
    df = load(a.prices)
    m, dropped = commodity_panel(df)
    base = book(m)
    print(f"  baseline (single grid, monthly): Sharpe {sharpe(base):.3f}, "
          f"{len(base)} months; universe rule excluded {sorted(dropped)}")

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 80)
    print("ATTACK 1 — DOES THE DATA PREDATE THE CONTRACT?")
    print("=" * 80)
    print("  A contract cannot have prices before it was listed. If it appears to, the")
    print("  vendor is serving a different contract under that symbol and the dollar")
    print("  multiplier is wrong.\n")
    first = df.groupby("symbol")["date"].min()
    bad = []
    print(f"  {'sym':5s} {'first obs':>11s} {'known listing':>14s} {'verdict':>28s}")
    for s in sorted(first.index):
        known = KNOWN_LISTING.get(s)
        if not known:
            continue
        f = first[s]
        kd = pd.Timestamp(known + "-01")
        gap = (kd - f).days / 30.44
        ok = gap < 3
        if not ok:
            bad.append((s, f, known, gap))
        print(f"  {s:5s} {f:%Y-%m-%d} {known:>14s} "
              f"{('ok' if ok else f'{gap:.0f} MONTHS TOO EARLY'):>28s}")
    if bad:
        print(f"\n  *** {len(bad)} CONTRACT(S) SHOW HISTORY BEFORE THEY EXISTED ***")
        for s, f, k, gap in bad:
            inst = BY_SYMBOL[s]
            print(f"    {s}: data from {f:%Y-%m}, listed {k}, multiplier "
                  f"{inst.multiplier} in universe.py")
        print("  This must be resolved before submission. Either the vendor is mapping")
        print("  the full-size contract to this symbol - in which case the multiplier is")
        print("  wrong by the micro ratio and every dollar figure for these names is")
        print("  wrong - or the history is padded. Check one date by hand against the CME")
        print("  settlement record.")
    else:
        print("\n  No contract shows history before its listing date.")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 80)
    print("ATTACK 2 — DOES THE MULTIPLIER MATCH THE PRICE SERIES?")
    print("=" * 80)
    print("  Implied notional and one day of dollar volatility, per contract. A micro")
    print("  should be roughly a tenth of its full-size sibling; if it looks full-size,")
    print("  the multiplier and the data disagree.\n")
    d = df[df["asset"] == "commodity"]
    print(f"  {'sym':5s} {'med price':>11s} {'mult':>8s} {'notional':>11s} "
          f"{'1-day $ vol':>12s}")
    for s in sorted(d["symbol"].unique()):
        inst = BY_SYMBOL[s]
        px = d[d["symbol"] == s]["settle_0"].median()
        vol = d[d["symbol"] == s]["r0"].std()
        notional = px * inst.dollar_price_mult
        print(f"  {s:5s} {px:>11.3f} {inst.multiplier:>8} ${notional:>10,.0f} "
              f"${notional*vol:>11,.0f}")
    print("\n  A one-day dollar move far outside roughly $100 to $3,000 on a $450,000")
    print("  book is worth a second look.")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 80)
    print("ATTACK 3 — IS THE CHOSEN CELL THE BEST CELL?")
    print("=" * 80)
    grid = {}
    for jj in (6, 9, 12, 15, 18):
        for vv in (3, 6, 12):
            grid[(jj, vv)] = sharpe(book(m, J_=jj, vw=vv))
    print(f"  {'formation':>10s} " + "".join(f"{'vol ' + str(v):>10s}"
                                             for v in (3, 6, 12)))
    for jj in (6, 9, 12, 15, 18):
        row = "".join(f"{grid[(jj, v)]:>10.3f}" for v in (3, 6, 12))
        mark = "   <-- chosen" if jj == 12 else ""
        print(f"  {str(jj) + 'm':>10s} {row}{mark}")
    best = max(grid, key=lambda k: grid[k])
    chosen = grid[(12, 6)]
    rank = sorted(grid.values(), reverse=True).index(chosen) + 1
    print(f"\n  chosen cell (12m, 6m): {chosen:.3f}, ranked {rank} of {len(grid)}")
    print(f"  best cell {best}: {grid[best]:.3f}")
    if rank == 1:
        print("  *** THE CHOSEN CELL IS THE BEST CELL. The claim that a round value was")
        print("  picked in advance is not credible to a reader, whatever the history.")
        print("  Either move to a cell that is not the maximum, or drop the claim. ***")
    else:
        print(f"  The chosen cell is NOT the maximum, which supports the claim that it was")
        print(f"  not selected on performance. Quote the rank.")

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 80)
    print("ATTACK 4 — IS IT JUST A STANDING SECTOR TILT?")
    print("=" * 80)
    stat = book(m, static=True)
    sn = book(m, sector_neutral=True)
    print(f"  {'real signal':32s} {sharpe(base):>8.3f}")
    print(f"  {'static tilt (uses future info)':32s} {sharpe(stat):>8.3f}")
    print(f"  {'sector-neutral (tilts removed)':32s} {sharpe(sn):>8.3f}")
    print("\n  The static book freezes each instrument's average rank over the whole")
    print("  sample and trades that constant - it is the tilt with the signal removed,")
    print("  and it is allowed to use future information. If it approaches the real")
    print("  strategy, the return is a standing bet on which commodities did well.")
    if np.isfinite(sharpe(stat)) and sharpe(stat) > sharpe(base) * 0.6:
        print("  *** THE STATIC TILT CAPTURES MOST OF THE RETURN. Serious problem. ***")
    else:
        print("  The static tilt does not reproduce the strategy, so the return comes")
        print("  from the signal changing its mind rather than from a fixed position.")

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 80)
    print("ATTACK 5 — IS 'CONVEX TO STRESS' ACTUALLY TRUE?")
    print("=" * 80)
    mkt = m.groupby("ym")["r0"].mean().dropna()
    j = pd.concat([base.rename("s"), mkt.rename("m")], axis=1).dropna()
    print("  Strategy return, by what the market did that month:\n")
    print(f"  {'market decile':>16s} {'mean strat':>12s} {'hit rate':>10s} {'n':>5s}")
    j["q"] = pd.qcut(j["m"], 5, labels=False)
    for q in range(5):
        seg = j[j["q"] == q]
        lab = ["worst 20%", "2nd", "middle", "4th", "best 20%"][q]
        print(f"  {lab:>16s} {seg['s'].mean()*100:>11.2f}% "
              f"{(seg['s'] > 0).mean():>9.0%} {len(seg):>5d}")
    down = j[j["m"] < 0]
    print(f"\n  all down months: mean {down['s'].mean()*100:+.2f}%, "
          f"hit rate {(down['s'] > 0).mean():.0%}, n={len(down)}")
    slow = j[(j["m"] < 0) & (j["m"] > j["m"].quantile(0.05))]
    crash = j[j["m"] <= j["m"].quantile(0.05)]
    print(f"  ordinary down months:  mean {slow['s'].mean()*100:+.2f}%  n={len(slow)}")
    print(f"  worst 5% of months:    mean {crash['s'].mean()*100:+.2f}%  n={len(crash)}")
    print("\n  If the strategy earns in ordinary declines but not in the sharpest ones,")
    print("  'convex to stress' is the wrong phrase. The defensible claim would be that")
    print("  it does well in sustained declines and badly in violent reversals, which is")
    print("  a different and more specific statement.")

    # ---------------------------------------------------------------- 6
    print("\n" + "=" * 80)
    print("ATTACK 6 — DELETE THE BEST MONTHS")
    print("=" * 80)
    print(f"  {'removed':>12s} {'Sharpe':>8s} {'return':>9s}")
    for k in (0, 1, 3, 6, 12):
        r = base.drop(base.nlargest(k).index) if k else base
        print(f"  {'best ' + str(k):>12s} {sharpe(r):>8.3f} {r.mean()*12*100:>8.2f}%")
    print("\n  Every strategy dies if enough good months are removed. What matters is")
    print("  whether it survives losing three, which is a plausible run of bad luck.")

    # ---------------------------------------------------------------- 7
    print("\n" + "=" * 80)
    print("ATTACK 7 — IF ONLY THE RECENT HALF EXISTED")
    print("=" * 80)
    mid = base.index[len(base) // 2]
    for lab, seg in (("full sample", base),
                     ("first half", base[base.index <= mid]),
                     ("second half", base[base.index > mid]),
                     ("last 5 years", base[base.index >= "2021-09"]),
                     ("last 3 years", base[base.index >= "2023-09"])):
        s = sharpe(seg)
        t = s * np.sqrt(len(seg) / 12) if np.isfinite(s) else np.nan
        print(f"  {lab:16s} Sharpe {s:>7.3f}   t {t:>+6.2f}   n {len(seg):>4d}")
    print("\n  A reader who trusts only recent data will compute the bottom rows. If they")
    print("  are materially weaker than the headline, the headline needs the qualifier")
    print("  attached to it rather than left for them to discover.")


if __name__ == "__main__":
    main()