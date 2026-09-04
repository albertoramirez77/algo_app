"""
micros.py — could this book actually have been traded in 2012?

    python micros.py --prices data/px_clean.parquet

THE PROBLEM

Four contracts in the universe are micros that did not exist for most of the sample.

    MCL  Micro WTI Crude      listed 2021-07
    MHG  Micro Copper         listed 2022-05
    SIL  Micro Silver         listed 2013-03
    MGC  Micro Gold           listed 2010-10

Prices are unaffected - a micro and its full-size sibling track the same underlying and
settle at the same price - so returns, correlations and Sharpe ratios are all correct.
What is NOT correct is implementability. Before Micro WTI listed, taking that exposure
required full-size CL at ten times the notional. The backtest assumes a $7,128 building
block was available in 2012 when the smallest tradeable unit was $71,280.

WHY THAT MATTERS HERE SPECIFICALLY

This strategy identifies integer contract granularity as its binding constraint - 17.2% of
intended positions already round to zero at $450,000. Assuming micros existed throughout
understates exactly the constraint the pitch says matters most. That is an optimistic
assumption in the one place it does the most damage.

THREE VERSIONS

    optimistic   micro multipliers throughout, which is what the backtest currently does
    realistic    full-size multiplier before each contract's listing date, micro after
    conservative full-size multipliers throughout, ignoring that micros ever listed

The realistic version is the truth. The conservative version is the bound: if the strategy
survives when the smallest tradeable unit is ten times larger for the entire sample, the
listing dates cannot be the thing holding it up.
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

# ratio of full-size contract to the micro, and the micro's first listing month
FULL_SIZE = {
    "MCL": (10.0, "2021-07"),   # CL is 1,000 bbl against MCL's 100
    "MHG": (10.0, "2022-05"),   # HG is 25,000 lb against MHG's 2,500
    "SIL": (5.0,  "2013-03"),   # SI is 5,000 oz against SIL's 1,000
    "MGC": (10.0, "2010-10"),   # GC is 100 oz against MGC's 10
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
    df = df[df["asset"] == "commodity"].copy()
    med = df.groupby("symbol")["settle_0"].median()
    cost = {}
    for s in med.index:
        i = BY_SYMBOL[s]
        n = med[s] * i.dollar_price_mult
        cost[s] = 1.5 * (i.tick_value / n * 1e4) + i.commission / n * 1e4
    cs = pd.Series(cost)
    drop = set(cs[cs > COST_MULTIPLE * cs.median()].index)
    df = df[~df["symbol"].isin(drop)].copy()
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
    return m, drop


def scale_for(sym: str, ym, mode: str) -> float:
    """Multiplier inflation factor for this contract in this month."""
    if sym not in FULL_SIZE or mode == "optimistic":
        return 1.0
    ratio, listed = FULL_SIZE[sym]
    if mode == "conservative":
        return ratio
    return ratio if ym < pd.Period(listed, freq="M") else 1.0     # realistic


def book(m: pd.DataFrame, mode: str, bps: float = 3.0, min_n: int = 6):
    prev, out, zero, tot = {}, {}, 0, 0
    for ym, g in m.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
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
            k = scale_for(sym, ym, mode)
            dpm = inst.dollar_price_mult * k          # bigger contract, same price
            comm = inst.commission * (k if k > 1 else 1.0)
            den = dpm * px * vol
            if den <= 0:
                continue
            tgt = wi * CAPITAL * VOL_TARGET * IDM / den
            n = float(np.round(tgt))
            tot += 1
            if n == 0 and abs(tgt) > 1e-9:
                zero += 1
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (comm + abs(dpm) * px * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
    return pd.Series(out).sort_index(), zero / max(tot, 1)


def st(r):
    r = r.dropna()
    if len(r) < 24:
        return dict(n=len(r), sharpe=np.nan, t=np.nan, ann=np.nan, vol=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12, vol=av,
                dd=float((eq / eq.cummax() - 1).min()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    a = ap.parse_args()
    m, dropped = load(a.prices)
    present = [s for s in FULL_SIZE if s in set(m["symbol"])]

    print("=" * 80)
    print("1. WHICH CONTRACTS ARE AFFECTED")
    print("=" * 80)
    print(f"  universe rule already excluded {sorted(dropped)}\n")
    print(f"  {'sym':5s} {'listed':>9s} {'full/micro':>11s} "
          f"{'months before listing':>22s}")
    first = m["ym"].min()
    for s in present:
        ratio, listed = FULL_SIZE[s]
        lp = pd.Period(listed, freq="M")
        n_before = len(m[(m["symbol"] == s) & (m["ym"] < lp)]["ym"].unique())
        print(f"  {s:5s} {listed:>9s} {ratio:>10.0f}x {n_before:>22d}")
    print(f"\n  sample begins {first}, so a contract listing in 2021 or 2022 was")
    print(f"  unavailable for most of the backtest.")

    print("\n" + "=" * 80)
    print("2. THE THREE VERSIONS")
    print("=" * 80)
    print(f"  {'version':34s} {'Sharpe':>8s} {'return':>8s} {'vol':>7s} "
          f"{'maxDD':>8s} {'zeroed':>8s}")
    res = {}
    for mode, lab in (("optimistic", "micros throughout (as backtested)"),
                      ("realistic", "full-size until each listing date"),
                      ("conservative", "full-size for the entire sample")):
        r, z = book(m, mode)
        s = st(r)
        res[mode] = (s, z, r)
        print(f"  {lab:34s} {s['sharpe']:>8.3f} {s['ann']*100:>7.2f}% "
              f"{s['vol']*100:>6.1f}% {s['dd']*100:>7.1f}% {z*100:>7.1f}%")

    o, rl, cv = res["optimistic"][0], res["realistic"][0], res["conservative"][0]
    print(f"\n  realistic vs as-backtested:   Sharpe {rl['sharpe']-o['sharpe']:+.3f}, "
          f"zeroed {(res['realistic'][1]-res['optimistic'][1])*100:+.1f}pp")
    print(f"  conservative vs as-backtested: Sharpe {cv['sharpe']-o['sharpe']:+.3f}, "
          f"zeroed {(res['conservative'][1]-res['optimistic'][1])*100:+.1f}pp")

    print("\n" + "=" * 80)
    print("3. WHAT THIS MEANS")
    print("=" * 80)
    print("  Prices and returns are identical in all three versions - a micro and its")
    print("  full-size sibling track the same underlying. Only the size of the smallest")
    print("  tradeable unit changes, so any difference is purely the granularity")
    print("  constraint, which is the constraint the pitch already identifies as binding.\n")
    if np.isfinite(rl["sharpe"]) and abs(rl["sharpe"] - o["sharpe"]) < 0.05:
        print("  The realistic version is materially unchanged. The listing dates do not")
        print("  affect the result and one sentence of disclosure is enough: micros were")
        print("  unavailable for part of the sample, the strategy was re-run using")
        print("  full-size contracts for those periods, and the Sharpe ratio moved by")
        print(f"  {rl['sharpe']-o['sharpe']:+.3f}.")
    else:
        print("  The realistic version differs materially. Report the REALISTIC number as")
        print("  the headline - it is the one that could have been traded - and explain")
        print("  why it differs from the naive version.")
    print()
    if np.isfinite(cv["sharpe"]) and cv["sharpe"] > 0.5:
        print("  Even under the conservative bound, where the smallest tradeable unit is")
        print("  ten times larger for the ENTIRE sample and micros are pretended never to")
        print("  have listed, the strategy survives. That is the strongest form of the")
        print("  answer and it is worth stating in the pitch, because a reader who spots")
        print("  the listing dates will assume the worst.")
    else:
        print("  Under the conservative bound the strategy does not survive, which means")
        print("  it depends on the existence of micro contracts. That is a legitimate")
        print("  dependency for a $450,000 account and should be stated plainly rather")
        print("  than left implicit.")


if __name__ == "__main__":
    main()