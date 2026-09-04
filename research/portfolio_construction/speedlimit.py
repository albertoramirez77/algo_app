"""
speedlimit.py — should any instrument be excluded because it is simply too expensive?

    python speedlimit.py --prices data/px_clean.parquet

THE PROBLEM THIS ADDRESSES

The bottom-up cost model found that micro natural gas costs 27.92 basis points per side -
more than ten times the median instrument - because one tick is enormous relative to a
$6,985 notional. It is also the second most heavily traded name in the book. A single
contract is therefore consuming roughly a quarter of all transaction costs while
contributing one seventeenth of the breadth.

THE RULE MUST BE EX ANTE, OR IT IS DATA MINING

Dropping an instrument because the backtest improves is curve fitting with extra steps. So
the exclusion rule here uses ONLY contract specifications - tick value, multiplier, price -
and never touches performance:

    cost per side (bp) = (0.5 x tick + 1.0 x tick + commission) / notional x 10,000

Every term is knowable before a single return is computed. An instrument either clears the
threshold or it does not, and its Sharpe contribution never enters the decision. This is
the "speed limit" test the fund's own strategy document describes: refuse to trade anything
whose costs consume too large a share of the expected edge.

WHAT IS REPORTED

The full curve across thresholds, not the single threshold that maximises Sharpe. Choosing
the best-performing cutoff after seeing the results would reintroduce exactly the bias the
ex-ante rule exists to avoid. A round number defensible in advance - 10bp, say - is worth
more than an optimised one.

There is a genuine trade-off and the curve should show it: excluding instruments lowers
costs but also destroys breadth, and seventeen names is not many to begin with.
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
N_GRIDS = 21
THRESHOLDS = [None, 20.0, 10.0, 6.0, 4.0, 3.0]


def load_daily(path: str) -> pd.DataFrame:
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
    df["ym"] = df["date"].dt.to_period("M")
    df["dom"] = df.groupby(["symbol", "ym"]).cumcount()
    return df


def ex_ante_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cost per side in basis points, from contract specifications ONLY.

    The median settle over the whole sample is used for notional rather than the last
    price, so the figure is a property of the contract rather than of one date. No return,
    Sharpe or turnover enters this calculation.
    """
    med = df.groupby("symbol")["settle_0"].median()
    rows = []
    for s in sorted(df["symbol"].unique()):
        inst = BY_SYMBOL[s]
        notional = med[s] * inst.dollar_price_mult
        tick_bp = inst.tick_value / notional * 1e4
        rows.append(dict(symbol=s, sector=inst.sector, notional=notional,
                         tick_bp=tick_bp,
                         cost_bp=0.5 * tick_bp + 1.0 * tick_bp
                         + inst.commission / notional * 1e4))
    return pd.DataFrame(rows).sort_values("cost_bp", ascending=False)


def grid_targets(df: pd.DataFrame, offset: int, keep: set, min_n: int = 6) -> pd.DataFrame:
    d = df[df["symbol"].isin(keep)].sort_values(["symbol", "date"]).copy()
    for leg in ("0", "1"):
        d[f"c{leg}"] = d.groupby("symbol")[f"r{leg}"].transform(
            lambda s: s.fillna(0.0).cumsum())
    snap = d[d["dom"] == offset][["symbol", "ym", "date", "c0", "c1", "settle_0"]].copy()
    if snap.empty:
        return pd.DataFrame()
    snap = snap.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = snap.groupby("symbol")
    snap["r0"] = g["c0"].diff(); snap["r1"] = g["c1"].diff()
    snap["bm"] = (g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
                  - g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum()))
    snap["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(snap["symbol"]).shift(1)
    snap["px_entry"] = g["settle_0"].shift(1)
    rows = []
    for dt, gg in snap.groupby("date"):
        s = gg[["symbol", "bm", "vol", "px_entry"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s["bm"].rank(); w = (r - r.mean()).to_numpy(); gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        for sym, wi, vol, px in zip(s["symbol"], w, s["vol"], s["px_entry"]):
            inst = BY_SYMBOL[sym]
            den = inst.dollar_price_mult * px * vol
            if den > 0:
                rows.append(dict(date=dt, symbol=sym,
                                 target=wi * CAPITAL * VOL_TARGET * IDM / den))
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, keep: set, cost_map: dict | None, flat=3.0):
    frames = [f for f in (grid_targets(df, o, keep) for o in range(N_GRIDS))
              if not f.empty]
    if not frames:
        return None
    d = df[df["symbol"].isin(keep)]
    dates = pd.DatetimeIndex(sorted(d["date"].unique()))
    syms = sorted(keep)
    ret = d.pivot_table(index="date", columns="symbol", values="r0").reindex(
        dates, columns=syms)
    px = d.pivot_table(index="date", columns="symbol", values="settle_0").reindex(
        dates, columns=syms).ffill()
    stacks = []
    for tf in frames:
        stacks.append((tf.pivot_table(index="date", columns="symbol", values="target")
                         .reindex(index=dates, columns=syms).ffill()).to_numpy())
    S = np.stack(stacks, axis=0)
    cnt = np.sum(~np.isnan(S), axis=0)
    T = np.divide(np.nansum(S, axis=0), np.maximum(cnt, 1),
                  out=np.zeros_like(cnt, dtype=float), where=cnt > 0)
    N = np.round(T)
    dpm = np.array([BY_SYMBOL[s].dollar_price_mult for s in syms])
    comm = np.array([BY_SYMBOL[s].commission for s in syms])
    bps = np.array([cost_map[s] if cost_map else flat for s in syms])
    P = np.nan_to_num(px.to_numpy(), nan=0.0)
    R = np.nan_to_num(ret.to_numpy(), nan=0.0)
    held = N[:-1]
    pnl = np.nansum(held * dpm * P[:-1] * np.expm1(R[1:]), axis=1)
    trades = np.abs(np.diff(N, axis=0))
    cost = np.nansum(trades * (comm + np.abs(dpm) * P[:-1] * bps / 1e4), axis=1)
    net = pd.Series((pnl - cost) / CAPITAL, index=dates[1:]).resample("ME").sum()
    net = net[net != 0]
    gross = pd.Series(pnl / CAPITAL, index=dates[1:]).resample("ME").sum().reindex(net.index)
    cser = pd.Series(cost / CAPITAL, index=dates[1:]).resample("ME").sum().reindex(net.index)
    yrs = len(net) / 12
    av = net.std(ddof=1) * np.sqrt(12)
    return dict(n_inst=len(syms), months=len(net),
                sharpe=(net.mean() * 12) / av if av > 0 else np.nan,
                ann=net.mean() * 12, vol=av,
                dd=float(((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min()),
                cost_yr=cser.sum() / yrs,
                cost_share=cser.sum() / gross.sum() if gross.sum() > 0 else np.nan,
                trades=float(trades.sum(axis=1).mean() * 21))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    a = ap.parse_args()

    df = load_daily(a.prices)
    C = ex_ante_costs(df)
    cost_map = dict(zip(C["symbol"], C["cost_bp"]))
    all_syms = set(C["symbol"])

    print("=" * 82)
    print("1. EX-ANTE COST BY INSTRUMENT — from contract specifications only")
    print("=" * 82)
    print("  Nothing below uses a return, a Sharpe or a turnover figure. Every number is")
    print("  knowable before the strategy is run.\n")
    print(f"  {'sym':5s} {'sector':10s} {'notional':>11s} {'1 tick bp':>10s} "
          f"{'cost/side bp':>13s}")
    for _, r in C.iterrows():
        flag = ""
        if r["cost_bp"] > 20:
            flag = "   <-- extreme"
        elif r["cost_bp"] > 10:
            flag = "   <-- expensive"
        print(f"  {r['symbol']:5s} {r['sector']:10s} ${r['notional']:>10,.0f} "
              f"{r['tick_bp']:>10.2f} {r['cost_bp']:>13.2f}{flag}")
    print(f"\n  median {C['cost_bp'].median():.2f} bp   "
          f"most expensive {C.iloc[0]['symbol']} at {C.iloc[0]['cost_bp']:.2f} bp "
          f"({C.iloc[0]['cost_bp']/C['cost_bp'].median():.0f}x the median)")

    print("\n" + "=" * 82)
    print("2. THE SPEED-LIMIT CURVE")
    print("=" * 82)
    print("  Each row excludes every instrument above the threshold and reruns the whole")
    print("  strategy. Costs are charged at the bottom-up per-instrument rate, not a flat")
    print("  assumption, so the comparison is honest in both directions.\n")
    print(f"  {'threshold':>11s} {'kept':>5s} {'excluded':>28s} {'Sharpe':>8s} "
          f"{'return':>8s} {'cost/yr':>8s} {'cost/gross':>11s}")
    results = {}
    for th in THRESHOLDS:
        keep = all_syms if th is None else set(C[C["cost_bp"] <= th]["symbol"])
        if len(keep) < 8:
            print(f"  {'<= ' + str(th) + 'bp':>11s} {len(keep):>5d}   "
                  f"fewer than 8 names left; not run")
            continue
        r = run(df, keep, cost_map)
        if r is None:
            continue
        results[th] = r
        gone = sorted(all_syms - keep)
        lab = "none" if not gone else ", ".join(gone)
        if len(lab) > 27:
            lab = lab[:24] + "..."
        thl = "all" if th is None else f"<= {th:.0f}bp"
        print(f"  {thl:>11s} {r['n_inst']:>5d} {lab:>28s} {r['sharpe']:>8.3f} "
              f"{r['ann']*100:>7.2f}% {r['cost_yr']*100:>7.2f}% "
              f"{r['cost_share']:>10.1%}")

    print("\n" + "=" * 82)
    print("3. WHAT THE CURVE SAYS")
    print("=" * 82)
    base = results.get(None)
    if base:
        best_th = max((t for t in results if t is not None),
                      key=lambda t: results[t]["sharpe"], default=None)
        for th in sorted((t for t in results if t is not None), reverse=True):
            r = results[th]
            d_sr = r["sharpe"] - base["sharpe"]
            d_cost = (r["cost_yr"] - base["cost_yr"]) * 100
            print(f"  <= {th:>4.0f}bp   Sharpe {d_sr:>+6.3f}   cost {d_cost:>+6.2f}%/yr   "
                  f"names {r['n_inst']:>2d} (from {base['n_inst']})")
        print()
        print("  Two forces oppose each other here. Excluding an expensive instrument")
        print("  lowers cost, which helps. It also removes a name from a cross-section of")
        print("  seventeen, which reduces breadth and hurts. The curve shows where they")
        print("  balance.")
        if best_th is not None:
            print(f"\n  Highest Sharpe occurs at <= {best_th:.0f}bp. DO NOT ADOPT THAT")
            print("  THRESHOLD FOR THAT REASON. Choosing the cutoff that maximises the")
            print("  backtest is precisely the bias the ex-ante rule exists to prevent.")
            print("  Pick a round number defensible in advance - 10bp is one - and report")
            print("  the whole curve so a reader can see the choice was not optimised.")

    print("\n" + "=" * 82)
    print("4. THE SINGLE-NAME QUESTION")
    print("=" * 82)
    worst = C.iloc[0]["symbol"]
    keep = all_syms - {worst}
    r = run(df, keep, cost_map)
    if base and r:
        print(f"  Excluding only {worst}, the most expensive contract at "
              f"{C.iloc[0]['cost_bp']:.1f}bp per side:\n")
        print(f"  {'':22s} {'Sharpe':>8s} {'return':>8s} {'vol':>7s} {'maxDD':>8s} "
              f"{'cost/yr':>8s}")
        print(f"  {'all 17 names':22s} {base['sharpe']:>8.3f} {base['ann']*100:>7.2f}% "
              f"{base['vol']*100:>6.1f}% {base['dd']*100:>7.1f}% "
              f"{base['cost_yr']*100:>7.2f}%")
        print(f"  {'without ' + worst:22s} {r['sharpe']:>8.3f} {r['ann']*100:>7.2f}% "
              f"{r['vol']*100:>6.1f}% {r['dd']*100:>7.1f}% {r['cost_yr']*100:>7.2f}%")
        print(f"  {'difference':22s} {r['sharpe']-base['sharpe']:>+8.3f} "
              f"{(r['ann']-base['ann'])*100:>+7.2f}% {'':>6s} {'':>7s} "
              f"{(r['cost_yr']-base['cost_yr'])*100:>+7.2f}%")
        share = 1 - r["cost_yr"] / base["cost_yr"] if base["cost_yr"] > 0 else np.nan
        print(f"\n  Removing one name of seventeen eliminates {share:.0%} of all")
        print(f"  transaction cost.")
        if r["sharpe"] > base["sharpe"]:
            print(f"  It also RAISES the Sharpe ratio, so the exclusion costs nothing in")
            print(f"  breadth that it does not repay in cost. That is a defensible universe")
            print(f"  rule and it was decided on contract specifications, not performance.")
        else:
            print(f"  It LOWERS the Sharpe ratio: the breadth {worst} provides is worth")
            print(f"  more than the cost it incurs. Keep it, and say why - a cost-based")
            print(f"  exclusion rule that was tested and rejected is worth reporting.")


if __name__ == "__main__":
    main()