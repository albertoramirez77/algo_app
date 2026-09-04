"""
backtest_bm.py — run the surviving strategy with integer contracts and real costs.

    python backtest_bm.py --prices data/px_clean.parquet

WHY A BACKTEST AND NOT A TABLE

A static capacity table has produced a wrong answer twice in this project. The first time
it divided every product's session count by the whole file's span, understating any
instrument that listed late. The second time it read CME's cent-quoted grain settles as
dollars, inflating seven of seventeen contracts by 100x and reporting that three
instruments were tradeable when the real figure was quite different.

The definitive answer is to run the strategy. Size it, round it to integers, charge the
costs, and measure what the rounding actually destroys.

WHAT IS BEING TESTED

Commodity basis-momentum, nearby returns, 12-month formation. It survived:

    Sharpe +0.602 (t 2.34)   placebo +2.3 sd   alpha over momentum +5.59%/yr (t 2.42)
    best 6 of 182 months = 50.1% of P&L       12 of 16 years positive
    all three subperiods positive              correlation to trend +0.072
    turnover 17%/month, so net SR 0.579 even at 10bp per side

It remains a PARTIAL replication: published Sharpe 0.9, expected t 3.62, measured 2.34,
and the spreading variant Boons & Prado report alongside nearby failed outright.

THE QUESTION HERE IS NARROWER

Does integer contract granularity at $450,000 destroy it? Rank weighting means the extreme
positions carry roughly twice the average weight and the middle of the book sits near zero,
so "can every instrument hold two contracts" was the wrong question. What matters is how
much of the fractional strategy's Sharpe survives rounding.
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
VOL_TARGET = 0.20
VOL_WINDOW = 6          # months of trailing return history for the vol estimate
IDM_CAP = 2.5
AUM_SWEEP = [250_000, 450_000, 1_000_000, 2_500_000, 5_000_000,
             10_000_000, 25_000_000, 50_000_000]


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
                px_last=("settle_0", "last"), n_days=("r0", "size"))
           .reset_index())
    m["asset"] = m["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    m = m[(m["n_days"] >= 10) & (m["asset"] == "commodity")].copy()

    c0 = m.groupby("symbol")["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    c1 = m.groupby("symbol")["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum())
    m["bm"] = c0 - c1
    # Sizing inputs must be knowable at the decision point: trailing vol, shifted, and the
    # PREVIOUS month's closing price, which is the price we would size against.
    m["vol"] = (m.groupby("symbol")["r0"]
                  .transform(lambda s: s.rolling(VOL_WINDOW, min_periods=3).std())
                  .groupby(m["symbol"]).shift(1)) * np.sqrt(12)
    m["px_entry"] = m.groupby("symbol")["px_last"].shift(1)
    m["fwd"] = m.groupby("symbol")["r0"].shift(-1)
    return m.sort_values(["symbol", "ym"]).reset_index(drop=True)


def idm_from(m: pd.DataFrame, n: int) -> float:
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    return rho, min(1.0 / np.sqrt((1/n) + (1 - 1/n) * max(rho, 0.01)), IDM_CAP)


def run(m: pd.DataFrame, capital: float, idm: float, integer: bool,
        bps_per_side: float = 3.0, min_n: int = 6) -> dict:
    """
    One pass of the strategy. `integer=False` gives the fractional ideal; `integer=True`
    rounds every position to a whole contract, which is the only thing actually tradeable.
    """
    prev = {}
    rows = []
    for ym, g in m.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        rk = s["bm"].rank()
        w = rk - rk.mean()
        gross = w.abs().sum()
        if gross <= 0:
            continue
        w = w / gross                                    # sum |w| = 1

        pnl = cost = 0.0
        held, dollars = {}, 0.0
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"],
                                         s["fwd"]):
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult                 # multiplier x price scale
            contract_dvol = dpm * px * vol               # annualised $ vol of one lot
            if contract_dvol <= 0:
                continue
            target = wi * capital * VOL_TARGET * idm / contract_dvol
            n = float(np.round(target)) if integer else target
            if n == 0:
                # position rounded away entirely: the granularity cost, made explicit
                pass
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            dollars += abs(n) * dpm * px
            traded = abs(n - prev.get(sym, 0.0))
            if traded > 0:
                cost += traded * (inst.commission +
                                  abs(dpm) * px * bps_per_side / 1e4)
        for sym in set(prev) - set(held):                # exited positions
            inst = BY_SYMBOL[sym]
            cost += abs(prev[sym]) * inst.commission
        prev = held
        rows.append(dict(ym=ym, pnl=pnl, cost=cost, net=pnl - cost,
                         gross_notional=dollars,
                         n_pos=sum(1 for v in held.values() if v != 0),
                         n_zeroed=sum(1 for v in held.values() if v == 0),
                         n_names=len(s)))
    if len(rows) < 60:
        return dict(n=len(rows))
    d = pd.DataFrame(rows).set_index("ym")
    r = d["net"] / capital
    rg = d["pnl"] / capital
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    eq = (1 + r).cumprod()
    return dict(n=len(r), years=yrs,
                ann_ret=r.mean() * 12, ann_vol=av,
                sharpe=(r.mean() * 12) / av if av > 0 else np.nan,
                t=((r.mean() * 12) / av) * np.sqrt(yrs) if av > 0 else np.nan,
                gross_sharpe=(rg.mean() * 12) / (rg.std(ddof=1) * np.sqrt(12)),
                max_dd=float((eq / eq.cummax() - 1).min()),
                cost_pct=d["cost"].sum() / capital / yrs,
                lev=d["gross_notional"].mean() / capital,
                n_pos=d["n_pos"].mean(), n_zeroed=d["n_zeroed"].mean(),
                n_names=d["n_names"].mean(), series=r)


def line(label: str, s: dict) -> None:
    if "sharpe" not in s:
        print(f"  {label:26s} too few months ({s.get('n', 0)})")
        return
    print(f"  {label:26s} SR {s['sharpe']:>+6.3f}  t {s['t']:>+5.2f}  "
          f"ret {s['ann_ret']*100:>+6.2f}%  vol {s['ann_vol']*100:>5.2f}%  "
          f"dd {s['max_dd']*100:>+6.1f}%  cost {s['cost_pct']*100:>4.2f}%  "
          f"lev {s['lev']:>4.1f}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--capital", type=float, default=450_000)
    ap.add_argument("--bps", type=float, default=3.0)
    a = ap.parse_args()

    m = load_monthly(a.prices)
    n_inst = m["symbol"].nunique()
    rho, idm = idm_from(m, n_inst)

    print("=" * 78)
    print("1. SETUP")
    print("=" * 78)
    print(f"  {n_inst} commodities, {m['ym'].nunique()} months, "
          f"{m['ym'].min()} to {m['ym'].max()}")
    print(f"  average pairwise correlation {rho:+.3f}   IDM {idm:.2f}")
    print(f"  vol target {VOL_TARGET:.0%}, sizing vol from a trailing "
          f"{VOL_WINDOW}-month window, lagged")
    print(f"  cost {a.bps:.0f}bp per side of notional plus per-contract commission")

    print("\n  contract notionals — the units fix, verified:")
    last = m[m["ym"] == m["ym"].max()].set_index("symbol")
    for sym in sorted(last.index):
        inst = BY_SYMBOL[sym]
        px = last.at[sym, "px_last"]
        print(f"    {sym:4s} settle {px:>9.2f}  x{inst.price_scale:<5.2f} "
              f"x{inst.multiplier:>8,.0f}  = ${px*inst.dollar_price_mult:>10,.0f}")

    print("\n" + "=" * 78)
    print(f"2. FRACTIONAL IDEAL vs INTEGER REALITY at ${a.capital:,.0f}")
    print("=" * 78)
    frac = run(m, a.capital, idm, integer=False, bps_per_side=a.bps)
    intg = run(m, a.capital, idm, integer=True, bps_per_side=a.bps)
    line("fractional (unreachable)", frac)
    line("integer (tradeable)", intg)
    if "sharpe" in frac and "sharpe" in intg:
        keep = intg["sharpe"] / frac["sharpe"] if frac["sharpe"] != 0 else np.nan
        print(f"\n  integer keeps {keep:.0%} of the fractional Sharpe")
        print(f"  positions held {intg['n_pos']:.1f} of {intg['n_names']:.1f} names; "
              f"{intg['n_zeroed']:.1f} rounded away to zero each month")
        if "series" in frac and "series" in intg:
            j = pd.concat([frac["series"].rename("f"), intg["series"].rename("i")],
                          axis=1).dropna()
            te = (j["i"] - j["f"]).std(ddof=1) * np.sqrt(12)
            print(f"  tracking error of integer against fractional: {te*100:.2f}%/yr")

    print("\n" + "=" * 78)
    print("3. WHERE DOES GRANULARITY STOP BINDING?")
    print("=" * 78)
    print(f"  {'AUM':>12s} {'int SR':>8s} {'frac SR':>8s} {'kept':>6s} "
          f"{'zeroed':>7s} {'cost%':>6s} {'lev':>5s}")
    rows = []
    for aum in AUM_SWEEP:
        f = run(m, aum, idm, integer=False, bps_per_side=a.bps)
        i = run(m, aum, idm, integer=True, bps_per_side=a.bps)
        if "sharpe" not in i or "sharpe" not in f:
            continue
        keep = i["sharpe"] / f["sharpe"] if f["sharpe"] != 0 else np.nan
        rows.append(dict(aum=aum, int_sr=i["sharpe"], frac_sr=f["sharpe"], keep=keep,
                         zeroed=i["n_zeroed"], cost=i["cost_pct"], lev=i["lev"]))
        print(f"  ${aum:>11,.0f} {i['sharpe']:>+8.3f} {f['sharpe']:>+8.3f} "
              f"{keep:>5.0%} {i['n_zeroed']:>7.1f} {i['cost_pct']*100:>5.2f}% "
              f"{i['lev']:>4.1f}x")
    print("\n  'zeroed' is the number of intended positions per month that round to no")
    print("  contracts at all. That is the granularity cost, stated directly.")

    print("\n" + "=" * 78)
    print("4. WHICH INSTRUMENTS SURVIVE ROUNDING AT $450,000?")
    print("=" * 78)
    counts = {}
    prev = {}
    for ym, g in m.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < 6:
            continue
        rk = s["bm"].rank(); w = rk - rk.mean(); gr = w.abs().sum()
        if gr <= 0:
            continue
        w = w / gr
        for sym, wi, vol, px in zip(s["symbol"], w, s["vol"], s["px_entry"]):
            inst = BY_SYMBOL[sym]
            cdv = inst.dollar_price_mult * px * vol
            tgt = abs(wi) * a.capital * VOL_TARGET * idm / cdv if cdv > 0 else 0
            c = counts.setdefault(sym, dict(n=0, zero=0, tgt=[]))
            c["n"] += 1
            c["zero"] += (round(tgt) == 0)
            c["tgt"].append(tgt)
    tab = pd.DataFrame([
        dict(symbol=k, months=v["n"], zero_pct=v["zero"] / v["n"],
             median_target=float(np.median(v["tgt"])),
             p90_target=float(np.percentile(v["tgt"], 90)))
        for k, v in counts.items()]).sort_values("median_target", ascending=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
    print("\n  median_target is the typical ABSOLUTE position for that instrument. Rank")
    print("  weighting means the extremes carry roughly twice the average, so p90 is the")
    print("  number that matters for whether the signal can be expressed at all.")
    ok = (tab["zero_pct"] < 0.5).sum()
    print(f"\n  instruments holding a position in the majority of months: {ok} of {len(tab)}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if "sharpe" in intg and "sharpe" in frac:
        keep = intg["sharpe"] / frac["sharpe"] if frac["sharpe"] != 0 else 0
        checks = [
            ("integer Sharpe above 0.35 after costs", intg["sharpe"] > 0.35),
            ("integer keeps 70%+ of fractional Sharpe", keep > 0.70),
            ("fewer than a third of positions rounded away",
             intg["n_zeroed"] < intg["n_names"] / 3),
            ("gross leverage under 4x", intg["lev"] < 4.0),
        ]
        for k, v in checks:
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        print()
        if all(v for _, v in checks):
            print("  Tradeable at $450,000. Report the integer Sharpe as the headline —")
            print("  the fractional number is not reachable and quoting it would be")
            print("  dishonest. Keep the partial-replication caveat: t 2.34 against an")
            print("  expected 3.62, and the spreading variant failed.")
        else:
            print("  Granularity degrades it materially at this size. That is a legitimate")
            print("  finding and belongs in Capital and Liquidity: here is an effect that")
            print("  survives its diagnostics and cannot be fully expressed at $450,000,")
            print("  and here is the AUM at which it can. Do not hide it in a caveat.")


if __name__ == "__main__":
    main()