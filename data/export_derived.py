"""
export_derived.py — write the series a reader needs to check every reported number,
without redistributing any vendor data.

    python data/export_derived.py --prices data/px_clean.parquet

From the five files written here a reader reproduces the Sharpe ratio, the t-statistic,
annualised return and volatility, the drawdown and its duration, the regime table, the
block bootstrap, the jackknife, the placebo distribution and the portfolio combination —
with no Databento account.

Outputs, all to data/derived/:

    monthly_pnl.csv         month, strategy return net of costs, gross return, cost
    pnl_by_instrument.csv   month, symbol, contribution to P&L
    signal_ranks.csv        month, symbol, signal value, cross-sectional rank
    cost_table.csv          symbol, notional, tick value, cost in bp, annual volatility
    benchmarks.csv          month, equal-weighted complex, front momentum, carry
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import final_numbers as fn
    from universe import BY_SYMBOL
except ImportError:  # pragma: no cover
    raise SystemExit(
        "Run through the Makefile (`make derived`) so engine/ is on PYTHONPATH, or:\n"
        "    PYTHONPATH=engine python data/export_derived.py --prices data/px_clean.parquet"
    )

OUT = Path("data/derived")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    df = fn.load_daily(a.prices)
    costs = fn.ex_ante_costs(df)
    med = costs["cost_bp"].median()
    keep = set(costs.loc[costs["cost_bp"] <= fn.COST_MULTIPLE * med, "symbol"])
    cost_map = dict(zip(costs["symbol"], costs["cost_bp"]))
    print(f"  universe: {len(keep)} instruments "
          f"(median cost {med:.2f}bp, threshold {fn.COST_MULTIPLE * med:.2f}bp)")

    # ---- 1. cost table: the ex-ante universe rule, computed from specs alone --------
    costs.assign(excluded=~costs["symbol"].isin(keep)).to_csv(
        OUT / "cost_table.csv", index=False)

    # ---- 2. the tranched book, month by month --------------------------------------
    frames = [f for f in (fn.grid_targets(df, o, keep) for o in range(fn.N_GRIDS))
              if not f.empty]
    book = fn.tranched(df, frames, keep, cost_map=cost_map)

    daily = pd.DataFrame({"net": book["net"], "gross": book["gross"],
                          "cost": book["cost"]})
    monthly_pnl = daily.resample("ME").sum()
    monthly_pnl = monthly_pnl[monthly_pnl["net"] != 0]
    monthly_pnl.index.name = "month"
    monthly_pnl.round(8).to_csv(OUT / "monthly_pnl.csv")

    s = fn.st(monthly_pnl["net"])
    print(f"  monthly_pnl.csv        {len(monthly_pnl)} months, "
          f"Sharpe {s['sharpe']:.3f}, t {s['t']:.2f}")

    # ---- 3. signal and rank, month by month ----------------------------------------
    m = fn.monthly(df, keep=keep)
    sig = m[["ym", "symbol", "bm", "vol", "carry"]].dropna(subset=["bm"]).copy()
    sig["rank"] = sig.groupby("ym")["bm"].rank()
    sig["weight"] = sig.groupby("ym")["rank"].transform(lambda r: r - r.mean())
    sig["weight"] = sig.groupby("ym")["weight"].transform(
        lambda w: w / np.abs(w).sum() if np.abs(w).sum() > 0 else w)
    sig = sig.rename(columns={"ym": "month", "bm": "signal"})
    sig["month"] = sig["month"].astype(str)
    num = sig.select_dtypes("number").columns
    sig[num] = sig[num].round(8)
    sig.to_csv(OUT / "signal_ranks.csv", index=False)
    print(f"  signal_ranks.csv       {len(sig)} rows")

    # ---- 4. P&L contribution per instrument ----------------------------------------
    # Single-grid monthly attribution: the same construction the jackknife uses.
    rows = []
    for sym in sorted(keep):
        r = fn.mbook(m, drop=None) - fn.mbook(m, drop=sym)
        for ym, v in r.items():
            rows.append(dict(month=ym, symbol=sym, contribution=v))
    contrib = pd.DataFrame(rows)
    contrib["month"] = contrib["month"].astype(str)
    contrib["contribution"] = contrib["contribution"].round(8)
    contrib.to_csv(OUT / "pnl_by_instrument.csv", index=False)
    print(f"  pnl_by_instrument.csv  {len(rows)} rows")

    # ---- 5. benchmarks on the identical universe and period ------------------------
    mkt = (m.groupby("ym")["r0"].mean().rename("equal_weighted_complex"))
    bench = pd.DataFrame({
        "equal_weighted_complex": mkt,
        "front_momentum": fn.mbook(m, sig="mom0"),
        "carry": fn.mbook(m, sig="carry"),
    })
    bench.index = bench.index.astype(str)
    bench.index.name = "month"
    bench.round(8).to_csv(OUT / "benchmarks.csv")
    print(f"  benchmarks.csv         {len(bench)} months")
    print(f"  cost_table.csv         {len(costs)} instruments\n"
          f"  wrote everything to {OUT}/")


if __name__ == "__main__":
    main()
