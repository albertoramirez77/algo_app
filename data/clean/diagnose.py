"""
diagnose.py — pin down the three verification failures exactly.

    python diagnose.py --prices data/px_clean.parquet

Three checks failed in verify.py. Guessing at causes is how this project produced a
timestamp bug that corrupted an entire dataset. So this locates each one precisely and
answers the only question that matters: DOES IT TOUCH THE STRATEGY?

The strategy trades commodities only. Anything confined to equity, rates or FX is noise in
a file that happens to be wider than the strategy needs.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    a = ap.parse_args()

    df = pd.read_parquet(a.prices)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    comm = df["asset"] == "commodity"

    print("=" * 78)
    print("FAILURE 1 — 6,007 duplicate (date, symbol) rows")
    print("=" * 78)
    dup_mask = df.duplicated(["date", "symbol"], keep=False)
    dups = df[dup_mask]
    print(f"  {len(dups):,} rows involved, {dups.groupby(['date','symbol']).ngroups:,} "
          f"distinct date-instrument pairs")
    print(f"\n  by asset class:")
    for asset, g in dups.groupby("asset"):
        tot = (df["asset"] == asset).sum()
        print(f"    {asset:10s} {len(g):>6,} of {tot:>7,} rows  ({len(g)/tot:.1%})")

    print(f"\n  by instrument, commodities only:")
    cd = dups[dups["asset"] == "commodity"]
    if len(cd):
        for sym, g in cd.groupby("symbol"):
            tot = (df["symbol"] == sym).sum()
            print(f"    {sym:5s} {len(g):>5,} of {tot:>6,}  ({len(g)/tot:.1%})")
    else:
        print("    NONE — every duplicate is in a financial instrument")

    # are duplicates roll days? a roll day has two different contract_0 values
    print(f"\n  are these roll days?")
    grp = dups.groupby(["date", "symbol"])["contract_0"].nunique()
    print(f"    pairs where contract_0 DIFFERS between the duplicate rows: "
          f"{(grp > 1).sum():,} of {len(grp):,} ({(grp > 1).mean():.0%})")
    print("    A roll day maps the continuous series to two contracts on one date. That")
    print("    is expected and the deduplication rule exists precisely for it.")

    print(f"\n  does the tie-break rule change the price it keeps?")
    if len(cd):
        keep_hi = (cd.sort_values(["symbol", "date", "oi_0"], na_position="first")
                     .drop_duplicates(["date", "symbol"], keep="last"))
        keep_lo = (cd.sort_values(["symbol", "date", "oi_0"], na_position="last")
                     .drop_duplicates(["date", "symbol"], keep="last"))
        j = keep_hi.merge(keep_lo, on=["date", "symbol"], suffixes=("_hi", "_lo"))
        diff = (j["settle_0_hi"] != j["settle_0_lo"]).mean()
        rel = ((j["settle_0_hi"] - j["settle_0_lo"]).abs() /
               j["settle_0_hi"].abs()).replace([np.inf, -np.inf], np.nan)
        print(f"    price differs between the two choices in {diff:.0%} of cases")
        print(f"    median relative difference when it does: {rel.median():.4%}")
        print(f"    max relative difference: {rel.max():.2%}")
        print("    If the prices are near-identical the tie-break barely matters. If they")
        print("    differ materially, the rule is a real modelling choice you must defend.")
    else:
        print("    not applicable — no commodity duplicates")

    print("\n" + "=" * 78)
    print("FAILURE 2 — 2 non-positive settlement prices")
    print("=" * 78)
    bad = df[(df["settle_0"] <= 0) | (df["settle_1"] <= 0)]
    if len(bad):
        cols = [c for c in ("date", "symbol", "asset", "contract_0", "settle_0",
                            "contract_1", "settle_1") if c in bad.columns]
        print(bad[cols].to_string(index=False))
        n_comm = (bad["asset"] == "commodity").sum()
        print(f"\n  in commodities: {n_comm} of {len(bad)}")
        if n_comm == 0:
            print("  IRRELEVANT to the strategy — the book trades commodities only.")
        else:
            print("  These touch the strategy. A non-positive settle makes log(price)")
            print("  undefined, so the return is NaN and that instrument-month drops out")
            print("  of the cross-section. Two rows out of 130,864 cannot move a Sharpe,")
            print("  but you should be able to name the date and the cause.")
            print("  April 2020 WTI settled NEGATIVE for real. If the date is 2020-04-20")
            print("  that is not a data error, it is history.")
    else:
        print("  none found")

    print("\n" + "=" * 78)
    print("FAILURE 3 — MNQ and MYM medians above 5,000")
    print("=" * 78)
    print("  This is a false alarm in the CHECK, not a defect in the data.\n")
    for sym in ("MNQ", "MYM", "MES", "M2K"):
        if sym not in set(df["symbol"]):
            continue
        s = df[df["symbol"] == sym]
        inst = BY_SYMBOL[sym]
        med = s["settle_0"].median()
        print(f"    {sym:5s} median settle {med:>10,.1f}   "
              f"x{inst.multiplier} = ${med*inst.dollar_price_mult:>10,.0f} notional  "
              f"({inst.sector})")
    print("\n  The Nasdaq trades near 20,000 and the Dow near 40,000. Those are index")
    print("  levels, not fixed-point integers, and the resulting contract notionals are")
    print("  sane. The check used a flat 5,000 threshold written for commodity prices.")
    print("  Both instruments are EQUITY and the strategy trades commodities only.")

    print("\n" + "=" * 78)
    print("BOTTOM LINE — does any of this touch the strategy?")
    print("=" * 78)
    n_comm_dup = len(cd)
    n_comm_bad = (bad["asset"] == "commodity").sum() if len(bad) else 0
    print(f"  commodity rows with duplicates:        {n_comm_dup:,}")
    print(f"  commodity rows with bad prices:        {n_comm_bad}")
    print(f"  equity/rates/FX issues:                irrelevant, not traded")
    print()
    if n_comm_dup:
        print("  The duplicates ARE in commodities, so the deduplication rule is doing")
        print("  real work. It is applied identically in the strategy and in the")
        print("  independent verification, and both produce Sharpe 0.760 — so the rule is")
        print("  at least consistent. Be ready to say what it is: on a roll day, keep the")
        print("  contract with the higher open interest.")
    else:
        print("  Nothing here touches the commodity book.")


if __name__ == "__main__":
    main()