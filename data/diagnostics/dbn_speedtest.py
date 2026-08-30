"""
dbn_speedtest.py — two tiny requests. Tells us why the download is failing.

    python dbn_speedtest.py

Costs a fraction of a cent. Answers three questions:

  1. How fast is the connection actually moving data?
     The log showed one product-year taking 3,450 seconds for 376,392 records.
     That is ~110 records/sec. Databento streaming normally runs orders of magnitude
     faster. If this test confirms a slow rate, the bottleneck is the connection or
     throttling, not the request size, and no amount of chunking will fix it.

  2. Is the payload full of instruments we do not want?
     387,035 rows across 2,190 distinct contracts was reported for CL over 2010-2013.
     Crude lists roughly 110-130 outright months at a time, so about 150-180 distinct
     outrights would be expected over that span. 2,190 is an order of magnitude too many.
     The extra instruments are almost certainly calendar spreads, which carry a root like
     CLZ4-CLZ5 and settle at a spread price of a few cents.

  3. How long would the real download take at the observed rate?
"""

from __future__ import annotations

import os
import time

import pandas as pd

DATASET = "GLBX.MDP3"
DAY_START, DAY_END = "2024-03-14", "2024-03-15"


def client():
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("export DATABENTO_API_KEY='db-...' first")
    print(f"databento {getattr(db, '__version__', '?')}")
    return db.Historical(key)


def one_day(c, root: str) -> dict:
    t = time.time()
    try:
        df = c.timeseries.get_range(
            dataset=DATASET, schema="statistics", symbols=[f"{root}.FUT"],
            stype_in="parent", start=DAY_START, end=DAY_END).to_df()
    except Exception as e:
        return dict(root=root, error=f"{type(e).__name__}: {str(e)[:60]}")
    el = time.time() - t

    if df.index.name and df.index.name not in df.columns:
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)

    col = "symbol" if "symbol" in df.columns else "raw_symbol"
    syms = df[col].astype(str)
    spread = syms.str.contains("-", regex=False)
    outright = ~spread

    return dict(
        root=root, seconds=round(el, 1), records=len(df),
        rec_per_sec=round(len(df) / el, 1) if el > 0 else None,
        instruments=syms.nunique(),
        outrights=syms[outright].nunique(),
        spreads=syms[spread].nunique(),
        spread_share=round(spread.mean(), 3),
        example_outright=syms[outright].iloc[0] if outright.any() else None,
        example_spread=syms[spread].iloc[0] if spread.any() else None,
    )


def main() -> None:
    c = client()
    print(f"\nOne trading day ({DAY_START}) of statistics, two products.\n")
    rows = []
    for root in ("ZC", "CL"):
        r = one_day(c, root)
        rows.append(r)
        if "error" in r:
            print(f"  {root}: {r['error']}")
            continue
        print(f"  {root}: {r['records']:>7,} records in {r['seconds']:>6.1f}s "
              f"= {r['rec_per_sec']:>9,.0f} rec/s")
        print(f"      {r['instruments']:>5,} instruments = "
              f"{r['outrights']:,} outrights + {r['spreads']:,} spreads "
              f"({r['spread_share']:.0%} of records are spreads)")
        if r["example_spread"]:
            print(f"      e.g. outright {r['example_outright']}  "
                  f"spread {r['example_spread']}")

    good = [r for r in rows if "error" not in r and r.get("rec_per_sec")]
    if not good:
        print("\nBoth requests failed. That is a connection or account problem, not a")
        print("request-size problem — a single trading day is as small as it gets.")
        return

    print("\n" + "=" * 70)
    rate = min(r["rec_per_sec"] for r in good)
    cl = next((r for r in good if r["root"] == "CL"), good[0])
    per_year = cl["records"] * 252
    print(f"observed rate: {rate:,.0f} records/sec")
    print(f"CL is ~{cl['records']:,} records/day -> ~{per_year:,.0f}/year "
          f"-> ~{per_year*16:,.0f} for 16 years")
    print(f"at {rate:,.0f} rec/s that is {per_year*16/rate/3600:,.1f} hours "
          f"for CL alone")

    print("\nREAD IT LIKE THIS")
    print("  rate under ~1,000 rec/s   -> connection or throttling is the bottleneck.")
    print("                               Chunking will not fix it. Use batch downloads,")
    print("                               or a different network.")
    print("  spreads a large share     -> we are downloading (and paying for) instruments")
    print("                               the strategy never trades, AND they can be")
    print("                               silently selected as the front contract.")
    if any(r.get("spreads", 0) for r in good):
        print("\nSPREADS ARE PRESENT. This is a correctness bug, not just a cost one:")
        print("build_front_series picks the nearest-expiry instrument, and a calendar")
        print("spread has an expiry too. A spread settling at 0.15 would be selected as")
        print("the front 'price' of crude oil and nothing would complain.")


if __name__ == "__main__":
    main()