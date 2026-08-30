"""
dbn_strategy_test.py — which request shape should we actually use?

    python dbn_strategy_test.py

Three ways to ask for the same product-year. Measures each. Costs a few cents total.

WHY THIS EXISTS

The log showed 376,392 records taking 3,450 seconds for one product-year, and 2,190
distinct contracts for CL over 2010-2013. Crude lists roughly 110-130 outright months at a
time, so ~150-180 distinct outrights is what that span should produce. The rest are
calendar spreads (CLZ4-CLZ5), which parent symbology returns because they share the root.

The connection is not the constraint — 151 Mbps at 25 ms latency. The constraint is that
we are asking the server to scan thousands of instruments per request when the strategy
only ever holds the front contract.

THE THREE APPROACHES

  parent      CL.FUT           every instrument sharing the root: outrights AND spreads
  oi rank     CL.n.0, CL.n.1   front two by open interest, ranked on the PREVIOUS session
  calendar    CL.c.0, CL.c.1   front two by a fixed calendar rule

Both continuous forms are non-anticipating: the open-interest rank uses the prior day's
statistics, and the calendar rule is knowable years ahead. Neither can leak.

WHAT WE GIVE UP BY MOVING OFF parent

With parent we hold every expiry and apply our own roll (5 business days before the
earlier of first notice or last trade). With continuous symbology we adopt the vendor's
roll, or we pull the front two series and switch between them ourselves. The test reports
which contract each series actually resolves to on each day, so you can check where the
roll lands relative to expiry before committing.
"""

from __future__ import annotations

import os
import time

import pandas as pd

DATASET = "GLBX.MDP3"
TEST_START, TEST_END = "2023-03-01", "2023-03-08"   # ONE WEEK. See note below.
ROOT = "CL"                                          # the worst case in the universe
DAYS = 5                                             # trading days in the test window
YEARS, PRODUCTS = 16, 13                             # for extrapolation

# One week, not one year. At the throughput observed in the logs (~110 records/sec on
# CL.FUT) a single year of the parent baseline would take roughly an hour on its own,
# which defeats the point of a diagnostic. A week is enough to measure the rate and the
# instrument count, and both extrapolate linearly.


def client():
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("export DATABENTO_API_KEY='db-...' first")
    print(f"databento {getattr(db, '__version__', '?')}\n")
    return db.Historical(key)


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name and df.index.name not in df.columns:
        return df.reset_index()
    return df.reset_index(drop=True)


def attempt(c, label: str, symbols: list[str], stype: str) -> dict:
    t = time.time()
    try:
        raw = c.timeseries.get_range(
            dataset=DATASET, schema="statistics", symbols=symbols,
            stype_in=stype, start=TEST_START, end=TEST_END).to_df()
    except Exception as e:
        return dict(approach=label, error=f"{type(e).__name__}: {str(e)[:70]}")
    el = time.time() - t
    if raw.empty:
        return dict(approach=label, error="empty result")

    df = flatten(raw)
    col = "symbol" if "symbol" in df.columns else "raw_symbol"
    s = df[col].astype(str)
    spreads = s.str.contains("-", regex=False)

    return dict(
        approach=label, seconds=round(el, 1), records=len(df),
        rec_per_sec=round(len(df) / el) if el > 0 else None,
        instruments=s.nunique(),
        spreads=int(s[spreads].nunique()),
        sample=s.iloc[0],
    )


def resolve_underlying(c, cont: str) -> pd.DataFrame:
    """Which real contract does a continuous symbol point at, and when does it roll?"""
    try:
        r = c.symbology.resolve(
            dataset=DATASET, symbols=[cont], stype_in="continuous",
            stype_out="raw_symbol", start_date=TEST_START, end_date=TEST_END)
        res = r.get("result", {}) if isinstance(r, dict) else {}
        rows = [dict(start=iv.get("d0"), end=iv.get("d1"), contract=iv.get("s"))
                for ivs in res.values() for iv in (ivs or []) if isinstance(iv, dict)]
        return pd.DataFrame(rows).sort_values("start") if rows else pd.DataFrame()
    except Exception as e:
        print(f"  resolve failed for {cont}: {type(e).__name__}")
        return pd.DataFrame()


def main() -> None:
    c = client()
    print(f"One year of {ROOT} statistics, {TEST_START} to {TEST_END}\n")

    trials = [
        ("parent  (all expiries + spreads)", [f"{ROOT}.FUT"], "parent"),
        ("oi rank (front two)", [f"{ROOT}.n.0", f"{ROOT}.n.1"], "continuous"),
        ("calendar (front two)", [f"{ROOT}.c.0", f"{ROOT}.c.1"], "continuous"),
    ]
    rows = []
    for label, syms, stype in trials:
        print(f"  {label} ... ", end="", flush=True)
        r = attempt(c, label, syms, stype)
        rows.append(r)
        if "error" in r:
            print(r["error"])
        else:
            print(f"{r['records']:>8,} rec  {r['seconds']:>6.1f}s  "
                  f"{r['rec_per_sec']:>7,}/s  {r['instruments']:>5,} instruments "
                  f"({r['spreads']:,} spreads)")

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("\nEvery approach failed. That is an account or endpoint problem.")
        return

    print("\n" + "=" * 74)
    base = next((r for r in ok if r["approach"].startswith("parent")), None)
    for r in ok:
        per_year = r["records"] * 252 / DAYS
        full = per_year * YEARS * PRODUCTS
        hours = full / max(r["rec_per_sec"], 1) / 3600
        line = (f"  {r['approach']:34s} full download ≈ {full:>12,.0f} records, "
                f"{hours:>6.1f}h")
        if base and r is not base and base["records"]:
            line += f"   ({base['records'] / max(r['records'], 1):,.0f}x less)"
        print(line)

    print("\nWHICH CONTRACT DOES EACH CONTINUOUS SERIES POINT AT?")
    for cont in (f"{ROOT}.n.0", f"{ROOT}.c.0"):
        d = resolve_underlying(c, cont)
        if d.empty:
            continue
        print(f"\n  {cont} roll schedule:")
        print("   ", d.head(14).to_string(index=False).replace("\n", "\n    "))

    print("\nHOW TO CHOOSE")
    print("  If a continuous form is dramatically faster AND its roll lands comfortably")
    print("  before first notice, take it. Document the rule you adopted: 'front contract")
    print("  by open interest rank on the previous session' is non-anticipating and is")
    print("  how the hedgers themselves roll, which is easy to defend.")
    print("  If the roll sits too close to expiry, pull both .0 and .1 and switch between")
    print("  them on your own 5-business-day rule instead.")


if __name__ == "__main__":
    main()