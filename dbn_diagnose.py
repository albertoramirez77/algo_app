"""
dbn_diagnose.py — find out what Databento actually has, before spending anything.

    python dbn_diagnose.py --ping        # 1 call,  ~2 seconds. Does the API work at all?
    python dbn_diagnose.py               # ~15 calls, under a minute
    python dbn_diagnose.py --resolve     # adds exact listing dates. Slower.

WHY THE PREVIOUS VERSION REPORTED "never resolves" FOR EVERYTHING

It called symbology.resolve with start_date == end_date. Databento date ranges are
half-open, [start, end), so a range where start equals end is empty and resolves nothing —
for any symbol, including crude oil. It then repeated that useless call eight times per
product across thirteen products, which is why it appeared to hang.

This version asks one question per product, using get_record_count over the full range.
A record count is a better primitive than a cost anyway: cost cannot distinguish "cheap"
from "empty", and record count can.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

DATASET = "GLBX.MDP3"
SCHEMA = "statistics"

# We fetch the FULL-SIZE contract even where we trade the micro. The micro is a fractional
# clone of the same underlying with an identical price series, and the full-size contract
# has history to 2010 while Micro WTI listed in 2021 and Micro Copper in 2022. Requesting
# MCL.FUT from 2010 is what produced 422 symbology_invalid_request.
PROBE = {
    "MCL": "CL", "QG": "NG", "MGC": "GC", "SIL": "SI", "MHG": "HG",
    "ZC": "ZC", "ZW": "ZW", "KE": "KE", "ZS": "ZS", "ZM": "ZM", "ZL": "ZL",
    "LE": "LE", "HE": "HE",
}


def short(e: Exception) -> str:
    m = str(e).replace("\n", " ")
    return f"{type(e).__name__}: {m[:70]}"


def client():
    try:
        import databento as db
    except ImportError:
        raise SystemExit("pip install databento")
    print(f"python {sys.version.split()[0]}   databento {getattr(db, '__version__', '?')}")
    if sys.version_info < (3, 10):
        print("  note: recent databento releases want Python 3.10+. If calls behave oddly,")
        print("  that is the first thing to rule out.")
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("export DATABENTO_API_KEY='db-...' first")
    return db.Historical(key)


def dataset_range(c) -> tuple[str, str]:
    try:
        r = c.metadata.get_dataset_range(dataset=DATASET)
        g = (lambda *k: next((str(r[x])[:10] for x in k if x in r), None))
        start = g("start", "start_date") or "2010-06-06"
        end = g("end", "end_date") or str(date.today())
        print(f"{DATASET} available {start} to {end}\n")
        return start, end
    except Exception as e:
        print(f"get_dataset_range failed ({short(e)}); assuming 2010-06-06 to today\n")
        return "2010-06-06", str(date.today())


def count(c, root: str, start: str, end: str):
    """
    One call. Returns (records, error_string).

    A 422 symbology_invalid_request here means the root does not exist anywhere in
    [start, end) — normally because the contract listed later than `start`.
    """
    try:
        n = c.metadata.get_record_count(
            dataset=DATASET, schema=SCHEMA, symbols=[f"{root}.FUT"],
            stype_in="parent", start=start, end=end)
        return int(n), ""
    except Exception as e:
        return None, short(e)


def _search(c, root: str, cands: list[str], end: str) -> tuple[str | None, int]:
    """
    Binary search over candidate start dates. Availability is monotone — once a contract
    lists it stays listed — so if a request starting at date D succeeds, every later
    candidate does too. That makes bisection valid and cuts 17 probes to about 5.
    """
    lo, hi, best, calls = 0, len(cands) - 1, None, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        n, _ = count(c, root, cands[mid], end)
        calls += 1
        if n is not None and n > 0:
            best, hi = cands[mid], mid - 1
        else:
            lo = mid + 1
    return best, calls


def first_available(c, root: str, ds_start: str, end: str) -> tuple[str | None, int]:
    """
    Two-stage: find the first calendar year that works (~5 calls), then refine to the
    month within the preceding twelve (~4 calls). Without the refinement a contract that
    listed in April would be treated as unavailable until the following January, throwing
    away eight months of usable history.
    """
    years = [max(f"{y}-01-01", ds_start)
             for y in range(int(ds_start[:4]), int(end[:4]) + 1)]
    y, calls = _search(c, root, years, end)
    if y is None:
        return None, calls

    yr = int(y[:4])
    months = [f"{yr-1}-{m:02d}-01" for m in range(1, 13)] + [y]
    months = [m for m in months if m >= ds_start]
    if len(months) > 1:
        m, extra = _search(c, root, months, end)
        calls += extra
        if m:
            return m, calls
    return y, calls


def exact_listing(c, root: str, start: str, end: str) -> str | None:
    """Optional: earliest mapping interval from symbology. One call, over a REAL range."""
    try:
        r = c.symbology.resolve(
            dataset=DATASET, symbols=[f"{root}.FUT"], stype_in="parent",
            stype_out="instrument_id", start_date=start, end_date=end)
        result = r.get("result", {}) if isinstance(r, dict) else {}
        d0s = [iv.get("d0") for ivs in result.values() for iv in (ivs or [])
               if isinstance(iv, dict) and iv.get("d0")]
        return min(d0s)[:10] if d0s else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true", help="single call sanity check")
    ap.add_argument("--resolve", action="store_true", help="also get exact listing dates")
    ap.add_argument("--products", default=None,
                    help="comma-separated subset, e.g. ZC,CL — for a quick test")
    a = ap.parse_args()

    c = client()
    ds_start, ds_end = dataset_range(c)
    # half-open range: push the end out one day so the last session is included
    end = str(pd.Timestamp(ds_end).date() + timedelta(days=1))

    if a.ping:
        t = time.time()
        n, err = count(c, "ZC", "2024-01-02", "2024-01-09")
        print(f"ZC one week: records={n} {err}   ({time.time()-t:.1f}s)")
        print("\nIf that returned a number, the API and your key are fine.")
        return

    probe_set = PROBE
    if a.products:
        want = {x.strip().upper() for x in a.products.split(",")}
        probe_set = {k: v for k, v in PROBE.items() if k in want or v in want}
        if not probe_set:
            raise SystemExit(f"none of {want} in the universe")

    rows, t0 = [], time.time()
    for sym, root in probe_set.items():
        t = time.time()
        n, err = count(c, root, ds_start, end)
        calls, start_used, note = 1, ds_start, ""

        if n is None or n == 0:
            # did not exist at the dataset start; find when it did
            start_used, extra = first_available(c, root, ds_start, end)
            calls += extra
            if start_used is None:
                note = f"never resolves — check the root. last error: {err}"
            else:
                n, err = count(c, root, start_used, end)
                calls += 1
                note = "lists later than dataset start"

        cost = None
        if n:
            try:
                cost = float(c.metadata.get_cost(
                    dataset=DATASET, schema=SCHEMA, symbols=[f"{root}.FUT"],
                    stype_in="parent", start=start_used, end=end))
                calls += 1
            except Exception as e:
                note = (note + " " + short(e)).strip()

        listed = exact_listing(c, root, start_used or ds_start, end) if (a.resolve and n) else None
        rows.append(dict(trade=sym, fetch=root, usable_from=start_used,
                         listed=listed, records=n, cost=cost, note=note))
        print(f"  {sym:4s} <- {root:3s}  {str(start_used):10s}  "
              f"records={str(n):>10s}  cost={cost}  {calls} calls  "
              f"{time.time()-t:.1f}s  {note}")

    t = pd.DataFrame(rows)
    print(f"\n{'='*82}")
    print(t.to_string(index=False))
    print("=" * 82)
    print(f"\ntotal time {time.time()-t0:.0f}s")
    print(f"products with data: {(t['records'].fillna(0) > 0).sum()} of {len(t)}")
    print(f"total cost: ${t['cost'].fillna(0).sum():,.2f}   "
          f"total records: {t['records'].fillna(0).sum():,.0f}")

    print("\nHOW TO READ THIS")
    print("  records None/0  -> broken. Wrong root, or the range starts before it listed")
    print("  records large, cost 0 -> statistics is genuinely a cheap schema. Proceed")
    print("  usable_from later than 2010 -> that root lists later; the universe is")
    print("                                 time-varying and the engine must know")

    late = t[t["usable_from"].notna() & (t["usable_from"] > ds_start)]
    if len(late):
        print("\nPASTE INTO fetch_prices.py  ->  ROOT_START:")
        print("ROOT_START = {")
        for _, r in late.iterrows():
            print(f'    "{r["fetch"]}": "{r["usable_from"]}",')
        print("}")
    else:
        print("\nEvery root exists from the dataset start. ROOT_START can stay empty.")

    print("\nNOTE: ROOT_START is about the PRICE root. LISTED_FROM in immediacy.py is a")
    print("different thing — when the contract you actually TRADE became available. Micro")
    print("WTI listed in 2021 even though CL prices go back to 2010, and the engine must")
    print("exclude MCL before then. Set LISTED_FROM from the CME product pages, not here.")


if __name__ == "__main__":
    main()