"""
verify_listing.py — when did each MICRO contract actually become tradeable?

    python verify_listing.py

LISTED_FROM in immediacy.py controls when a name enters the book. Set it too early and
the backtest assumes access to a contract that did not exist; too late and you throw away
tradeable history. It is the last input in this project still resting on my recollection
rather than a measurement.

This asks Databento directly, on the MICRO roots (not the full-size roots we fetch prices
from), and prints a block to paste in. Record-count calls are metadata, so this is free.
"""
from __future__ import annotations
import os
import pandas as pd

DATASET = "GLBX.MDP3"
MICROS = {"MCL": "Micro WTI", "MGC": "Micro Gold",
          "SIL": "Micro Silver", "MHG": "Micro Copper", "KE": "KC Wheat"}


def main() -> None:
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("export DATABENTO_API_KEY='db-...' first")
    c = db.Historical(key)

    r = c.metadata.get_dataset_range(dataset=DATASET)
    end = next((str(r[k])[:10] for k in ("end", "end_date") if k in r), None)
    print(f"dataset available through {end}\n")

    out = {}
    for root, name in MICROS.items():
        # Ask a NARROW window at each candidate, not an open-ended range. The previous
        # version bisected on the start date of a range running to the present, so the
        # request always contained the contract's later life and always succeeded —
        # every root came back as first available in 2010, which is impossible.
        def has_data(day: str) -> bool:
            try:
                nxt = str((pd.Timestamp(day) + pd.Timedelta(days=31)).date())
                n = c.metadata.get_record_count(
                    dataset=DATASET, schema="statistics", symbols=[f"{root}.FUT"],
                    stype_in="parent", start=day, end=min(nxt, end))
                return n > 0
            except Exception:
                return False

        months = pd.date_range("2010-06-06", end, freq="MS").strftime("%Y-%m-%d").tolist()
        lo, hi, best = 0, len(months) - 1, None
        while lo <= hi:                      # availability is monotone once listed
            mid = (lo + hi) // 2
            if has_data(months[mid]):
                best, hi = months[mid], mid - 1
            else:
                lo = mid + 1

        if best is None:
            print(f"  {root:4s} {name:14s} no data found in any month")
            continue
        out[root] = best
        print(f"  {root:4s} {name:14s} first month with data {best}")

    print("\nPASTE INTO immediacy.py -> LISTED_FROM:")
    print("LISTED_FROM = {")
    for k, v in out.items():
        print(f'    "{k}": "{v}",')
    print("}")
    print("\nNote: this is when Databento first has data for the micro root, which is a")
    print("lower bound on tradeability and is the conservative direction to err.")


if __name__ == "__main__":
    main()