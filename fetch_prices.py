"""
fetch_prices.py — build px.parquet from Databento CME data.

Chunked, cached and resumable. Safe to Ctrl-C and restart: completed chunks are on disk
and are not re-downloaded.

    python dbn_diagnose.py                        # free: what exists and what it costs
    python fetch_prices.py --inspect --symbol ZC  # pennies: confirm stat_type codes
    python fetch_prices.py --estimate             # free: cost with record counts
    python fetch_prices.py --run --out px.parquet
    python fetch_prices.py --run --products MCL,QG   # retry just the ones that failed

WHY THE PREVIOUS VERSION TIMED OUT

It asked for sixteen years of statistics across every listed expiry in one streaming
request. Crude and natural gas list monthly contracts years forward, so those requests are
by far the largest in the universe and the gateway returns 504. This version requests one
year at a time, and on failure bisects the range and retries, so a stubborn year degrades
into halves and quarters rather than failing outright.

WHY WE NO LONGER DOWNLOAD SIXTEEN YEARS OF DEFINITIONS

We need each contract's expiry only to decide the roll date. The definition schema emits a
snapshot per instrument per day, so sixteen years of it across a hundred live expiries is
far larger than the statistics we actually want. Instead:

  - for expired contracts, the expiry is derived from the data already downloaded: the last
    date on which the contract has a settlement.
  - for contracts still live, that trick fails (their last settlement is today), so we pull
    the definition schema for a short recent window only.

Deriving an expiry from data is not look-ahead. Futures expiry dates are published by the
exchange years ahead; this is public ex-ante information that we are recovering
conveniently rather than peeking at. The roll rule itself remains a fixed calendar offset.

WHY THE statistics SCHEMA AND NOT DAILY BARS

`ohlcv-1d` is aggregated on UTC dates from electronic-session trades, so one "day" straddles
two CME sessions and its close is the last trade rather than the venue settlement.
Settlement is what a position is marked and rolled at.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from datetime import date, timedelta

import pandas as pd

from immediacy import UNIVERSE, LISTED_FROM

DATASET = "GLBX.MDP3"
START = "2010-06-06"
END = str(date.today())
CACHE = "cache"

# Databento statistics stat_type codes. VERIFY WITH --inspect BEFORE TRUSTING THEM.
STAT_SETTLEMENT = 3
STAT_CLEARED_VOLUME = 6
STAT_OPEN_INTEREST = 9

# Roots that list later than START. Populate from dbn_diagnose.py output.
ROOT_START: dict[str, str] = {}

MAX_BISECT = 4        # 1 year -> down to ~3 weeks before giving up
RETRIES = 3           # per chunk, before bisecting
PAUSE = 1.5           # seconds between chunk requests
BACKOFF = 8           # base seconds for exponential backoff on retry
DEGRADED: list[str] = []

# 503 and 502 with an empty message are throttling, not size. Bisecting a request that
# was rejected for rate reasons just multiplies the request count, which is how one
# failing year became fifteen in the log. Back off hard instead of splitting harder.


def client():
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("export DATABENTO_API_KEY='db-...' first")
    return db.Historical(key)


def classify(e: Exception) -> str:
    """
    Throttling and size failures need opposite responses, and conflating them is what
    turned one failing year into fifteen in the log.

      throttle (503, 502)  the server is refusing load. Splitting the request DOUBLES
                           the number of requests, making it worse. Wait, retry whole.
      size (504, timeout,  the payload is too big for one response. Splitting helps.
            premature end)
    """
    m = f"{type(e).__name__} {e}".lower()
    if "503" in m or "502" in m:
        return "throttle"
    if any(t in m for t in ("504", "gateway", "timed out", "timeout",
                            "prematurely", "chunked", "connection")):
        return "size"
    return "fatal"


BUDGET = {"calls": 0}
MAX_CALLS_PER_YEAR = 12   # circuit breaker: stop flailing, report, move on


def window(root: str) -> tuple[str, str]:
    return max(START, ROOT_START.get(root, START)), END


def stream(c, root: str, schema: str, start: str, end: str, depth: int = 0):
    """
    One streaming request, with retries, bisecting the date range on repeated failure.
    A range that is too large for the gateway becomes two ranges that are not.
    """
    for attempt in range(RETRIES + 1):
        if BUDGET["calls"] >= MAX_CALLS_PER_YEAR:
            raise RuntimeError(f"call budget exhausted for this year "
                               f"({MAX_CALLS_PER_YEAR}); moving on")
        BUDGET["calls"] += 1
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                df = c.timeseries.get_range(
                    dataset=DATASET, schema=schema, symbols=[f"{root}.FUT"],
                    stype_in="parent", start=start, end=end).to_df()
            for w in caught:
                if "reduced quality" in str(w.message) or "degraded" in str(w.message):
                    DEGRADED.append(f"{root} {start[:7]}")
            return df
        except Exception as e:
            kind = classify(e)
            if kind == "fatal":
                raise
            if kind == "throttle" and attempt >= RETRIES:
                # Never bisect a throttle. More requests is the opposite of the fix.
                raise RuntimeError(f"throttled after {RETRIES + 1} attempts: "
                                   f"{str(e)[:60]}")
            if attempt < RETRIES:
                wait = BACKOFF * (2 ** attempt)
                print(f"      {type(e).__name__} -> waiting {wait}s "
                      f"({attempt + 1}/{RETRIES})", flush=True)
                time.sleep(wait)
                continue
            if depth >= MAX_BISECT:
                raise
            s, en = pd.Timestamp(start), pd.Timestamp(end)
            if (en - s).days <= 2:
                raise
            mid = (s + (en - s) / 2).normalize()
            print(f"      {type(e).__name__} -> splitting "
                  f"{start[:10]}..{end[:10]}", flush=True)
            # Each half is attempted independently. The previous version called the
            # left half first and let its exception propagate, so one failing half
            # destroyed the whole year and the right half was never tried — which is
            # why every 'splitting' line in the log walks leftwards only.
            parts, failures = [], []
            for a_, b_ in ((start, str(mid.date())), (str(mid.date()), end)):
                try:
                    parts.append(stream(c, root, schema, a_, b_, depth + 1))
                except Exception as sub:
                    failures.append(f"{a_[:10]}..{b_[:10]} {type(sub).__name__}")
            if not parts:
                raise RuntimeError(f"all sub-ranges failed: {'; '.join(failures)}")
            if failures:
                print(f"      partial: recovered {len(parts)}/2, lost {failures}",
                      flush=True)
            return pd.concat(parts, ignore_index=False)
    raise RuntimeError("unreachable")


def flatten(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Databento's .to_df() returns a timestamp-indexed frame, and the index name is
    sometimes also present as a column. A bare reset_index() then raises
    "cannot insert ts_recv, already exists". Promote the index only when it adds
    something.
    """
    if raw.index.name and raw.index.name not in raw.columns:
        return raw.reset_index()
    return raw.reset_index(drop=True)


def shape_statistics(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    s = flatten(raw)
    tscol = "ts_ref" if "ts_ref" in s.columns else "ts_recv"
    s["date"] = pd.to_datetime(s[tscol], utc=True, errors="coerce") \
                  .dt.tz_localize(None).dt.normalize()
    sym = "symbol" if "symbol" in s.columns else "raw_symbol"

    def pick(stat, val, name):
        x = s.loc[s["stat_type"] == stat, ["date", sym, val]].copy()
        x.columns = ["date", "contract", name]
        # CME publishes a preliminary then a final settlement; keep the last.
        return x.groupby(["date", "contract"], as_index=False)[name].last()

    out = pick(STAT_SETTLEMENT, "price", "settle")
    for st, val, nm in ((STAT_OPEN_INTEREST, "quantity", "open_interest"),
                        (STAT_CLEARED_VOLUME, "quantity", "volume")):
        out = out.merge(pick(st, val, nm), on=["date", "contract"], how="left")
    out = out.dropna(subset=["settle"])

    # Parent symbology returns every instrument sharing the root, which on CME includes
    # calendar spreads (CLZ4-CLZ5) alongside outrights (CLZ4). This is not merely
    # wasteful. A spread has an expiry, so build_front_series can select it as the
    # nearest-expiry contract and use its settlement — a few cents — as the price of
    # crude oil. Nothing downstream would complain.
    n0 = out["contract"].nunique()
    out = out[~out["contract"].astype(str).str.contains("-", regex=False)]
    dropped = n0 - out["contract"].nunique()
    if dropped:
        print(f"      dropped {dropped:,} spread instruments, kept "
              f"{out['contract'].nunique():,} outrights", flush=True)
    return out


def fetch_year(c, root: str, year: int, lo: str, hi: str) -> pd.DataFrame:
    """One product-year of statistics, cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{root}_{year}_stats.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)

    s = max(f"{year}-01-01", lo)
    e = min(f"{year + 1}-01-01", hi)
    if s >= e:
        return pd.DataFrame()

    t = time.time()
    BUDGET["calls"] = 0
    df = shape_statistics(stream(c, root, "statistics", s, e))
    df.to_parquet(path, index=False)
    time.sleep(PAUSE)
    print(f"    {year}  {len(df):>7,} rows  {time.time()-t:5.1f}s", flush=True)
    return df


def live_expiries(c, root: str) -> pd.DataFrame:
    """
    Definitions for a short recent window only — just enough to date the contracts that
    have not expired yet. Pulling sixteen years of definitions is what makes this schema
    expensive, and we do not need it.
    """
    start = str(pd.Timestamp(END).date() - timedelta(days=90))
    try:
        d = stream(c, root, "definition", start, END)
    except Exception as e:
        print(f"    definitions unavailable ({type(e).__name__}); "
              f"live contracts will use derived expiries")
        return pd.DataFrame(columns=["contract", "expiry"])
    if d is None or d.empty:
        return pd.DataFrame(columns=["contract", "expiry"])
    d = flatten(d)
    col = "expiration" if "expiration" in d.columns else "expiration_date"
    if col not in d.columns or "raw_symbol" not in d.columns:
        return pd.DataFrame(columns=["contract", "expiry"])
    out = (d[["raw_symbol", col]].dropna().drop_duplicates("raw_symbol")
             .rename(columns={"raw_symbol": "contract", col: "expiry"}))
    out["expiry"] = pd.to_datetime(out["expiry"], utc=True, errors="coerce") \
                      .dt.tz_localize(None).dt.normalize()
    return out.dropna(subset=["expiry"])


def attach_expiries(stats: pd.DataFrame, live: pd.DataFrame,
                    data_end: pd.Timestamp) -> pd.DataFrame:
    """
    Expired contracts: expiry = last date with a settlement.
    Live contracts (last settlement within 5 days of the data end): use the definition.
    """
    last = stats.groupby("contract", as_index=False)["date"].max() \
                .rename(columns={"date": "derived"})
    m = last.merge(live, on="contract", how="left")
    still_live = m["derived"] >= (data_end - pd.Timedelta(days=5))
    m["expiry"] = m["expiry"].where(still_live & m["expiry"].notna(), m["derived"])
    unresolved = int((still_live & m["expiry"].isna()).sum())
    if unresolved:
        print(f"    {unresolved} live contracts without a definition expiry — dropped")
    return stats.merge(m[["contract", "expiry"]], on="contract", how="left")


def fetch_product(c, ct) -> pd.DataFrame:
    root = ct.fetch_root
    lo, hi = window(root)
    print(f"  {ct.symbol:4s} <- {root:3s}  {lo[:10]} to {hi[:10]}", flush=True)

    frames = []
    for year in range(int(lo[:4]), int(hi[:4]) + 1):
        try:
            frames.append(fetch_year(c, root, year, lo, hi))
        except Exception as e:
            print(f"    {year}  FAILED {type(e).__name__}: {str(e)[:60]}")
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    stats = pd.concat(frames, ignore_index=True)
    stats = stats.drop_duplicates(["date", "contract"])
    out = attach_expiries(stats, live_expiries(c, root),
                          pd.Timestamp(stats["date"].max()))
    out = out.dropna(subset=["expiry"])
    out["symbol"] = ct.symbol          # label with the TRADED contract
    print(f"    total {len(out):,} rows, {out['contract'].nunique()} contracts, "
          f"{out['date'].min():%Y-%m-%d} to {out['date'].max():%Y-%m-%d}")
    return out


def inspect(root: str) -> None:
    c = client()
    df = c.timeseries.get_range(
        dataset=DATASET, schema="statistics", symbols=[f"{root}.FUT"],
        stype_in="parent", start="2024-03-14", end="2024-03-15").to_df()
    if df.empty:
        print("Empty. Try another trading day.")
        return
    print(df.groupby("stat_type").agg(n=("price", "size"),
                                      example_price=("price", "first"),
                                      example_qty=("quantity", "first")).to_string())
    print("\ncolumns:", list(df.columns))
    print(f"\nGuesses: settlement={STAT_SETTLEMENT} volume={STAT_CLEARED_VOLUME} "
          f"open_interest={STAT_OPEN_INTEREST}")
    print("Settlement should look like a real price; volume and open interest should be")
    print("large integers in the quantity column. Edit the constants if not.")


def estimate() -> None:
    c = client()
    rows = []
    for ct in UNIVERSE:
        lo, hi = window(ct.fetch_root)
        kw = dict(dataset=DATASET, schema="statistics",
                  symbols=[f"{ct.fetch_root}.FUT"], stype_in="parent", start=lo, end=hi)
        r = dict(trade=ct.symbol, fetch=ct.fetch_root, records=None, cost=None, note="")
        for k, fn in (("records", "get_record_count"), ("cost", "get_cost")):
            try:
                r[k] = float(getattr(c.metadata, fn)(**kw))
            except Exception as e:
                r["note"] = f"{fn}: {type(e).__name__}"
        if r["records"] == 0:
            r["note"] = "ZERO RECORDS — broken, not free"
        rows.append(r)
        print(f"  {ct.symbol:4s} <- {ct.fetch_root:3s}  records={r['records']}  "
              f"${r['cost']}  {r['note']}")
    t = pd.DataFrame(rows)
    print(f"\nTOTAL ${t['cost'].fillna(0).sum():,.2f}   "
          f"records {t['records'].fillna(0).sum():,.0f}")


def run(out_path: str, only: str | None) -> None:
    c = client()
    targets = UNIVERSE
    if only:
        want = {x.strip().upper() for x in only.split(",")}
        targets = [ct for ct in UNIVERSE if ct.symbol in want or ct.fetch_root in want]
        if not targets:
            raise SystemExit(f"none of {want} in the universe")

    print(f"cache: ./{CACHE}/   (safe to Ctrl-C; completed years are not re-downloaded)\n")
    frames, failed = [], []
    for ct in targets:
        try:
            df = fetch_product(c, ct)
            if df.empty:
                failed.append(ct.symbol)
            else:
                frames.append(df)
        except KeyboardInterrupt:
            print("\ninterrupted — completed years are cached; rerun to resume")
            raise
        except Exception as e:
            print(f"    FAILED {type(e).__name__}: {str(e)[:80]}")
            failed.append(ct.symbol)

    if not frames:
        raise SystemExit("nothing fetched")

    px = (pd.concat(frames, ignore_index=True)
            [["date", "symbol", "contract", "settle", "volume", "open_interest", "expiry"]])
    px.to_parquet(out_path, index=False)
    print(f"\n-> {out_path}   {len(px):,} rows, {px['symbol'].nunique()} products")

    if failed:
        print(f"\nFAILED: {failed}")
        print(f"  retry just those:  python fetch_prices.py --run --products "
              f"{','.join(failed)}")
    if DEGRADED:
        u = sorted(set(DEGRADED))
        print(f"\nDegraded-quality periods flagged by the venue: {len(u)}")
        print(f"  {', '.join(u[:12])}{' ...' if len(u) > 12 else ''}")
        print("  Not fatal. check_data.py will show whether they left gaps.")
    print("\nThe engine excludes each name before its LISTED_FROM date, so the book runs")
    print(f"fewer than 13 contracts in the early years: {LISTED_FROM}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--symbol", default="ZC")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--products", default=None, help="subset, e.g. MCL,QG")
    ap.add_argument("--out", default="px.parquet")
    a = ap.parse_args()
    if a.inspect:
        inspect(a.symbol)
    elif a.estimate:
        estimate()
    elif a.run:
        run(a.out, a.products)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()