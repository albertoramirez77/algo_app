"""
fetch_prices_batch.py — the right tool for this job.

Databento's own docs on timeseries.get_range: "This method only returns after all of the
data has been downloaded, which can take a long time. For large requests, consider using
batch.submit_job instead." That is what fetch_prices.py was using, and it is why one
product-year took up to 1,477 seconds despite ~260 records — the cost was server-side
continuous-symbol resolution repeated on every streaming call, not bandwidth or payload.

Batch jobs are prepared server-side and handed to you as flat files. You submit once, the
job processes in the background, you download once. Streaming's per-request resolution
cost is paid one time instead of once per year per product. Databento also bills duplicate
streaming requests repeatedly; a batch job is billed once no matter how many times you
download it.

    python fetch_prices_batch.py --submit          # 13 jobs, seconds
    python fetch_prices_batch.py --status           # poll until all are 'done'
    python fetch_prices_batch.py --download --out px.parquet

Safe to run --status repeatedly; safe to close the laptop between steps. Job IDs are saved
to jobs.json so this survives a restart.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from immediacy import UNIVERSE, LISTED_FROM

DATASET = "GLBX.MDP3"
START = "2010-06-06"
JOBS_FILE = "jobs.json"
DOWNLOAD_DIR = "batch_downloads"

MODE = "continuous"   # see fetch_prices.py for the parent-vs-continuous tradeoff


def client():
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("export DATABENTO_API_KEY='db-...' first")
    return db.Historical(key)


def clamped_end(c) -> str:
    try:
        r = c.metadata.get_dataset_range(dataset=DATASET)
        return next((str(r[k])[:10] for k in ("end", "end_date") if k in r),
                    str(pd.Timestamp.today().date()))
    except Exception:
        return str(pd.Timestamp.today().date())


def load_jobs() -> dict:
    if os.path.exists(JOBS_FILE):
        return json.load(open(JOBS_FILE))
    return {}


def save_jobs(jobs: dict) -> None:
    json.dump(jobs, open(JOBS_FILE, "w"), indent=2)


def submit(products: str | None) -> None:
    c = client()
    end = clamped_end(c)
    print(f"data available through {end}\n")

    targets = UNIVERSE
    if products:
        want = {x.strip().upper() for x in products.split(",")}
        targets = [ct for ct in UNIVERSE if ct.symbol in want]

    jobs = load_jobs()
    for ct in targets:
        root = ct.fetch_root
        syms = [f"{root}.n.0"] if MODE == "continuous" else [f"{root}.FUT"]
        stype = "continuous" if MODE == "continuous" else "parent"
        try:
            job = c.batch.submit_job(
                dataset=DATASET, symbols=syms, schema="statistics",
                start=START, end=end, encoding="csv", compression="zstd",
                stype_in=stype, split_duration="none",
            )
            jobs[ct.symbol] = dict(job_id=job["id"], root=root, state=job["state"],
                                   submitted=str(pd.Timestamp.now()))
            print(f"  {ct.symbol:4s} <- {root:3s}  job {job['id']}")
        except Exception as e:
            print(f"  {ct.symbol:4s} <- {root:3s}  FAILED {type(e).__name__}: {e}")
        time.sleep(3.2)   # documented limit is 20 submit_job calls per MINUTE

    # separately: one definitions job per product, short recent window, for live contracts
    recent_start = str(pd.Timestamp(end).date() - pd.Timedelta(days=90))
    for ct in targets:
        root = ct.fetch_root
        try:
            job = c.batch.submit_job(
                dataset=DATASET, symbols=[f"{root}.FUT"], schema="definition",
                start=recent_start, end=end, encoding="csv", compression="zstd",
                stype_in="parent", split_duration="none",
            )
            jobs[f"{ct.symbol}_def"] = dict(job_id=job["id"], root=root,
                                            state=job["state"], kind="definition")
            print(f"  {ct.symbol:4s} definitions  job {job['id']}")
        except Exception as e:
            print(f"  {ct.symbol:4s} definitions  FAILED {type(e).__name__}: {e}")
        time.sleep(3.2)

    save_jobs(jobs)
    print(f"\n{len(jobs)} jobs submitted, saved to {JOBS_FILE}")
    print("Jobs process in the background. Check with --status. This is safe to close")
    print("the laptop during — the job runs on Databento's servers, not yours.")


def status() -> None:
    c = client()
    jobs = load_jobs()
    if not jobs:
        raise SystemExit(f"no jobs found; run --submit first (looked in {JOBS_FILE})")

    done, pending, failed = 0, 0, 0
    for label, info in jobs.items():
        try:
            d = c.batch.get_job_details(job_id=info["job_id"])
        except Exception as e:
            print(f"  {label:10s} could not check: {type(e).__name__}")
            failed += 1
            continue
        info["state"] = d["state"]
        state = d["state"]
        cost = f"${d['cost_usd']:.4f}" if d.get("cost_usd") is not None else "pending"
        recs = f"{d['record_count']:,}" if d.get("record_count") is not None else "-"
        print(f"  {label:10s} {state:12s} records={recs:>10s}  cost={cost}")
        if state == "done":
            done += 1
        elif state in ("received", "queued", "processing"):
            pending += 1
        else:
            failed += 1
    save_jobs(jobs)

    print(f"\n{done} done, {pending} still processing, {failed} failed/other")
    if pending:
        print("Batch jobs typically take a few minutes to at most a couple hours for a")
        print("request this size. Re-run --status to check again; no need to keep the")
        print("terminal open in between.")
    if done == len([j for j in jobs if not j.endswith("_def")]) + \
              len([j for j in jobs if j.endswith("_def")]) and pending == 0:
        print("\nAll done. Run --download next.")


FIXED_POINT = 1e9


def normalize_price(s: pd.Series, label: str = "") -> pd.Series:
    """
    Databento encodes prices as signed integers where one unit is 1e-9. Batch CSV with
    pretty_px=False (the default) hands back those raw integers, so crude arrives as
    11570000000 rather than 11.57.

    Returns are unaffected, since the scale cancels in a ratio — which is exactly why
    this is easy to miss. Position sizing is not: contracts = budget / (sigma * mult *
    price), so a price 1e9 too large drives every position to zero and the backtest
    silently produces no trades at all.

    Detected rather than assumed, so re-running with pretty_px=True does not
    double-scale.
    """
    med = float(s.median())
    if med > 1e6:
        print(f"      {label}: fixed-point prices detected (median {med:,.0f}); "
              f"scaling by 1e-9")
        return s / FIXED_POINT
    return s


UINT64_MAX = 18446744073709551615
MAX_VALID_NS = 4_102_444_800_000_000_000        # 2100-01-01, anything beyond is a sentinel


def to_session_date(s: pd.Series) -> pd.Series:
    """
    Databento marks an UNDEFINED timestamp with UINT64_MAX (2**64 - 1). pandas tries to
    read that as nanoseconds and raises OutOfBoundsDatetime.

    ts_ref is populated on 100% of settlement records but undefined on a large share of
    volume and open-interest records. Coercing silently turns those into NaT, which is how
    open interest ended up NaN on 35% of rows in the curve file without anyone noticing.
    Here the sentinel is removed explicitly and the loss is reported.
    """
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.5:
        num = num.where((num > 0) & (num < MAX_VALID_NS))
        out = pd.to_datetime(num, unit="ns", utc=True, errors="coerce")
    else:
        out = pd.to_datetime(s, utc=True, errors="coerce")
    return out.dt.tz_localize(None).dt.normalize()


def read_batch_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def download(out_path: str) -> None:
    c = client()
    jobs = load_jobs()
    if not jobs:
        raise SystemExit("no jobs found; run --submit first")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    stats_frames: dict[str, pd.DataFrame] = {}
    def_frames: dict[str, pd.DataFrame] = {}

    for label, info in jobs.items():
        d = c.batch.get_job_details(job_id=info["job_id"])
        if d["state"] != "done":
            print(f"  {label:10s} not done yet (state={d['state']}), skipping")
            continue
        job_dir = os.path.join(DOWNLOAD_DIR, info["job_id"])
        if not os.path.isdir(job_dir) or not os.listdir(job_dir):
            print(f"  {label:10s} downloading...")
            c.batch.download(job_id=info["job_id"], output_dir=DOWNLOAD_DIR)
        csvs = [f for f in os.listdir(job_dir) if f.endswith(".csv") or f.endswith(".csv.zst")]
        if not csvs:
            print(f"  {label:10s} no csv found in {job_dir}")
            continue
        df = read_batch_csv(os.path.join(job_dir, csvs[0]))
        (def_frames if label.endswith("_def") else stats_frames)[label] = df
        print(f"  {label:10s} {len(df):,} rows")

    frames = []
    for ct in UNIVERSE:
        s = stats_frames.get(ct.symbol)
        if s is None or s.empty:
            print(f"  {ct.symbol}: no statistics data, skipping")
            continue
        s = s.copy()
        # ts_ref is the SESSION the statistic refers to. ts_recv is wall-clock receipt,
        # and settlement records arrive during the Globex evening session, so ts_recv
        # stamps a large share of them on the previous calendar day — in UTC, Sunday.
        # Using it produced 16.9% Sunday rows, 305 "sessions" per year against 252, and
        # 95.3% of COT trade dates executing on a Sunday. Only ts_ref is a session key.
        tscol = next((c for c in ("ts_ref", "ts_recv", "ts_event") if c in s.columns), None)
        s["date"] = to_session_date(s[tscol])
        lost = s["date"].isna()
        if lost.any():
            by = s.loc[lost, "stat_type"].value_counts().to_dict() if "stat_type" in s else {}
            print(f"      {ct.symbol}: dropped {lost.sum():,} records with an undefined "
                  f"{tscol} (by stat_type: {by})")
            s = s[~lost]
        symcol = "symbol" if "symbol" in s.columns else "raw_symbol"

        if MODE == "continuous" and "instrument_id" in s.columns:
            s["contract"] = "id" + s["instrument_id"].astype("int64").astype(str)
        else:
            s["contract"] = s[symcol]

        def pick(stat, valcol, name):
            x = s.loc[s["stat_type"] == stat, ["date", "contract", valcol]].copy()
            x.columns = ["date", "contract", name]
            return x.groupby(["date", "contract"], as_index=False)[name].last()

        px = pick(3, "price", "settle")
        for st, val, nm in ((9, "quantity", "open_interest"), (6, "quantity", "volume")):
            px = px.merge(pick(st, val, nm), on=["date", "contract"], how="left")
        px = px.dropna(subset=["settle"])
        px["settle"] = normalize_price(px["settle"], ct.symbol)

        if MODE == "continuous":
            px["expiry"] = px["date"] + pd.Timedelta(days=400)
        else:
            px = px[~px["contract"].astype(str).str.contains("-", regex=False)]
            last = px.groupby("contract", as_index=False)["date"].max() \
                     .rename(columns={"date": "expiry"})
            defs = def_frames.get(f"{ct.symbol}_def")
            if defs is not None and not defs.empty:
                col = "expiration" if "expiration" in defs.columns else "expiration_date"
                symcol2 = "raw_symbol" if "raw_symbol" in defs.columns else "symbol"
                if col in defs.columns:
                    d2 = defs[[symcol2, col]].dropna().drop_duplicates(symcol2)
                    d2.columns = ["contract", "expiry_def"]
                    d2["expiry_def"] = pd.to_datetime(d2["expiry_def"], utc=True) \
                                          .dt.tz_localize(None).dt.normalize()
                    last = last.merge(d2, on="contract", how="left")
                    last["expiry"] = last["expiry_def"].fillna(last["expiry"])
                    last = last.drop(columns=["expiry_def"])
            px = px.merge(last[["contract", "expiry"]], on="contract", how="left")

        px["symbol"] = ct.symbol
        frames.append(px)
        print(f"  {ct.symbol}: {len(px):,} rows, {px['contract'].nunique()} contracts, "
              f"{px['date'].min():%Y-%m-%d} to {px['date'].max():%Y-%m-%d}")

    if not frames:
        raise SystemExit("nothing assembled")
    out = (pd.concat(frames, ignore_index=True)
             [["date", "symbol", "contract", "settle", "volume", "open_interest", "expiry"]])
    out.to_parquet(out_path, index=False)
    print(f"\n-> {out_path}   {len(out):,} rows, {out['symbol'].nunique()} products")
    print(f"\nEngine excludes each name before LISTED_FROM: {LISTED_FROM}")


def validate(path: str) -> None:
    """
    The check that would have caught the ts_recv defect in two lines. Run it on any price
    file before the analysis touches it.
    """
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    n = len(df)
    wk = df["date"].dt.dayofweek
    weekend = (wk >= 5).mean()
    # Per symbol, not per file: a product that lists partway through the sample has a
    # shorter span, and dividing by the file's full span understates it.
    per_yr = df.groupby("symbol")["date"].agg(
        lambda d: d.nunique() / max((d.max() - d.min()).days / 365.25, 1e-9))
    dup = df.duplicated(["date", "symbol"]).sum()

    print(f"{path}: {n:,} rows, {df['symbol'].nunique()} products, "
          f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    print(f"  weekend rows          {weekend:.2%}   (MUST be ~0%)")
    print(f"  sessions per year     {per_yr.min():.0f}-{per_yr.max():.0f}   "
          f"(MUST be ~250)")
    print(f"  duplicate date+symbol {dup:,}")
    print(f"  non-positive settle   {(df['settle'] <= 0).sum():,}")

    bad = []
    if weekend > 0.01:
        bad.append("weekend rows present -> dated from ts_recv, not ts_ref. Refetch.")
    if per_yr.max() > 265 or per_yr.min() < 235:
        bad.append("session count wrong -> the date column is not a session key.")
    if dup:
        bad.append("duplicate (date, symbol) rows -> pivot_table will silently average "
                   "them, halving those returns.")
    print("\n  " + ("PASS" if not bad else "FAIL"))
    for b in bad:
        print(f"    - {b}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--products", default=None)
    ap.add_argument("--out", default="px.parquet")
    ap.add_argument("--validate", action="store_true",
                    help="check an existing price file before the analysis uses it")
    a = ap.parse_args()
    if a.validate:
        validate(a.out)
    elif a.submit:
        submit(a.products)
    elif a.status:
        status()
    elif a.download:
        download(a.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()