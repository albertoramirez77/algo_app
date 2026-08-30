"""
fetch_curve.py — second-nearby prices and front-contract expiries.

    python fetch_curve.py --submit
    python fetch_curve.py --status
    python fetch_curve.py --download --out px_curve.parquet

Writes a NEW file. Your existing px.parquet is not touched.

WHAT THIS PULLS AND WHY

  statistics on ROOT.n.0    front contract settlement, open interest, volume
  statistics on ROOT.n.1    second contract settlement  -> the basis
  definition on ROOT.n.0    the front contract's expiration date, one record per day

That third job is the trick. Requesting definitions under parent symbology means every
instrument sharing the root — thousands, including spreads — over sixteen years. Requesting
them under CONTINUOUS symbology returns one definition per day, for whichever contract is
currently front. Tiny, and it is exactly what we need: days-to-expiry for the contract we
actually hold.

Three jobs per product, thirteen products, thirty-nine jobs. Submission is rate-limited to
one call every 3.2 seconds against the documented twenty-per-minute ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from immediacy import UNIVERSE as NARROW_UNIVERSE

# --universe wide switches to the 35-instrument cross-asset set. The narrow 13 remain
# available so results can be compared like for like.
try:
    from universe import UNIVERSE as WIDE_UNIVERSE
except ImportError:
    WIDE_UNIVERSE = None
UNIVERSE = NARROW_UNIVERSE

DATASET = "GLBX.MDP3"
START = "2010-06-06"
ROLL = "c"                 # "c" = calendar (ordered by EXPIRY), "n" = open-interest rank
JOBS_FILE = "jobs_curve.json"
DOWNLOAD_DIR = "batch_curve"

# WHY CALENDAR ROLL AND NOT OPEN-INTEREST RANK
#
# .n.0 is the contract with the HIGHEST OPEN INTEREST and .n.1 the second highest. That
# ordering has nothing to do with maturity: for grains, open interest sits in the benchmark
# months rather than the nearby, and .n.1 can expire BEFORE .n.0. A basis computed as
# ln(F0/F1) then carries a randomly flipping sign, which attenuates and can invert any
# slope estimated from it.
#
# Measured on the .n data: median 65 days to "front" expiry, p95 of 132, and 1,129 days on
# which both legs resolved to the same instrument. Those are integrity failures, not
# market facts.
#
# .c.0 and .c.1 are ordered by expiry. Definitions are pulled for BOTH legs so the basis
# can be annualised by the real maturity gap, which is what the literature does.
FIXED_POINT = 1e9

STAT_SETTLEMENT, STAT_VOLUME, STAT_OI = 3, 6, 9


def client():
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("export DATABENTO_API_KEY='db-...' first")
    return db.Historical(key)


def avail_end(c) -> str:
    try:
        r = c.metadata.get_dataset_range(dataset=DATASET)
        return next(str(r[k])[:10] for k in ("end", "end_date") if k in r)
    except Exception:
        return str(pd.Timestamp.today().date())


def jobs_io(jobs=None):
    if jobs is None:
        return json.load(open(JOBS_FILE)) if os.path.exists(JOBS_FILE) else {}
    json.dump(jobs, open(JOBS_FILE, "w"), indent=2)


def submit(products: str | None) -> None:
    c = client(); end = avail_end(c)
    print(f"data available through {end}\n")
    targets = UNIVERSE
    if products:
        want = {x.strip().upper() for x in products.split(",")}
        targets = [t for t in UNIVERSE if t.symbol in want]

    jobs = jobs_io()
    plan = []
    for ct in targets:
        for leg in ("0", "1"):
            root = getattr(ct, "fetch_root", None) or getattr(ct, "root")
            sym = f"{root}.{ROLL}.{leg}"
            plan.append((f"{ct.symbol}|s{leg}", sym, "statistics"))
            plan.append((f"{ct.symbol}|d{leg}", sym, "definition"))

    for label, sym, schema in plan:
        if label in jobs:
            print(f"  {label:12s} already submitted ({jobs[label]['job_id']})")
            continue
        try:
            j = c.batch.submit_job(
                dataset=DATASET, symbols=[sym], schema=schema, start=START, end=end,
                encoding="csv", compression="zstd", stype_in="continuous",
                split_duration="none")
            jobs[label] = dict(job_id=j["id"], symbol=sym, schema=schema)
            print(f"  {label:12s} {sym:10s} {schema:11s} -> {j['id']}")
        except Exception as e:
            print(f"  {label:12s} {sym:10s} FAILED {type(e).__name__}: {str(e)[:70]}")
        time.sleep(3.2)
    jobs_io(jobs)
    print(f"\n{len(jobs)} jobs recorded in {JOBS_FILE}. Check with --status.")


def status() -> None:
    c = client(); jobs = jobs_io()
    if not jobs:
        raise SystemExit("no jobs; run --submit first")
    done = pend = fail = 0
    for label, info in sorted(jobs.items()):
        try:
            d = c.batch.get_job_details(job_id=info["job_id"])
        except Exception as e:
            print(f"  {label:12s} check failed {type(e).__name__}"); fail += 1; continue
        st = d["state"]
        recs = f"{d['record_count']:,}" if d.get("record_count") is not None else "-"
        cost = f"${d['cost_usd']:.4f}" if d.get("cost_usd") is not None else "-"
        print(f"  {label:12s} {st:12s} records={recs:>12s}  {cost}")
        done += st == "done"; pend += st in ("received", "queued", "processing")
        fail += st not in ("done", "received", "queued", "processing")
    print(f"\n{done} done, {pend} processing, {fail} other")
    if pend == 0 and done:
        print("All finished. Run --download.")


def read_job(c, info) -> pd.DataFrame:
    d = os.path.join(DOWNLOAD_DIR, info["job_id"])
    if not os.path.isdir(d) or not os.listdir(d):
        c.batch.download(job_id=info["job_id"], output_dir=DOWNLOAD_DIR)
    files = [f for f in os.listdir(d) if ".csv" in f]
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(os.path.join(d, files[0]))
    df.columns = [x.strip().lower() for x in df.columns]
    return df


def to_date(s) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()


def shape_stats(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ts = next((c for c in ("ts_ref", "ts_recv", "ts_event") if c in df.columns), None)
    df = df.copy(); df["date"] = to_date(df[ts])
    df["contract"] = "id" + df["instrument_id"].astype("int64").astype(str)

    def pick(stat, col, name):
        x = df.loc[df["stat_type"] == stat, ["date", "contract", col]].copy()
        x.columns = ["date", "contract", name]
        return x.groupby(["date", "contract"], as_index=False)[name].last()

    out = pick(STAT_SETTLEMENT, "price", f"settle_{tag}")
    out = out.rename(columns={"contract": f"contract_{tag}"})
    if out[f"settle_{tag}"].median() > 1e6:
        out[f"settle_{tag}"] /= FIXED_POINT
    if tag == "0":
        for st, col, nm in ((STAT_OI, "quantity", "oi_0"), (STAT_VOLUME, "quantity", "vol_0")):
            p = pick(st, col, nm).rename(columns={"contract": "contract_0"})
            out = out.merge(p, on=["date", "contract_0"], how="left")
    return out


def download(out_path: str) -> None:
    c = client(); jobs = jobs_io()
    parts: dict[str, dict[str, pd.DataFrame]] = {}
    for label, info in sorted(jobs.items()):
        d = c.batch.get_job_details(job_id=info["job_id"])
        if d["state"] != "done":
            print(f"  {label:12s} not done ({d['state']}), skipping"); continue
        sym, kind = label.split("|")
        parts.setdefault(sym, {})[kind] = read_job(c, info)
        print(f"  {label:12s} {len(parts[sym][kind]):,} rows")

    frames = []
    for ct in UNIVERSE:
        p = parts.get(ct.symbol, {})
        if "s0" not in p or "s1" not in p:
            print(f"  {ct.symbol}: missing a statistics leg, skipping"); continue
        a = shape_stats(p["s0"], "0"); b = shape_stats(p["s1"], "1")
        if a.empty or b.empty:
            print(f"  {ct.symbol}: empty after shaping, skipping"); continue
        w = a.merge(b, on="date", how="inner")

        for leg in ("0", "1"):
            dfn = p.get(f"d{leg}"); col = f"expiry_{leg}"
            if dfn is not None and not dfn.empty:
                ts = next((x for x in ("ts_recv", "ts_event") if x in dfn.columns), None)
                ecol = "expiration" if "expiration" in dfn.columns else "expiration_date"
                if ts and ecol in dfn.columns:
                    e = dfn[[ts, ecol]].copy(); e.columns = ["date", col]
                    e["date"] = to_date(e["date"]); e[col] = to_date(e[col])
                    e = e.dropna().groupby("date", as_index=False)[col].last()
                    w = w.merge(e, on="date", how="left")
            if col not in w.columns:
                w[col] = pd.NaT

        w["symbol"] = ct.symbol
        frames.append(w)
        both = w["expiry_0"].notna() & w["expiry_1"].notna()
        inv = (w.loc[both, "expiry_1"] <= w.loc[both, "expiry_0"]).mean() if both.any() else np.nan
        gap = (w.loc[both, "expiry_1"] - w.loc[both, "expiry_0"]).dt.days.median() if both.any() else np.nan
        dte = (w.loc[both, "expiry_0"] - w.loc[both, "date"]).dt.days.median() if both.any() else np.nan
        print(f"  {ct.symbol}: {len(w):,} rows  {w['date'].min():%Y-%m-%d} to "
              f"{w['date'].max():%Y-%m-%d}  exp cov {both.mean():.0%}  "
              f"med dte {dte:.0f}d  med gap {gap:.0f}d  INVERTED {inv:.1%}")

    if not frames:
        raise SystemExit("nothing assembled")
    px = pd.concat(frames, ignore_index=True)
    px.to_parquet(out_path, index=False)
    print(f"\n-> {out_path}   {len(px):,} rows, {px['symbol'].nunique()} products")
    print("\ncolumns:", list(px.columns))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--products", default=None)
    ap.add_argument("--out", default="px_curve_c.parquet")
    ap.add_argument("--roll", choices=["c", "n"], default="c")
    ap.add_argument("--universe", choices=["narrow", "wide"], default="narrow")
    a = ap.parse_args()
    global ROLL, JOBS_FILE, DOWNLOAD_DIR, UNIVERSE
    if a.universe == "wide":
        if WIDE_UNIVERSE is None:
            raise SystemExit("universe.py not found — put it beside this script")
        UNIVERSE = WIDE_UNIVERSE
    print(f"universe: {a.universe} ({len(UNIVERSE)} instruments)")
    ROLL = a.roll
    JOBS_FILE = f"jobs_curve_{ROLL}_{a.universe}.json"
    DOWNLOAD_DIR = f"batch_curve_{ROLL}_{a.universe}"
    print(f"roll: {ROLL} "
          f"({'calendar, ordered by expiry' if ROLL == 'c' else 'open-interest rank'})")
    if a.submit: submit(a.products)
    elif a.status: status()
    elif a.download: download(a.out)
    else: ap.print_help()

if __name__ == "__main__":
    main()