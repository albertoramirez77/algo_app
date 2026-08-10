"""
fetch_cot.py — build cot.parquet from CFTC Disaggregated Commitments of Traders files.

Handles .xls, .xlsx, .txt and .csv. The CFTC ships the same data in several formats, and
every annual archive contains a file with the SAME internal name (f_year.*), so unzipping
several years into one folder makes your operating system rename them f_year.xls,
f_year_1.xls, f_year_2.xls and so on. That is expected and this script handles it.

    python fetch_cot.py --coverage --raw-dir ./cot_raw
        Which years do you actually have? Run this first.

    python fetch_cot.py --list-markets --raw-dir ./cot_raw --search CORN
        Contracts ranked by open interest, with CFTC codes.

    python fetch_cot.py --raw-dir ./cot_raw --out cot.parquet
        Build the file the backtest wants.

The first read of a 26 MB .xls takes about a minute. Each file is cached as parquet
alongside the original, so later runs are instant.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

READABLE = ("txt", "csv", "xls", "xlsx")

# Field -> candidate column names after normalisation. The CFTC has used several
# spellings over the years; known variants are all listed.
WANT = {
    "market":     ["market_and_exchange_names", "market_and_exchange_name"],
    "code":       ["cftc_contract_market_code"],
    "oi":         ["open_interest_all"],
    "prod_long":  ["prod_merc_positions_long_all"],
    "prod_short": ["prod_merc_positions_short_all"],
    "mm_long":    ["m_money_positions_long_all"],
    "mm_short":   ["m_money_positions_short_all"],
}
DATE_COLS = ["as_of_date_in_form_yymmdd", "report_date_as_mm_dd_yyyy",
             "report_date_as_yyyy_mm_dd", "as_of_date_in_form_yyyy_mm_dd"]


def norm(s) -> str:
    out, prev = [], False
    for ch in str(s).strip().lower():
        if ch.isalnum():
            out.append(ch); prev = False
        elif not prev:
            out.append("_"); prev = True
    return "".join(out).strip("_")


def parse_dates(s: pd.Series) -> pd.Series:
    """
    CFTC date columns arrive in three encodings depending on the file format:

        Excel serial float   46231.0      (what any reader gives you from .xls)
        YYMMDD integer       260728       (the As_of_Date column)
        string               07/28/2026   or   2026-07-28

    Excel serials for 2006-2030 fall in 38000-47500; YYMMDD values for the same period
    fall in 060101-301231. The ranges do not overlap, so they can be separated safely.
    """
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().mean() > 0.9:
        med = float(v.median())
        if 20000 < med < 80000:                                  # Excel serial
            return pd.to_datetime(v, unit="D", origin="1899-12-30", errors="coerce")
        if 100000 <= med <= 999999:                              # YYMMDD
            iv = v.round().astype("Int64").astype(str).str.zfill(6)
            return pd.to_datetime(iv, format="%y%m%d", errors="coerce")
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        out = pd.to_datetime(s, format=fmt, errors="coerce")
        if out.notna().mean() > 0.9:
            return out
    return pd.to_datetime(s, errors="coerce", format="mixed")


def read_one(path: str) -> pd.DataFrame:
    """Read one raw file, caching the parsed subset as parquet beside it."""
    cache = os.path.splitext(path)[0] + ".cached.parquet"
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        return pd.read_parquet(cache)

    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("xls", "xlsx"):
        df = pd.read_excel(path, engine="xlrd" if ext == "xls" else "openpyxl")
    else:
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")

    df.columns = [norm(c) for c in df.columns]
    wanted = set(DATE_COLS) | {n for v in WANT.values() for n in v}
    df = df[[c for c in df.columns if c in wanted]].copy()
    try:
        df.to_parquet(cache, index=False)
    except Exception:
        pass
    return df


def read_all(raw_dir: str) -> pd.DataFrame:
    files = sorted({f for ext in READABLE
                    for f in glob.glob(os.path.join(raw_dir, "**", f"*.{ext}"),
                                       recursive=True)
                    if ".cached." not in f})
    if not files:
        present = sorted({os.path.splitext(f)[1] or "(no extension)" for f in
                          glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)
                          if os.path.isfile(f)})
        raise SystemExit(
            f"No readable data files under {raw_dir}\n"
            f"Extensions actually present: {present or 'none - the folder is empty'}\n\n"
            f"This script reads {READABLE}. If you have .zip files, unzip them first.\n"
            f"If the folder looks empty, check the path:   ls {raw_dir}"
        )
    frames = []
    for f in files:
        d = read_one(f)
        frames.append(d)
        print(f"  {os.path.basename(f):34s} {len(d):>8,} rows")
    return pd.concat(frames, ignore_index=True)


def resolve(df: pd.DataFrame) -> dict:
    col, missing = {}, []
    for key, cands in WANT.items():
        hit = next((c for c in cands if c in df.columns), None)
        col[key] = hit
        if hit is None:
            missing.append(key)
    hit = next((c for c in DATE_COLS if c in df.columns), None)
    col["date"] = hit
    if hit is None:
        missing.append("date")
    if missing:
        print("\nCOLUMNS FOUND IN YOUR FILES:")
        for c in sorted(df.columns):
            print("   ", c)
        raise SystemExit(f"\nCould not locate: {missing}. Add the real names to WANT.")
    return col


def coverage(df: pd.DataFrame, col: dict) -> None:
    d = parse_dates(df[col["date"]])
    t = pd.DataFrame({"year": d.dt.year, "date": d}).dropna()
    if t.empty:
        raise SystemExit("No dates parsed. Inspect the date column manually.")
    g = t.groupby("year").agg(weeks=("date", "nunique"),
                              first=("date", "min"), last=("date", "max"))
    print(g.to_string())
    have = set(int(y) for y in g.index)
    print(f"\n  years present: {len(have)}   ({min(have)}–{max(have)})")
    gaps = sorted(set(range(2010, max(have) + 1)) - have)
    if gaps:
        print(f"  MISSING YEARS: {gaps}")
        print("  Download the Disaggregated Futures Only archive for each.")
    else:
        print("  No gaps between 2010 and the latest year present.")
    partial = g[(g["weeks"] < 45) & (g.index < max(have))]
    if len(partial):
        print(f"\n  Years with under 45 weeks (partial download?): {list(partial.index)}")


def list_markets(df, col, search, top) -> None:
    t = df[[col["code"], col["market"], col["oi"]]].copy()
    t.columns = ["cftc_code", "market", "oi"]
    t["oi"] = pd.to_numeric(t["oi"], errors="coerce").fillna(0)
    t["cftc_code"] = t["cftc_code"].astype(str).str.strip()
    g = (t.groupby(["cftc_code", "market"], as_index=False)["oi"].max()
           .sort_values("oi", ascending=False))
    if search:
        g = g[g["market"].str.upper().str.contains(search.upper(), na=False)]
    pd.set_option("display.max_rows", None, "display.width", 200,
                  "display.max_colwidth", 62)
    print(g.head(top).to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    print(f"\n  showing {min(top, len(g))} of {len(g)}, ranked by peak open interest")
    print("  Match by NAME, copy the CODE into UNIVERSE in immediacy.py.")
    print("  Names change over time. Codes are the stable key.")


def build(df, col, codes) -> pd.DataFrame:
    df = df.copy()
    df["_code"] = df[col["code"]].astype(str).str.strip()
    keep = df[df["_code"].isin(codes)].copy()
    if keep.empty:
        raise SystemExit("No rows matched your codes. Run --list-markets and fix UNIVERSE.")
    out = pd.DataFrame({
        "report_date": parse_dates(keep[col["date"]]),
        "symbol": keep["_code"].map(codes),
        "prod_long": pd.to_numeric(keep[col["prod_long"]], errors="coerce"),
        "prod_short": pd.to_numeric(keep[col["prod_short"]], errors="coerce"),
        "open_interest": pd.to_numeric(keep[col["oi"]], errors="coerce"),
        "mm_long": pd.to_numeric(keep[col["mm_long"]], errors="coerce"),
        "mm_short": pd.to_numeric(keep[col["mm_short"]], errors="coerce"),
    }).dropna(subset=["report_date", "prod_long", "prod_short", "open_interest"])
    out = out[out["open_interest"] > 0]
    return (out.drop_duplicates(["report_date", "symbol"])
               .sort_values(["symbol", "report_date"]).reset_index(drop=True))


def validate(out: pd.DataFrame, universe) -> None:
    print(out.groupby("symbol").agg(weeks=("report_date", "size"),
                                    first=("report_date", "min"),
                                    last=("report_date", "max"),
                                    med_oi=("open_interest", "median")).to_string())
    miss = sorted({c.symbol for c in universe} - set(out["symbol"]))
    print()
    if miss:
        print(f"  MISSING: {miss}  -> wrong codes in UNIVERSE. Fix before continuing.")
    tue = (out["report_date"].dt.dayofweek == 1).mean()
    print(f"  report dates on a Tuesday: {tue:.1%}   (expect >90%)")
    hp = ((out["prod_short"] - out["prod_long"]) / out["open_interest"]).mean()
    print(f"  mean hedging pressure: {hp:+.4f}   (MUST be positive; published ~+0.14)")
    if hp < 0:
        print("  ^^ NEGATIVE: prod_long/prod_short swapped, or wrong trader category.")
    yr = (out.assign(y=out["report_date"].dt.year)
             .groupby("symbol")["y"].agg(["min", "max", "nunique"]))
    broken = yr[yr["nunique"] < (yr["max"] - yr["min"] + 1)]
    if len(broken):
        print("\n  CONTRACTS MISSING YEARS IN THE MIDDLE (the code changed at some point):")
        print(broken.to_string())


def validate_output(path: str) -> int:
    """
    Check an assembled cot.parquet against what immediacy.py assumes, BEFORE it is
    analysed. Returns the number of failures so the shell can gate on it.
    """
    from immediacy import UNIVERSE, SHUTDOWN_START, SHUTDOWN_CLEARED

    df = pd.read_parquet(path)
    fails, warns = [], []

    def bad(msg): fails.append(msg)
    def warn(msg): warns.append(msg)

    print(f"validating {path}   {len(df):,} rows, {df['symbol'].nunique()} products")

    need = ["report_date", "symbol", "prod_long", "prod_short", "open_interest"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"  FAIL  missing required columns: {missing}")
        return 1

    rd = pd.to_datetime(df["report_date"])

    if df[need].isna().any().any():
        bad(f"NaN present in required columns: "
            f"{df[need].isna().sum()[lambda s: s > 0].to_dict()}")

    # --- the divisor behind Q. A zero here makes one name's Q infinite, which turns
    # EVERY name's cross-sectional z that week into NaN.
    n_zero = int((df["open_interest"] <= 0).sum())
    if n_zero:
        bad(f"{n_zero:,} rows with open_interest <= 0 — Q divides by the lagged value, "
            f"so one zero silently zeroes the whole week's cross-section")

    for c in ("prod_long", "prod_short"):
        over = int((df[c] > df["open_interest"]).sum())
        if over:
            warn(f"{over:,} rows where {c} exceeds open interest")

    # --- the measurement cadence
    tue = float((rd.dt.dayofweek == 1).mean())
    if tue < 0.90:
        bad(f"only {tue:.1%} of report dates fall on a Tuesday (expect >90%)")

    dup = int(df.duplicated(["report_date", "symbol"]).sum())
    if dup:
        bad(f"{dup:,} duplicate (report_date, symbol) rows")

    # --- the sign that says the trader category is the right one
    hp = float(((df["prod_short"] - df["prod_long"]) / df["open_interest"]).mean())
    if hp <= 0:
        bad(f"mean hedging pressure is {hp:+.4f} — must be positive (published ~+0.14). "
            f"prod_long/prod_short are swapped, or this is the wrong category")

    # --- the universe the engine expects
    absent = sorted({c.symbol for c in UNIVERSE} - set(df["symbol"]))
    if absent:
        bad(f"no COT data for {absent} — wrong cftc_code in UNIVERSE")

    # --- gaps. A code change mid-sample leaves a hole that no summary statistic shows.
    for sym, g in df.groupby("symbol"):
        w = pd.to_datetime(g["report_date"]).sort_values().reset_index(drop=True)
        gaps = w.diff().dt.days.dropna()
        long_gaps = gaps[gaps > 21]
        if len(long_gaps):
            worst = w[int(long_gaps.idxmax())]
            warn(f"{sym}: {len(long_gaps)} gap(s) over 21 days, longest "
                 f"{int(long_gaps.max())} days ending {worst:%Y-%m-%d}")

    # --- the 2025 shutdown. Present as a fact, not a failure: the archives back-filled
    # these at their original measurement dates and the engine holds them.
    held = int(rd.between(SHUTDOWN_START, SHUTDOWN_CLEARED).sum())
    if held:
        print(f"  note  {held:,} rows measured inside the 2025 shutdown window "
              f"({SHUTDOWN_START:%Y-%m-%d} to {SHUTDOWN_CLEARED:%Y-%m-%d}); the engine "
              f"holds them until the backlog cleared")

    # --- the dtype the join keys are compared on
    if str(df["report_date"].dtype) != "datetime64[ns]":
        warn(f"report_date dtype is {df['report_date'].dtype}, not datetime64[ns]; "
             f"price dates are ns, so every merge relies on pandas upcasting")

    for msg in warns: print(f"  warn  {msg}")
    for msg in fails: print(f"  FAIL  {msg}")
    if not fails and not warns:
        print("  all checks passed")
    elif not fails:
        print(f"  {len(warns)} warning(s), no failures")
    else:
        print(f"\n  {len(fails)} FAILURE(S). Do not analyse this file until they are "
              f"explained.")
    return len(fails)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="./cot_raw")
    ap.add_argument("--out", default="cot.parquet")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--list-markets", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="check --out against the schema immediacy.py expects")
    ap.add_argument("--search", default=None)
    ap.add_argument("--top", type=int, default=60)
    a = ap.parse_args()

    if a.validate:
        raise SystemExit(1 if validate_output(a.out) else 0)

    print(f"Reading {a.raw_dir}  (first pass on .xls is slow; results are cached)")
    df = read_all(a.raw_dir)
    col = resolve(df)
    print(f"\ncolumn map: {col}\n")

    if a.coverage:
        coverage(df, col); return
    if a.list_markets:
        list_markets(df, col, a.search, a.top); return

    from immediacy import UNIVERSE
    out = build(df, col, {c.cftc_code: c.symbol for c in UNIVERSE})
    validate(out, UNIVERSE)
    out.to_parquet(a.out, index=False)
    print(f"\n-> {a.out}   ({len(out):,} rows)")


if __name__ == "__main__":
    main()