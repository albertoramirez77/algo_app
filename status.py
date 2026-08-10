"""
status.py — where am I, and what do I run next?

    python status.py

Checks every artifact the backtest needs, reports what is present, what is missing, and
prints the single next command to run. Safe to run any time. Touches no API and costs
nothing.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

OK, WARN, BAD = "[ OK ]", "[WARN]", "[MISS]"


def hdr(t: str) -> None:
    print(f"\n{'-' * 72}\n{t}\n{'-' * 72}")


def check_prices() -> tuple[bool, list[str]]:
    hdr("1. PRICE DATA  (px.parquet)")
    issues = []
    if not os.path.exists("px.parquet"):
        print(f"  {BAD} px.parquet not found")
        return False, ["run: python fetch_prices_batch.py --download --out px.parquet"]

    px = pd.read_parquet("px.parquet")
    px["date"] = pd.to_datetime(px["date"])
    n_prod = px["symbol"].nunique()
    print(f"  {OK if n_prod == 13 else WARN} {len(px):,} rows, {n_prod} of 13 products")
    print(f"       {px['date'].min():%Y-%m-%d} to {px['date'].max():%Y-%m-%d}")

    if n_prod < 13:
        from immediacy import UNIVERSE
        missing = sorted({c.symbol for c in UNIVERSE} - set(px["symbol"]))
        print(f"  {WARN} missing products: {missing}")
        issues.append(f"re-run those: python fetch_prices_batch.py --submit --products "
                      f"{','.join(missing)}")

    per = px.groupby("symbol").agg(rows=("date", "size"),
                                   contracts=("contract", "nunique"),
                                   first=("date", "min"), last=("date", "max"))
    print()
    print(per.to_string())

    # sanity: settlements must be positive and finite
    bad_px = px[(px["settle"] <= 0) | (~px["settle"].notna())]
    if len(bad_px):
        print(f"\n  {WARN} {len(bad_px):,} rows with non-positive or missing settlement")
        issues.append("investigate zero/negative settlements before trusting results")
    else:
        print(f"\n  {OK} all settlements positive and present")

    dup = px.duplicated(["date", "symbol"]).sum()
    if dup:
        print(f"  {WARN} {dup:,} duplicate (date, symbol) rows")
        issues.append("de-duplicate px.parquet")
    else:
        print(f"  {OK} no duplicate (date, symbol) rows")

    # expected ~250 rows/year/product in continuous mode
    yrs = (px["date"].max() - px["date"].min()).days / 365.25
    exp = 250 * yrs
    thin = per[per["rows"] < exp * 0.5]
    if len(thin):
        print(f"  {WARN} products with under half the expected ~{exp:,.0f} rows: "
              f"{list(thin.index)}")
        issues.append("those products have gaps; consider re-submitting them")

    return n_prod == 13 and not issues, issues


def check_cot() -> tuple[bool, list[str]]:
    hdr("2. POSITION DATA  (cot.parquet)")
    issues = []
    raw = [f for ext in ("txt", "csv", "xls", "xlsx")
           for f in glob.glob(f"cot_raw/**/*.{ext}", recursive=True)
           if ".cached." not in f]
    print(f"  raw files in ./cot_raw/: {len(raw)}")

    if not os.path.exists("cot.parquet"):
        print(f"  {BAD} cot.parquet not found")
        if not raw:
            return False, [
                "Download the CFTC Disaggregated Futures Only annual archives for "
                "2010-2026 from cftc.gov -> Market Reports -> Commitments of Traders "
                "-> Historical Compressed, unzip them all into ./cot_raw/, then run: "
                "python fetch_cot.py --raw-dir ./cot_raw --out cot.parquet"]
        return False, ["run: python fetch_cot.py --raw-dir ./cot_raw --out cot.parquet"]

    cot = pd.read_parquet("cot.parquet")
    cot["report_date"] = pd.to_datetime(cot["report_date"])
    n_sym = cot["symbol"].nunique()
    years = sorted(cot["report_date"].dt.year.unique())
    print(f"  {OK if n_sym == 13 else WARN} {len(cot):,} rows, {n_sym} of 13 contracts")
    print(f"       years present: {years[0]}-{years[-1]} ({len(years)} of 17)")

    gaps = sorted(set(range(2010, max(years) + 1)) - set(years))
    if gaps:
        print(f"  {WARN} MISSING YEARS: {gaps}")
        issues.append(f"download CFTC archives for {gaps}, unzip into ./cot_raw/, "
                      f"re-run fetch_cot.py")

    hp = ((cot["prod_short"] - cot["prod_long"]) / cot["open_interest"]).mean()
    if hp > 0:
        print(f"  {OK} mean hedging pressure {hp:+.4f} (positive, as required)")
    else:
        print(f"  {BAD} mean hedging pressure {hp:+.4f} — NEGATIVE")
        issues.append("prod_long/prod_short are swapped, or the wrong trader category "
                      "was read. Every position would be backwards. Fix before running.")

    tue = (cot["report_date"].dt.dayofweek == 1).mean()
    print(f"  {OK if tue > 0.9 else WARN} {tue:.0%} of report dates fall on a Tuesday")

    return n_sym == 13 and not issues, issues


def check_alignment() -> list[str]:
    hdr("3. DO THE TWO DATASETS OVERLAP?")
    if not (os.path.exists("px.parquet") and os.path.exists("cot.parquet")):
        print("  skipped — need both files first")
        return []
    px = pd.read_parquet("px.parquet"); cot = pd.read_parquet("cot.parquet")
    px["date"] = pd.to_datetime(px["date"])
    cot["report_date"] = pd.to_datetime(cot["report_date"])
    lo = max(px["date"].min(), cot["report_date"].min())
    hi = min(px["date"].max(), cot["report_date"].max())
    yrs = (hi - lo).days / 365.25
    print(f"  usable backtest window: {lo:%Y-%m-%d} to {hi:%Y-%m-%d}  ({yrs:.1f} years)")

    shared = set(px["symbol"]) & set(cot["symbol"])
    print(f"  contracts present in BOTH: {len(shared)} of 13")
    if len(shared) < 13:
        print(f"    only in prices: {sorted(set(px['symbol']) - shared)}")
        print(f"    only in COT   : {sorted(set(cot['symbol']) - shared)}")

    out = []
    if yrs < 8:
        out.append(f"only {yrs:.1f} years of overlap — the stress gate needs a 260-week "
                   f"warm-up, so the effective backtest is ~5 years shorter than this")
    return out


def main() -> None:
    print("=" * 72)
    print("PROJECT STATUS")
    print("=" * 72)
    px_ok, px_issues = check_prices()
    cot_ok, cot_issues = check_cot()
    align_issues = check_alignment()

    hdr("NEXT STEP")
    issues = px_issues + cot_issues + align_issues
    if px_ok and cot_ok and not issues:
        print("  Everything is in place. Run the six data checks:\n")
        print("      python check_data.py --cot cot.parquet --prices px.parquet\n")
        print("  Then, if those pass, the in-sample backtest:\n")
        print("      python immediacy.py --run --cot cot.parquet --prices px.parquet\n")
        print("  Do NOT pass --oos-unlock yet. The last 25% of the sample stays sealed")
        print("  until every parameter decision is made.")
    else:
        for i, s in enumerate(issues, 1):
            print(f"  {i}. {s}")
    print()


if __name__ == "__main__":
    main()