"""
curve_to_px.py — reuse the correctly-dated price file you already have.

    python curve_to_px.py --in px_curve.parquet --out px_clean.parquet
    python mechanism_test.py --cot cot.parquet --prices px_clean.parquet

WHY THIS EXISTS

px.parquet was dated from ts_recv — wall-clock receipt, not the session the statistic
refers to. That put 16.9% of rows on a Sunday, produced 305 "sessions" per year against a
real 252, and executed 95.3% of COT trade dates on a Sunday. Every hypothesis-1 number
rests on it.

px_curve.parquet was built from ts_ref and is verified clean: zero weekend rows, 252
sessions per year. Its settle_0 is the front contract by open-interest rank.

THE OI-ORDERING PROBLEM DOES NOT APPLY HERE. That defect broke the BASIS, which needs the
first leg to be nearer-dated than the second. A single leg's own returns are unaffected:
.n.0 is a liquid front-month series regardless of how it is ordered against .n.1.

So this converts one leg of the clean file into the schema mechanism_test.py expects, and
the mechanism question gets answered from data already on disk rather than another
download cycle.
"""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="px_curve.parquet")
    ap.add_argument("--out", default="px_clean.parquet")
    a = ap.parse_args()

    c = pd.read_parquet(a.src)
    c["date"] = pd.to_datetime(c["date"])

    out = pd.DataFrame({
        "date": c["date"],
        "symbol": c["symbol"],
        "contract": c["contract_0"],
        "settle": c["settle_0"],
        "volume": c.get("vol_0"),
        "open_interest": c.get("oi_0"),
        # expiry_0 is the front contract's real expiration where the definition job landed.
        # Where it is missing, a far-future sentinel keeps build_front_series from
        # filtering the row out: with one contract per date its nearest-expiry selection
        # is a no-op anyway, and roll detection keys on the contract identifier changing.
        "expiry": c["expiry_0"].fillna(c["date"] + pd.Timedelta(days=400))
        if "expiry_0" in c.columns else c["date"] + pd.Timedelta(days=400),
    }).dropna(subset=["settle"])

    # Same tie-break as test_curve.load(): na_position="first" so the leg WITH open
    # interest wins. pandas sorts NaN last by default, and oi_0 is missing wherever the
    # open-interest record carried no session key, so the default would keep the row
    # without it — on roll dates, which is where it matters most.
    out = (out.sort_values(["symbol", "date", "open_interest"], na_position="first")
              .drop_duplicates(["date", "symbol"], keep="last")
              .sort_values(["symbol", "date"])
              .reset_index(drop=True))

    wk = out["date"].dt.dayofweek
    # Span must be measured PER SYMBOL. Dividing every product's session count by the
    # whole file's span understates any product with a shorter history — KE lists in
    # 2013, so it looked like 204 sessions a year instead of its real ~252.
    per = out.groupby("symbol")["date"].agg(
        lambda d: d.nunique() / max((d.max() - d.min()).days / 365.25, 1e-9))
    print(f"{a.src} -> {a.out}")
    print(f"  {len(out):,} rows, {out['symbol'].nunique()} products, "
          f"{out['date'].min():%Y-%m-%d} to {out['date'].max():%Y-%m-%d}")
    print(f"  weekend rows      {(wk >= 5).mean():.2%}   (must be ~0%)")
    print(f"  sessions per year {per.min():.0f}-{per.max():.0f}   (must be ~250)")
    print("   ", "  ".join(f"{s}:{v:.0f}" for s, v in per.items()))
    print(f"  missing volume    {out['volume'].isna().mean():.1%}")
    print(f"  missing open int  {out['open_interest'].isna().mean():.1%}")

    bad = []
    if (wk >= 5).mean() > 0.01:
        bad.append("weekend rows -> this file is not ts_ref-dated either")
    if per.max() > 265 or per.min() < 235:
        bad.append("session count wrong -> the date column is not a session key")
    print("  " + ("PASS" if not bad else "FAIL"))
    for b in bad:
        print(f"    - {b}")
    if bad:
        raise SystemExit("not written")

    out.to_parquet(a.out, index=False)
    print(f"\nNow run:\n  python mechanism_test.py --cot cot.parquet --prices {a.out}")
    print("\nMissing volume and open interest are expected: ts_ref is undefined on many")
    print("volume and open-interest records. That degrades only the impact term in the")
    print("cost model, which the engine already falls back on. It does not touch returns,")
    print("and the mechanism test does not use either field.")


if __name__ == "__main__":
    main()