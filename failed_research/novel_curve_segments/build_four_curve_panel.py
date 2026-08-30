"""Build a four-maturity daily panel from a long-form contract-level dataset.

Expected input columns:
    symbol, date, contract, expiry, settle
Optional:
    oi, volume

For each symbol/date the four contracts with the smallest expiry strictly after
that date are selected. Ordering is by expiry, never open interest.

Output columns:
    symbol,date,contract_0..3,settle_0..3,expiry_0..3
plus raw metadata used during construction where available.

No returns are fabricated across contract changes. Downstream code should
calculate contract-life returns from the contract identity columns.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REQUIRED=["symbol","date","contract","expiry","settle"]


def read_any(path):
    p=Path(path)
    if p.suffix.lower()==".parquet": return pd.read_parquet(p)
    if p.suffix.lower() in {".csv",".csv.gz"}: return pd.read_csv(p)
    raise ValueError(f"Unsupported input type: {p.suffix}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--allow-equal-date",action="store_true",help="Use expiry >= date instead of expiry > date")
    a=ap.parse_args()
    df=read_any(a.input).copy()
    miss=[c for c in REQUIRED if c not in df.columns]
    if miss: raise SystemExit(f"Missing required columns: {miss}")
    df["date"]=pd.to_datetime(df["date"]); df["expiry"]=pd.to_datetime(df["expiry"])
    df=df.replace([float("inf"),float("-inf")],pd.NA)
    df=df.dropna(subset=["symbol","date","contract","expiry","settle"])
    df=df[df["settle"]>0].copy()
    op="ge" if a.allow_equal_date else "gt"
    # One observation per symbol/date/contract; prefer highest OI if present,
    # then volume, then last input row. This only resolves duplicate quotes;
    # maturity ordering remains expiry-based.
    sort_cols=["symbol","date","contract"]
    if "oi" in df.columns: sort_cols += ["oi"]
    if "volume" in df.columns: sort_cols += ["volume"]
    df=df.sort_values(sort_cols).drop_duplicates(["symbol","date","contract"],keep="last")
    eligible=df[df.apply(lambda r: r["expiry"]>r["date"] if op=="gt" else r["expiry"]>=r["date"],axis=1)].copy()
    eligible=eligible.sort_values(["symbol","date","expiry","contract"])
    eligible["rank"] = eligible.groupby(["symbol","date"]).cumcount()
    eligible=eligible[eligible["rank"]<4].copy()
    wide=eligible.pivot(index=["symbol","date"],columns="rank",values=["contract","settle","expiry"])
    # Flatten deterministically.
    wide.columns=[f"{a}_{int(b)}" for a,b in wide.columns]
    wide=wide.reset_index()
    need=[f"{a}_{k}" for k in range(4) for a in ("contract","settle","expiry")]
    wide=wide.dropna(subset=need).sort_values(["symbol","date"]).reset_index(drop=True)
    if wide.empty: raise SystemExit("No symbol/date has four eligible contracts")
    out=Path(a.output)
    if out.suffix.lower()==".parquet": wide.to_parquet(out,index=False)
    elif out.suffix.lower()==".csv": wide.to_csv(out,index=False)
    else: raise SystemExit("Output must be .parquet or .csv")
    print(f"wrote {len(wide):,} rows to {out}")
    print(f"symbols={wide.symbol.nunique()} days={wide.date.nunique()}")

if __name__=="__main__":main()
