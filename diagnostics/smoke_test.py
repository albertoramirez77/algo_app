"""Static/schema smoke tests for the research suite.

Run before a long backtest. It checks:
- required script files exist
- scripts compile
- two-leg file has required columns
- four-leg file, if supplied, has required columns
- algebraic decomposition is full-rank when parameterized as common+differential
"""
from __future__ import annotations
import argparse, py_compile
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REQ2=["symbol","date","contract_0","contract_1","settle_0","settle_1","expiry_0","expiry_1"]
REQ4=REQ2+["contract_2","contract_3","settle_2","settle_3","expiry_2","expiry_3"]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--prices",required=True); ap.add_argument("--four-leg") ; a=ap.parse_args()
    scripts=["novel_spanning_2leg_v2.py","build_four_curve_panel.py","novel_curve_identification_v2.py","run_research.py"]
    print("SCRIPT COMPILE")
    for s in scripts:
        py_compile.compile(str(HERE/s),doraise=True); print("  OK",s)
    df=pd.read_parquet(a.prices); miss=[c for c in REQ2 if c not in df.columns]
    if miss: raise SystemExit("2-leg schema missing: "+", ".join(miss))
    print(f"2-leg schema OK: {df.shape}, symbols={df.symbol.nunique()}")
    if a.four_leg:
        f=pd.read_parquet(a.four_leg); miss=[c for c in REQ4 if c not in f.columns]
        if miss: raise SystemExit("4-leg schema missing: "+", ".join(miss))
        print(f"4-leg schema OK: {f.shape}, symbols={f.symbol.nunique()}")
    # Exact algebra: [common,differential] spans [front,deferred].
    X=np.array([[1,0.5],[1,-0.5]],float)
    assert np.linalg.matrix_rank(X)==2
    print("common/differential parameterization full-rank: OK")
    print("SMOKE TEST PASSED")

if __name__=="__main__":main()
