"""One-command research driver.

Runs the valid 2-leg diagnostic. If a four-maturity file is supplied and has
contract_0..3, runs the full identification test. Otherwise it records the
exact missing schema instead of pretending the test ran.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess, sys
import pandas as pd

HERE=Path(__file__).resolve().parent


def run(cmd, log):
    log.append("$ " + " ".join(map(str,cmd)))
    p=subprocess.run(cmd,capture_output=True,text=True)
    log.append(p.stdout)
    if p.stderr: log.append("[stderr]\n"+p.stderr)
    log.append(f"[exit={p.returncode}]")
    return p.returncode


def schema(path):
    df=pd.read_parquet(path)
    return list(df.columns), df.shape


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--two-leg",required=True)
    ap.add_argument("--four-leg")
    ap.add_argument("--out",default="research_report.txt")
    a=ap.parse_args()
    logs=["LOCAL FRONT-END CURVE RESEARCH SUITE", "="*90]
    logs.append(f"two-leg schema/shape: {schema(a.two_leg)}")
    rc=run([sys.executable,str(HERE/"novel_spanning_2leg_v2.py"),"--prices",a.two_leg],logs)
    if a.four_leg:
        cols,shape=schema(a.four_leg); logs.append(f"four-leg schema/shape: {shape}")
        need=[f"{x}_{k}" for k in range(4) for x in ("contract","settle","expiry")]
        miss=[x for x in need if x not in cols]
        if miss:
            logs.append("FOUR-LEG TEST NOT RUN; missing: "+", ".join(miss))
        else:
            run([sys.executable,str(HERE/"novel_curve_identification_v2.py"),"--prices",a.four_leg],logs)
    else:
        logs.append("FOUR-LEG TEST NOT RUN: no --four-leg file supplied. This is expected for the current two-leg dataset.")
    Path(a.out).write_text("\n".join(logs)+"\n")
    print(f"wrote {a.out}")
    print("2-leg exit:",rc)

if __name__=="__main__":main()
