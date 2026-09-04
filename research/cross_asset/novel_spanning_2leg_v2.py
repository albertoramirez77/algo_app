"""Valid 2-leg spanning diagnostic for the current px_clean.parquet.

This intentionally does NOT claim novelty. It answers the narrower question:
Does the maturity-differential component survive common momentum and current basis?

Usage:
    python novel_spanning_2leg_v2.py --prices data/px_clean.parquet
"""
from __future__ import annotations
import argparse, math
import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except Exception:
    BY_SYMBOL = {}

FORMATION = 12
MIN_CS = 6


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    req = ["symbol","date","contract_0","contract_1","settle_0","settle_1","expiry_0","expiry_1"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    df["date"] = pd.to_datetime(df["date"])
    for c in ["expiry_0","expiry_1"]:
        df[c] = pd.to_datetime(df[c])
    if BY_SYMBOL:
        df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
        df = df[df["asset"] == "commodity"].copy()
    keys = ["symbol","date"]
    if "oi_0" in df.columns:
        df = (df.sort_values(keys + ["oi_0"], na_position="first")
                .drop_duplicates(keys, keep="last"))
    else:
        df = df.drop_duplicates(keys, keep="last")
    df = df.sort_values(keys).reset_index(drop=True)
    for k in (0,1):
        blk = df.groupby("symbol")[f"contract_{k}"].transform(lambda s: (s != s.shift()).cumsum())
        prev = df.groupby(["symbol",blk])[f"settle_{k}"].shift()
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(df[f"settle_{k}"] / prev)
        df[f"r{k}"] = r.where(np.isfinite(r))
    return df


def monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M")
    d["dom"] = d.groupby(["symbol","ym"]).cumcount()
    marks = d.loc[d["dom"] == 0, ["symbol","ym","date"]].rename(columns={"date":"mark"})
    for k in (0,1):
        d[f"c{k}"] = d.groupby("symbol")[f"r{k}"].transform(lambda s: s.fillna(0.0).cumsum())
    s = d.merge(marks, left_on=["symbol","ym","date"], right_on=["symbol","ym","mark"], how="inner")
    s = s.sort_values(["symbol","ym"]).reset_index(drop=True)
    g = s.groupby("symbol", sort=False)
    s["r0_m"] = g["c0"].diff()
    s["r1_m"] = g["c1"].diff()
    s["m0"] = g["r0_m"].transform(lambda x: x.rolling(FORMATION, min_periods=FORMATION).sum())
    s["m1"] = g["r1_m"].transform(lambda x: x.rolling(FORMATION, min_periods=FORMATION).sum())
    s["bm"] = s["m0"] - s["m1"]
    s["common_mom"] = 0.5 * (s["m0"] + s["m1"])
    s["basis"] = np.log(s["settle_0"] / s["settle_1"])
    s["fwd"] = g["r0_m"].shift(-1)
    return s


def ols_cross_section(g: pd.DataFrame, y: str, xs: list[str]):
    z = g[[y] + xs].replace([np.inf,-np.inf],np.nan).dropna()
    if len(z) < max(MIN_CS, len(xs) + 3):
        return None
    X = np.column_stack([np.ones(len(z)), z[xs].to_numpy(float)])
    Y = z[y].to_numpy(float)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    return np.linalg.lstsq(X,Y,rcond=None)[0][1:]


def fmb(p: pd.DataFrame, y: str, xs: list[str]):
    betas=[]
    for _,g in p.groupby("ym",sort=False):
        b=ols_cross_section(g,y,xs)
        if b is not None: betas.append(b)
    if len(betas)<24: return None
    B=np.asarray(betas)
    out={"months":len(B)}
    for j,x in enumerate(xs):
        xj=B[:,j]; mu=float(xj.mean()); u=xj-mu; n=len(xj)
        L=min(3,n-1)
        lrv=float(np.dot(u,u)/n)
        for lag in range(1,L+1):
            gamma=float(np.dot(u[lag:],u[:-lag])/n)
            lrv += 2*(1-lag/(L+1))*gamma
        se=math.sqrt(max(lrv,0)/n)
        out[f"{x}_beta"]=mu
        out[f"{x}_t_hac3"]=mu/se if se>0 else np.nan
    return out


def rank_portfolio(p: pd.DataFrame, signal: str):
    out=[]
    for ym,g in p.groupby("ym",sort=True):
        z=g[["symbol",signal,"fwd"]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)<MIN_CS: continue
        w=z[signal].rank()-z[signal].rank().mean(); den=np.abs(w).sum()
        if den<=0: continue
        out.append((ym,float(((w/den)*z["fwd"]).sum())))
    return pd.Series(dict(out)).sort_index()


def pstats(r: pd.Series):
    r=r.dropna()
    if len(r)<48:return {}
    vol=r.std(ddof=1)*math.sqrt(12); ret=r.mean()*12; sr=ret/vol if vol>0 else np.nan
    return {"n":len(r),"ann":ret,"vol":vol,"sharpe":sr,"t":sr*math.sqrt(len(r)/12)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--prices",default="data/px_clean.parquet"); a=ap.parse_args()
    p=monthly_panel(load(a.prices))
    print("="*90); print("VALID 2-LEG SPANNING TEST"); print("="*90)
    print(f"rows={len(p):,}, instruments={p.symbol.nunique()}, months={p.ym.nunique()}")
    tests={
        "common_momentum + basis": ["common_mom","basis"],
        "common_momentum + basis + differential": ["common_mom","basis","bm"],
        "differential + basis": ["bm","basis"],
        "common_momentum + differential": ["common_mom","bm"],
    }
    for name,xs in tests.items():
        print(f"\n{name}"); print(fmb(p,"fwd",xs))
    print("\nRANK PORTFOLIOS")
    for x in ["common_mom","bm"]: print(x,pstats(rank_portfolio(p,x)))
    print("\nCORRELATIONS")
    print(p[["m0","m1","common_mom","bm","basis"]].corr().round(3).to_string())
    print("\nInterpretation: if bm remains positive after common_mom and basis, it survives the valid 2-leg spanning test. That is NOT yet a novelty proof; four maturities are required.")

if __name__ == "__main__": main()
