"""FULL FOUR-MATURITY IDENTIFICATION TEST.

Core question:
    Does a front-end, same-underlying residual predict next-month returns
    after the rest of the curve and conventional controls are removed?

Primary signal (fully ex ante):
    SEG01 = 12m momentum of maturity 0 - 12m momentum of maturity 1

For each commodity and month t, estimate using ONLY months t-60 ... t-1:
    SEG01 = a + b1*SEG12 + b2*SEG23 + b3*CURVE_LEVEL + e

LOCAL_RESIDUAL_t = SEG01_t - fitted value_t

This is deliberately stronger than simply trading basis-momentum. It asks
whether the front segment contains information that cannot be explained by
the deeper same-commodity curve.

Additional tests:
    - front / middle / remote residual symmetry
    - Fama-MacBeth predictive regressions with HAC inference
    - rank portfolio diagnostics
    - subperiod stability
    - commodity jackknife
    - within-month signal permutation placebo
    - correlation and variance decomposition

The script refuses to run if fewer than four maturities exist.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except Exception:
    BY_SYMBOL = {}

FORMATION = 12
FIT_WINDOW = 60
MIN_FIT = 36
MIN_CS = 8
PLACEBOS = 200

REQ = [
    "symbol","date",
    "contract_0","contract_1","contract_2","contract_3",
    "settle_0","settle_1","settle_2","settle_3",
    "expiry_0","expiry_1","expiry_2","expiry_3",
]


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    miss = [c for c in REQ if c not in df.columns]
    if miss:
        raise SystemExit(
            "PRIMARY NOVEL TEST CANNOT RUN. Missing columns: " + ", ".join(miss)
        )
    df["date"] = pd.to_datetime(df["date"])
    for k in range(4):
        df[f"expiry_{k}"] = pd.to_datetime(df[f"expiry_{k}"])
    if BY_SYMBOL:
        df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
        df = df[df["asset"] == "commodity"].copy()
    keys = ["symbol","date"]
    if "oi_0" in df.columns:
        df = (df.sort_values(keys+["oi_0"], na_position="first")
                .drop_duplicates(keys, keep="last"))
    else:
        df = df.drop_duplicates(keys, keep="last")
    df = df.sort_values(keys).reset_index(drop=True)
    for k in range(4):
        # Returns only within the life of the exact contract occupying slot k.
        blk = df.groupby("symbol")[f"contract_{k}"].transform(
            lambda s: (s != s.shift()).cumsum()
        )
        prev = df.groupby(["symbol", blk])[f"settle_{k}"].shift()
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(df[f"settle_{k}"] / prev)
        df[f"r{k}"] = r.where(np.isfinite(r))
    return df


def monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M")
    d["dom"] = d.groupby(["symbol","ym"]).cumcount()
    marks = d.loc[d.dom == 0, ["symbol","ym","date"]].rename(columns={"date":"mark"})
    for k in range(4):
        d[f"c{k}"] = d.groupby("symbol")[f"r{k}"].transform(lambda s: s.fillna(0).cumsum())
    s = d.merge(marks, left_on=["symbol","ym","date"], right_on=["symbol","ym","mark"], how="inner")
    s = s.sort_values(["symbol","ym"]).reset_index(drop=True)
    g = s.groupby("symbol", sort=False)
    for k in range(4):
        s[f"rm{k}"] = g[f"c{k}"].diff()
        s[f"m{k}"] = g[f"rm{k}"].transform(
            lambda x: x.rolling(FORMATION, min_periods=FORMATION).sum()
        )
    s["seg01"] = s.m0 - s.m1
    s["seg12"] = s.m1 - s.m2
    s["seg23"] = s.m2 - s.m3
    s["curve_level"] = s[[f"m{k}" for k in range(4)]].mean(axis=1)
    # Broad curve slope and quadratic curvature for descriptive controls.
    s["curve_slope"] = s.m0 - s.m3
    s["curve_curvature"] = s.m0 - 2*s.m1 + s.m2
    s["basis"] = np.log(s.settle_0 / s.settle_1)
    tau = (s.expiry_1 - s.expiry_0).dt.days / 365.25
    s["ann_basis"] = s.basis / tau.replace(0, np.nan)
    s["fwd"] = g.rm0.shift(-1)
    return s


def rolling_residual(panel: pd.DataFrame, y: str, xs: list[str], window=FIT_WINDOW, min_fit=MIN_FIT, out_name="resid") -> pd.DataFrame:
    """Per-symbol rolling projection using only observations strictly before t."""
    d = panel.sort_values(["symbol","ym"]).copy()
    vals = np.full(len(d), np.nan)
    pos = {idx:i for i,idx in enumerate(d.index)}
    for _, idxs in d.groupby("symbol", sort=False).groups.items():
        ids = list(idxs)
        arr = d.loc[ids, [y] + xs].to_numpy(float)
        for i in range(len(ids)):
            # prior observations, excluding current row
            lo = max(0, i-window)
            hist = arr[lo:i]
            if hist.shape[0] < min_fit:
                continue
            good = np.isfinite(hist).all(axis=1)
            H = hist[good]
            if H.shape[0] < min_fit:
                continue
            X = np.column_stack([np.ones(len(H)), H[:,1:]])
            Y = H[:,0]
            if np.linalg.matrix_rank(X) < X.shape[1]:
                continue
            try:
                beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            cur = arr[i]
            if not np.isfinite(cur).all():
                continue
            vals[pos[ids[i]]] = cur[0] - np.dot(np.r_[1.0, cur[1:]], beta)
    d[out_name] = vals
    return d


def prepare_signals(p: pd.DataFrame) -> pd.DataFrame:
    d = p.copy()
    # The primary local front residual: front segment unexplained by the rest of
    # the same commodity curve and its common level momentum.
    d = rolling_residual(
        d, "seg01", ["seg12","seg23","curve_level"],
        out_name="local_front"
    )
    # Symmetric falsification residuals: if the phenomenon is genuinely front-local,
    # analogous residuals deeper in the curve should not dominate.
    d = rolling_residual(
        d, "seg12", ["seg01","seg23","curve_level"],
        out_name="local_middle"
    )
    d = rolling_residual(
        d, "seg23", ["seg01","seg12","curve_level"],
        out_name="local_remote"
    )
    return d


def cross_section_beta(g: pd.DataFrame, y: str, xs: list[str]):
    z = g[[y]+xs].replace([np.inf,-np.inf],np.nan).dropna()
    if len(z) < max(MIN_CS, len(xs)+3):
        return None
    X = np.column_stack([np.ones(len(z)), z[xs].to_numpy(float)])
    Y = z[y].to_numpy(float)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    try:
        return np.linalg.lstsq(X,Y,rcond=None)[0][1:]
    except np.linalg.LinAlgError:
        return None


def fmb(p: pd.DataFrame, xs: list[str], y="fwd"):
    B=[]
    for _,g in p.groupby("ym",sort=False):
        b=cross_section_beta(g,y,xs)
        if b is not None:B.append(b)
    if len(B)<48:return None
    B=np.asarray(B,float)
    out={"months":len(B)}
    for j,x in enumerate(xs):
        z=B[:,j]; mu=float(z.mean()); u=z-mu; n=len(z); L=min(6,n-1)
        lrv=float(np.dot(u,u)/n)
        for lag in range(1,L+1):
            gamma=float(np.dot(u[lag:],u[:-lag])/n)
            lrv += 2*(1-lag/(L+1))*gamma
        se=math.sqrt(max(lrv,0)/n)
        out[x+"_beta"]=mu
        out[x+"_t_hac6"]=mu/se if se>0 else np.nan
    return out


def rank_portfolio(p: pd.DataFrame, signal: str):
    out=[]
    for ym,g in p.groupby("ym",sort=True):
        z=g[["symbol",signal,"fwd"]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)<MIN_CS:continue
        r=z[signal].rank(); w=r-r.mean(); den=np.abs(w).sum()
        if den<=0:continue
        out.append((ym,float(((w/den)*z.fwd).sum())))
    return pd.Series(dict(out)).sort_index()


def pstats(r: pd.Series):
    r=r.dropna()
    if len(r)<48:return {}
    ret=r.mean()*12; vol=r.std(ddof=1)*math.sqrt(12); sr=ret/vol if vol>0 else np.nan
    eq=(1+r).cumprod(); dd=float((eq/eq.cummax()-1).min())
    return {"n":len(r),"ann":ret,"vol":vol,"sharpe":sr,"t":sr*math.sqrt(len(r)/12),"maxdd":dd}


def corr_rows(p, a, b):
    return float(p[[a,b]].replace([np.inf,-np.inf],np.nan).corr().iloc[0,1])


def jackknife(p: pd.DataFrame, signal: str):
    rows=[]
    for sym in sorted(p.symbol.dropna().unique()):
        r=rank_portfolio(p[p.symbol!=sym],signal)
        st=pstats(r)
        rows.append((sym,st.get("sharpe",np.nan)))
    return pd.DataFrame(rows,columns=["dropped_symbol","sharpe"])


def placebo(p: pd.DataFrame, signal: str, n=PLACEBOS, seed=12345):
    rng=np.random.default_rng(seed)
    real=p.copy(); rr=rank_portfolio(real,signal); rst=pstats(rr)
    out=[]
    for b in range(n):
        q=[]
        for ym,g in p.groupby("ym",sort=True):
            z=g[["symbol",signal,"fwd"]].replace([np.inf,-np.inf],np.nan).dropna()
            if len(z)<MIN_CS:continue
            vals=z[signal].to_numpy().copy(); rng.shuffle(vals)
            r=pd.Series(vals,index=z.index)
            ranks=r.rank(); w=ranks-ranks.mean(); den=np.abs(w).sum()
            if den<=0:continue
            q.append(float(((w/den)*z.fwd).sum()))
        qs=pd.Series(q)
        if len(qs)>=48:
            out.append(pstats(qs)["sharpe"])
    arr=np.asarray(out,float)
    return {
        "real_sharpe":rst.get("sharpe",np.nan),
        "placebo_n":len(arr),
        "placebo_mean":float(np.nanmean(arr)) if len(arr) else np.nan,
        "placebo_sd":float(np.nanstd(arr,ddof=1)) if len(arr)>1 else np.nan,
        "real_z":float((rst.get("sharpe",np.nan)-np.nanmean(arr))/np.nanstd(arr,ddof=1)) if len(arr)>1 else np.nan,
        "placebo_95":tuple(np.nanpercentile(arr,[2.5,97.5])) if len(arr) else (np.nan,np.nan),
    }


def split_stats(p, signal):
    dates=sorted(p.ym.dropna().unique())
    if not dates:return pd.DataFrame()
    cuts=[]
    n=len(dates)
    edges=[0,n//3,2*n//3,n]
    for a,b in zip(edges[:-1],edges[1:]):
        sub=p[(p.ym>=dates[a])&(p.ym<=dates[b-1])]
        r=rank_portfolio(sub,signal); st=pstats(r)
        cuts.append((str(dates[a]),str(dates[b-1]),st.get("n"),st.get("sharpe"),st.get("t")))
    return pd.DataFrame(cuts,columns=["start","end","n","sharpe","t"])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prices",required=True)
    ap.add_argument("--out",default="curve_identification_report.txt")
    a=ap.parse_args()
    raw=load(a.prices)
    p=prepare_signals(monthly_panel(raw))

    lines=[]
    def pr(x=""):
        print(x); lines.append(str(x))

    pr("="*96)
    pr("LOCAL FRONT-END CURVE RESIDUAL — FULL IDENTIFICATION TEST")
    pr("="*96)
    pr(f"formation={FORMATION}m rolling_projection={FIT_WINDOW}m min_fit={MIN_FIT} curve_points=4")
    pr(f"rows={len(p):,} instruments={p.symbol.nunique()} months={p.ym.nunique()}")
    pr("Primary residual: SEG01 ~ SEG12 + SEG23 + CURVE_LEVEL, fit only on prior 60 months.")

    pr("\n[1] SIGNAL GEOMETRY")
    cols=["m0","m1","m2","m3","seg01","seg12","seg23","curve_level","curve_slope","curve_curvature","basis"]
    pr(p[cols].corr().round(3).to_string())

    pr("\n[2] PREDICTIVE SPANNING")
    models={
        "front segment alone":["seg01"],
        "deep segments + level":["seg12","seg23","curve_level"],
        "front + deep segments + level":["seg01","seg12","seg23","curve_level"],
        "local residual alone":["local_front"],
        "local residual + conventional controls":["local_front","m0","basis","curve_level"],
        "curve geometry + front residual":["local_front","seg12","seg23","curve_level","basis"],
    }
    results={}
    for name,xs in models.items():
        results[name]=fmb(p,xs)
        pr(f"\n{name}\n{results[name]}")

    pr("\n[3] ECONOMIC LOCALITY / SYMMETRY")
    for sig in ["local_front","local_middle","local_remote"]:
        r=rank_portfolio(p,sig); pr(f"{sig}: {pstats(r)}")

    pr("\n[4] BASELINE SEGMENTS")
    for sig in ["seg01","seg12","seg23","curve_level","curve_slope","curve_curvature"]:
        r=rank_portfolio(p,sig); pr(f"{sig}: {pstats(r)}")

    pr("\n[5] SUBPERIOD STABILITY — local_front")
    pr(split_stats(p,"local_front").to_string(index=False))

    pr("\n[6] COMMODITY JACKKNIFE — local_front")
    jk=jackknife(p,"local_front")
    pr(jk.to_string(index=False))
    pr(f"jackknife min={jk.sharpe.min():.3f} median={jk.sharpe.median():.3f} max={jk.sharpe.max():.3f}")

    pr("\n[7] WITHIN-MONTH SIGNAL PERMUTATION PLACEBO")
    pl=placebo(p,"local_front")
    pr(pl)

    pr("\n[8] PRACTICAL INTERPRETATION")
    pr("A positive local_front coefficient after controlling for deeper curve segments, curve level, front momentum and basis is the key result.")
    pr("Front specificity is strengthened if local_front materially outperforms local_middle and local_remote and remains positive across subperiods/jackknifes.")
    pr("None of this, by itself, proves a physical-inventory mechanism. That requires separate inventory/hedging/intermediary tests.")
    pr("A failure is scientifically useful: it implies the BM alpha is spanned by ordinary curve geometry rather than a distinct local-front component.")

    Path(a.out).write_text("\n".join(lines)+"\n")
    print(f"\nReport written to {a.out}")

if __name__=="__main__": main()
