"""
dd_diag.py — what actually happened during the 42-month drawdown?

    python dd_diag.py --prices data/px_clean.parquet
"""
import argparse
import numpy as np
import pandas as pd
from universe import BY_SYMBOL

CAPITAL, VOL_TARGET, IDM, J, VOL_WINDOW = 450_000.0, 0.20, 2.5, 12, 6
N_GRIDS, COST_MULTIPLE = 21, 3.0

def load(path):
    df = pd.read_parquet(path)
    for c in ("date","expiry_0","expiry_1"):
        if c in df.columns: df[c] = pd.to_datetime(df[c])
    df = df[df["contract_0"] != df["contract_1"]]
    df = (df.sort_values(["symbol","date","oi_0"], na_position="first")
            .drop_duplicates(["date","symbol"], keep="last")
            .sort_values(["symbol","date"]).reset_index(drop=True))
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df = df[df["asset"]=="commodity"].copy()
    med = df.groupby("symbol")["settle_0"].median()
    cost = {}
    for s in med.index:
        i = BY_SYMBOL[s]; n = med[s]*i.dollar_price_mult
        cost[s] = 1.5*(i.tick_value/n*1e4) + i.commission/n*1e4
    cs = pd.Series(cost)
    drop = set(cs[cs > COST_MULTIPLE*cs.median()].index)
    df = df[~df["symbol"].isin(drop)].copy()
    for leg in ("0","1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(lambda s: (s!=s.shift(1)).cumsum())
        prev = df.groupby(["symbol",blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"]/prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    df["ym"] = df["date"].dt.to_period("M"); df["dom"] = df.groupby(["symbol","ym"]).cumcount()
    return df

def grid_targets(df, offset, min_n=6):
    d = df.sort_values(["symbol","date"]).copy()
    for leg in ("0","1"):
        d[f"c{leg}"] = d.groupby("symbol")[f"r{leg}"].transform(lambda s: s.fillna(0.0).cumsum())
    snap = d[d["dom"]==offset][["symbol","ym","date","c0","c1","settle_0"]].copy()
    if snap.empty: return pd.DataFrame()
    snap = snap.sort_values(["symbol","ym"]).reset_index(drop=True)
    g = snap.groupby("symbol")
    snap["r0"]=g["c0"].diff(); snap["r1"]=g["c1"].diff()
    snap["bm"]=(g["r0"].transform(lambda s: s.rolling(J,min_periods=J).sum())
                -g["r1"].transform(lambda s: s.rolling(J,min_periods=J).sum()))
    snap["vol"]=(g["r0"].transform(lambda s: s.rolling(VOL_WINDOW,min_periods=3).std())*np.sqrt(12)).groupby(snap["symbol"]).shift(1)
    snap["px_entry"]=g["settle_0"].shift(1)
    rows=[]
    for dt,gg in snap.groupby("date"):
        s = gg[["symbol","bm","vol","px_entry"]].dropna()
        s = s[(s["vol"]>0)&(s["px_entry"]>0)]
        if len(s)<min_n: continue
        r=s["bm"].rank(); w=(r-r.mean()).to_numpy(); gr=np.abs(w).sum()
        if gr<=0: continue
        w=w/gr
        for sym,wi,vol,px in zip(s["symbol"],w,s["vol"],s["px_entry"]):
            inst=BY_SYMBOL[sym]; den=inst.dollar_price_mult*px*vol
            if den>0: rows.append(dict(date=dt,symbol=sym,target=wi*CAPITAL*VOL_TARGET*IDM/den))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--prices",default="data/px_clean.parquet")
    a=ap.parse_args()
    df=load(a.prices)
    frames=[f for f in (grid_targets(df,o) for o in range(N_GRIDS)) if not f.empty]
    dates=pd.DatetimeIndex(sorted(df["date"].unique())); syms=sorted(df["symbol"].unique())
    ret=df.pivot_table(index="date",columns="symbol",values="r0").reindex(dates,columns=syms)
    px=df.pivot_table(index="date",columns="symbol",values="settle_0").reindex(dates,columns=syms).ffill()
    stacks=[(tf.pivot_table(index="date",columns="symbol",values="target").reindex(index=dates,columns=syms).ffill()).to_numpy() for tf in frames if not tf.empty]
    S=np.stack(stacks,axis=0); cnt=np.sum(~np.isnan(S),axis=0)
    T=np.divide(np.nansum(S,axis=0),np.maximum(cnt,1),out=np.zeros_like(cnt,dtype=float),where=cnt>0)
    N=np.round(T)
    dpm=np.array([BY_SYMBOL[s].dollar_price_mult for s in syms])
    comm=np.array([BY_SYMBOL[s].commission for s in syms])
    med=df.groupby("symbol")["settle_0"].median()
    bps=np.array([1.5*(BY_SYMBOL[s].tick_value/(med[s]*BY_SYMBOL[s].dollar_price_mult)*1e4)+BY_SYMBOL[s].commission/(med[s]*BY_SYMBOL[s].dollar_price_mult)*1e4 for s in syms])
    P=np.nan_to_num(px.to_numpy(),nan=0.0); R=np.nan_to_num(ret.to_numpy(),nan=0.0)
    held=N[:-1]
    pnl=np.nansum(held*dpm*P[:-1]*np.expm1(R[1:]),axis=1)
    trades=np.abs(np.diff(N,axis=0))
    cost=np.nansum(trades*(comm+np.abs(dpm)*P[:-1]*bps/1e4),axis=1)
    daily=pd.Series((pnl-cost)/CAPITAL,index=dates[1:])
    net=daily.resample("ME").sum(); net=net[net!=0]

    eq=(1+net).cumprod(); peak=eq.cummax(); dd=eq/peak-1
    under=(dd<-1e-9).to_numpy(); runs=[]; start=None
    for i,u in enumerate(under):
        if u and start is None: start=i
        if not u and start is not None: runs.append((start,i-1)); start=None
    if start is not None: runs.append((start,len(under)-1))
    runs=[(net.index[s],net.index[e],e-s+1,dd.iloc[s:e+1].min()) for s,e in runs]
    runs.sort(key=lambda r:-r[2])

    print("=== ALL DRAWDOWN EPISODES, LONGEST FIRST ===\n")
    print(f"  {'start':>9s} {'end':>9s} {'months':>7s} {'trough':>8s}")
    for s,e,n,tr in runs[:8]:
        print(f"  {str(s):>9s} {str(e):>9s} {n:>7d} {tr*100:>7.1f}%")

    s,e,n,tr = runs[0]
    print(f"\n=== INSIDE THE LONGEST ({s} to {e}, {n} months) ===\n")
    seg = net[(net.index>=s)&(net.index<=e)]
    print(f"  months positive: {(seg>0).sum()} of {len(seg)}   mean monthly return: {seg.mean()*100:+.2f}%")
    print(f"  best month: {seg.idxmax()} {seg.max()*100:+.2f}%   worst month: {seg.idxmin()} {seg.min()*100:+.2f}%")
    yr = seg.groupby(seg.index.year).sum()
    print(f"\n  by year within the drawdown:")
    for y,v in yr.items():
        print(f"    {y}: {v*100:+.2f}%")

if __name__=="__main__":
    main()