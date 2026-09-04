"""
controls.py — the four questions the pitch still does not answer.

    python controls.py --prices data/px_clean.parquet

A  NET SECTOR CAP, done correctly this time
B  THE HEDGE, which the fund asked about directly and the pitch never answers
C  DECAY, because strategies age out
D  VOLATILITY STATE, because an informed signal may only pay in certain conditions

WHY A IS BEING REDONE

The first attempt capped each sector's share of GROSS weight and found the tilts unchanged
at every level: energy 0.27, grains 0.23, livestock 0.54, metals 0.38, oilseeds 0.34,
identical capped and uncapped. That is not a null result, it is the wrong test. A sector can
hold a third of gross weight while being entirely one-directional. The exposure that
matters is NET - livestock ran a persistent long of roughly half of capital - and gross
weight does not constrain it.

This caps the net directly, and adds the limiting case of full sector neutrality, where
weights are demeaned within each sector so net sector exposure is exactly zero by
construction.

WHY B MATTERS

"How do you hedge it?" was asked and the pitch answers with dollar-neutrality, which is a
property of the weights rather than a hedge. There is a real exposure worth hedging: the
rolling thirty-six-month dollar beta ranges from -1.2 to +1.3 even though the
unconditional beta is near zero. The dataset already contains eight currency futures, so
the hedge is constructible from instruments in hand. It is simulated as an overlay - short
the trailing dollar beta against the currency basket - and reported honestly, including
when the hedge costs more than it saves.

WHY C AND D MATTER

Boons and Prado published in 2019. If the effect has decayed since, the strategy is a
description of history. And the fund's own point about alternative data - that an informed
model only adds value when volatility is low enough for the information to matter - is a
conditional claim that has never been tested here in either direction.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

CAPITAL, VOL_TARGET, IDM, J, VOL_WINDOW = 450_000.0, 0.20, 2.5, 12, 6
COST_MULTIPLE = 3.0
NET_CAPS = [None, 0.60, 0.40, 0.25, 0.10, "neutral"]


def load(path: str):
    df = pd.read_parquet(path)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df = df[df["contract_0"] != df["contract_1"]]
    df = (df.sort_values(["symbol", "date", "oi_0"], na_position="first")
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"]).reset_index(drop=True))
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df = df[df["asset"] != "?"].copy()
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")

    comm = df[df["asset"] == "commodity"]
    med = comm.groupby("symbol")["settle_0"].median()
    cost = {}
    for s in med.index:
        i = BY_SYMBOL[s]
        n = med[s] * i.dollar_price_mult
        cost[s] = 1.5 * (i.tick_value / n * 1e4) + i.commission / n * 1e4
    cs = pd.Series(cost)
    drop = set(cs[cs > COST_MULTIPLE * cs.median()].index)
    if drop:
        print(f"  universe rule excludes {sorted(drop)}")

    m = (df.groupby(["symbol", "ym", "asset"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                px=("settle_0", "last"), nd=("r0", "size")).reset_index())
    m = m[m["nd"] >= 10].sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["spread"] = m["r0"] - m["r1"]
    m["bm"] = (g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
               - g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum()))
    m["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    m["sector"] = m["symbol"].map(lambda s: BY_SYMBOL[s].sector)

    cm = m[(m["asset"] == "commodity") & (~m["symbol"].isin(drop))].copy()
    fx = m[m["asset"] == "fx"].copy()
    return cm, fx


def book(m: pd.DataFrame, net_cap=None, bps: float = 3.0, drop: str | None = None,
         min_n: int = 6):
    """
    `net_cap` bounds each sector's NET notional as a share of capital. "neutral" demeans
    weights within each sector, forcing net sector exposure to exactly zero.
    """
    if drop:
        m = m[m["symbol"] != drop]
    prev, out, tilt_rows = {}, {}, []
    for ym, g in m.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd", "sector"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s["bm"].rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        sec = s["sector"].to_numpy()

        if net_cap == "neutral":
            # demean inside each sector: net sector weight becomes exactly zero
            for x in set(sec):
                msk = sec == x
                if msk.sum() > 1:
                    w[msk] -= w[msk].mean()
        elif net_cap is not None:
            for _ in range(6):
                over = False
                for x in set(sec):
                    msk = sec == x
                    net = w[msk].sum()
                    if abs(net) > net_cap + 1e-9:
                        over = True
                        # remove the excess evenly, preserving relative ordering
                        w[msk] -= (net - np.sign(net) * net_cap) / msk.sum()
                if not over:
                    break
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr

        pnl = cost = 0.0
        held, notional = {}, {}
        for sym, wi, vol, px, fwd, sc in zip(s["symbol"], w, s["vol"], s["px_entry"],
                                             s["fwd"], s["sector"]):
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            den = dpm * px * vol
            if den <= 0:
                continue
            n = float(np.round(wi * CAPITAL * VOL_TARGET * IDM / den))
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            notional[sc] = notional.get(sc, 0.0) + n * dpm * px      # SIGNED
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
        tilt_rows.append({**{k: v / CAPITAL for k, v in notional.items()}, "ym": ym})
    return pd.Series(out).sort_index(), pd.DataFrame(tilt_rows).set_index("ym")


def st(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return dict(n=len(r), sharpe=np.nan, t=np.nan, ann=np.nan, vol=np.nan, dd=np.nan)
    yrs = len(r) / 12
    av = r.std(ddof=1) * np.sqrt(12)
    sr = (r.mean() * 12) / av if av > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(n=len(r), sharpe=sr, t=sr * np.sqrt(yrs), ann=r.mean() * 12, vol=av,
                dd=float((eq / eq.cummax() - 1).min()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    a = ap.parse_args()

    cm, fx = load(a.prices)
    syms = sorted(cm["symbol"].unique())

    # -------------------------------------------------------------- A
    print("\n" + "=" * 82)
    print("A. NET SECTOR CAP — the correct test")
    print("=" * 82)
    print("  The previous attempt capped GROSS weight per sector and the net tilts did not")
    print("  move at all, because a sector can hold a third of gross weight while being")
    print("  entirely one-directional. This caps the NET, which is the exposure that")
    print("  actually existed.\n")
    print(f"  {'cap':>9s} {'Sharpe':>8s} {'return':>8s} {'maxDD':>8s} "
          f"{'jackknife':>10s} {'max |net tilt|':>15s}")
    res = {}
    for cap in NET_CAPS:
        r, tl = book(cm, net_cap=cap)
        s = st(r)
        jk = [st(book(cm, net_cap=cap, drop=x)[0])["sharpe"] for x in syms]
        jk = [v for v in jk if np.isfinite(v)]
        mt = tl.mean().abs().max() if len(tl) else np.nan
        res[cap] = dict(s=s, jk=min(jk) if jk else np.nan, tilt=mt, tl=tl, r=r)
        lab = "none" if cap is None else ("neutral" if cap == "neutral" else f"{cap:.0%}")
        print(f"  {lab:>9s} {s['sharpe']:>8.3f} {s['ann']*100:>7.2f}% "
              f"{s['dd']*100:>7.1f}% {res[cap]['jk']:>10.3f} {mt:>14.2f}x")

    base = res[None]
    print("\n  mean NET sector notional as a multiple of capital:")
    print(f"  {'sector':12s} {'uncapped':>10s} {'25% cap':>10s} {'neutral':>10s}")
    for c in sorted(base["tl"].columns):
        v0 = base["tl"][c].mean()
        v25 = res[0.25]["tl"][c].mean() if c in res[0.25]["tl"] else np.nan
        vn = res["neutral"]["tl"][c].mean() if c in res["neutral"]["tl"] else np.nan
        print(f"  {c:12s} {v0:>+10.2f} {v25:>+10.2f} {vn:>+10.2f}")
    print("\n  If these columns now differ, the cap binds and the earlier test was simply")
    print("  measuring the wrong quantity. If they still do not, the tilt is being")
    print("  produced by integer rounding rather than by the weights, which is a")
    print("  different problem and cannot be fixed with a weight constraint.")

    # -------------------------------------------------------------- B
    print("\n" + "=" * 82)
    print("B. THE HEDGE — answering the question directly")
    print("=" * 82)
    print("  Dollar-neutrality is a property of the weights, not a hedge. The real")
    print("  exposure is a rolling dollar beta that swings widely even though the")
    print("  unconditional beta is near zero. Eight currency futures are already in the")
    print("  dataset, so the hedge is built from instruments in hand.\n")
    if fx.empty:
        print("  no currency data available; hedge cannot be constructed")
    else:
        # dollar factor: each FX contract is USD per foreign unit, so flip the sign
        usd = -fx.groupby("ym")["fwd"].mean().dropna()
        r0 = base["r"]
        j = pd.concat([r0.rename("y"), usd.rename("x")], axis=1).dropna()
        print(f"  {'window':>10s} {'Sharpe':>8s} {'return':>8s} {'maxDD':>8s} "
              f"{'|beta| mean':>12s} {'beta range':>12s}")
        s0 = st(j["y"])
        print(f"  {'unhedged':>10s} {s0['sharpe']:>8.3f} {s0['ann']*100:>7.2f}% "
              f"{s0['dd']*100:>7.1f}%")
        for win in (24, 36, 60):
            betas, hedged = [], {}
            for i in range(win, len(j)):
                w = j.iloc[i - win:i]
                b = np.cov(w["y"], w["x"])[0, 1] / w["x"].var() if w["x"].var() > 0 else 0.0
                betas.append(b)
                hedged[j.index[i]] = j["y"].iloc[i] - b * j["x"].iloc[i]
            h = pd.Series(hedged).sort_index()
            sh = st(h)
            bs = np.array(betas)
            print(f"  {str(win) + 'm beta':>10s} {sh['sharpe']:>8.3f} "
                  f"{sh['ann']*100:>7.2f}% {sh['dd']*100:>7.1f}% "
                  f"{np.abs(bs).mean():>12.3f} {bs.max()-bs.min():>12.2f}")
        print("\n  A hedge is worth running only if it improves the Sharpe ratio or the")
        print("  drawdown. Hedging a beta estimated from a rolling window adds estimation")
        print("  error, and when the true exposure is near zero that error dominates. If")
        print("  the hedged rows are worse, the honest answer to 'how do you hedge it' is")
        print("  that the exposure is small enough that hedging costs more than it saves,")
        print("  and here is the measurement that shows it.")

    # -------------------------------------------------------------- C
    print("\n" + "=" * 82)
    print("C. DECAY — strategies age out")
    print("=" * 82)
    r = base["r"]
    print("  Boons and Prado published in 2019. If the effect has decayed since, the")
    print("  strategy is a description of history rather than a proposal.\n")
    print(f"  {'period':22s} {'Sharpe':>8s} {'return':>8s} {'n':>5s}")
    for lo, hi, lab in (("2011-06", "2015-12", "first half"),
                        ("2016-01", "2026-08", "second half"),
                        ("2011-06", "2018-12", "before publication"),
                        ("2019-01", "2026-08", "after publication")):
        seg = r[(r.index >= lo) & (r.index <= hi)]
        s = st(seg)
        print(f"  {lab:22s} {s['sharpe']:>8.3f} {s['ann']*100:>7.2f}% {s['n']:>5d}")
    yr = r.groupby(r.index.year).sum()
    x = np.arange(len(yr))
    slope = np.polyfit(x, yr.to_numpy(), 1)[0] if len(yr) > 3 else np.nan
    print(f"\n  linear trend in annual return: {slope*100:+.2f} percentage points per year")
    print("  A negative slope with post-publication weakness is decay. A flat or positive")
    print("  slope means the premium has survived being written about, which is the")
    print("  stronger claim and the one worth making if the data supports it.")

    # -------------------------------------------------------------- D
    print("\n" + "=" * 82)
    print("D. VOLATILITY STATE — when does the information actually pay?")
    print("=" * 82)
    print("  The fund's point about alternative data is that an informed model only adds")
    print("  value when conditions let the information matter. That is a conditional")
    print("  claim and it applies here too: does curve information pay in calm markets,")
    print("  in stressed ones, or equally?\n")
    mvol = (cm.groupby("ym")["r0"].std() * np.sqrt(12)).dropna()
    # expanding median so the split uses no future information
    med = mvol.expanding().median().shift(1)
    j = pd.concat([r.rename("y"), mvol.rename("v"), med.rename("m")], axis=1).dropna()
    hi = j[j["v"] > j["m"]]["y"]
    lo = j[j["v"] <= j["m"]]["y"]
    print(f"  {'state':28s} {'Sharpe':>8s} {'return':>8s} {'n':>5s}")
    for lab, seg in (("high cross-sectional vol", hi), ("low cross-sectional vol", lo)):
        s = st(seg)
        print(f"  {lab:28s} {s['sharpe']:>8.3f} {s['ann']*100:>7.2f}% {s['n']:>5d}")
    print(f"\n  difference in mean monthly return: "
          f"{(hi.mean()-lo.mean())*100:+.2f} percentage points")
    print("  The split uses an EXPANDING median, so the state is classified using only")
    print("  information available at the time. A strategy that only works in one state")
    print("  is a conditional strategy and should be described as one.")

    print("\n" + "=" * 82)
    print("WHAT TO PUT IN THE PITCH")
    print("=" * 82)
    n25 = res[0.25]
    print(f"  sector cap at 25% of capital net:  Sharpe "
          f"{n25['s']['sharpe']-base['s']['sharpe']:+.3f}, "
          f"jackknife {n25['jk']-base['jk']:+.3f}, "
          f"max tilt {n25['tilt']-base['tilt']:+.2f}x")
    print("  Report whichever way it falls. A control that was tested and rejected is a")
    print("  stronger sentence than a control that was promised and never measured.")


if __name__ == "__main__":
    main()