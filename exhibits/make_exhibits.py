"""
make_exhibits.py — the one page of tables and charts the guide permits.

    python make_exhibits.py --prices px_clean.parquet

Four exhibits, each carrying an argument. Nothing decorative.

    1  the channels result        the thesis of the pitch, in one image
    2  equity curve and drawdown  what the strategy actually did, including the current one
    3  placebo distribution       signal separated from machinery
    4  robustness surface         parameter plateau and cost sensitivity

Everything except Exhibit 1 is computed from the price file at runtime. Exhibit 1 uses the
figures printed by channels.py, which are listed explicitly below so any of them can be
traced back to the run that produced it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from universe import BY_SYMBOL
except ImportError:
    raise SystemExit("universe.py must sit beside this script")

CAPITAL, VOL_TARGET, IDM_CAP, J, VOL_WINDOW = 450_000.0, 0.20, 2.5, 12, 6

INK, MUTE, ACC, NEG = "#1a1a1a", "#8a8a8a", "#c1440e", "#2b6a8f"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": INK, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Statistical tiers come from channels.py; the proximity tiers are computed from the
# price file at runtime. Both are listed so any figure can be traced to its source.


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("date", "expiry_0", "expiry_1"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df = df[df["contract_0"] != df["contract_1"]]
    df = (df.sort_values(["symbol", "date", "oi_0"], na_position="first")
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["symbol", "date"]).reset_index(drop=True))
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df = df[df["asset"] == "commodity"].copy()
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
    m = (df.groupby(["symbol", "ym"])
           .agg(r0=("r0", lambda s: s.sum(min_count=1)),
                r1=("r1", lambda s: s.sum(min_count=1)),
                px=("settle_0", "last"), nd=("r0", "size")).reset_index())
    m = m[m["nd"] >= 10].sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = m.groupby("symbol")
    m["bm"] = (g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
               - g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum()))
    m["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(m["symbol"]).shift(1)
    m["px_entry"] = g["px"].shift(1)
    m["fwd"] = g["r0"].shift(-1)
    return m


CHAINS = {"ZS": ["ZM", "ZL"], "MCL": ["HO", "RB"]}


def r2(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    if len(y) < 250 or y.var() <= 0:
        return np.nan
    A = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.pinv(A.T @ A) @ (A.T @ y)
    return float(1.0 - (y - A @ b).var() / y.var())


def proximity(path):
    """
    Hedge quality by economic proximity, computed from daily returns. The peer tier is
    chosen IN SAMPLE from all other commodities — the alternative is allowed hindsight the
    curve never gets, and it still has to win.
    """
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["contract_0"] != df["contract_1"]]
    df = (df.sort_values(["symbol", "date", "oi_0"], na_position="first")
            .drop_duplicates(["date", "symbol"], keep="last"))
    df["asset"] = df["symbol"].map(lambda s: BY_SYMBOL[s].asset if s in BY_SYMBOL else "?")
    df = df[df["asset"] == "commodity"].sort_values(["symbol", "date"])
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    p0 = df.pivot_table(index="date", columns="symbol", values="r0").sort_index()
    p1 = df.pivot_table(index="date", columns="symbol", values="r1").sort_index()
    syms = [s for s in p0.columns if p0[s].notna().sum() > 500]

    curve, peer, chain, wins = [], [], [], 0
    for s in syms:
        y = p0[s].to_numpy()
        c = r2(y, p1[s].to_numpy().reshape(-1, 1)) if s in p1.columns else np.nan
        best = max((r2(y, p0[o].to_numpy().reshape(-1, 1)) for o in syms if o != s),
                   default=np.nan)
        if np.isfinite(c):
            curve.append(c)
        if np.isfinite(best):
            peer.append(best)
        if np.isfinite(c) and np.isfinite(best) and c > best:
            wins += 1
        if s in CHAINS:
            legs = [l for l in CHAINS[s] if l in p0.columns]
            if legs:
                v = r2(y, p0[legs].to_numpy())
                if np.isfinite(v):
                    chain.append((v, len(legs)))
    return dict(curve=np.mean(curve) if curve else np.nan,
                peer=np.mean(peer) if peer else np.nan,
                chain=np.mean([c for c, _ in chain]) if chain else np.nan,
                chain_n=np.mean([n for _, n in chain]) if chain else np.nan,
                wins=wins, n=len(syms))


def idm_of(m):
    n = max(m["symbol"].nunique(), 2)
    piv = m.pivot_table(index="ym", columns="symbol", values="r0")
    cm = piv.corr().to_numpy()
    rho = float(np.nanmean(cm[np.triu_indices_from(cm, k=1)]))
    return min(1 / np.sqrt((1 / n) + (1 - 1 / n) * max(rho if np.isfinite(rho) else .2, .01)),
               IDM_CAP)


def book(m, idm, bps=3.0, seed=None, J_=None, vw=None):
    if J_ or vw:
        mm = m.copy()
        g = mm.groupby("symbol")
        if J_:
            mm["bm"] = (g["r0"].transform(lambda s: s.rolling(J_, min_periods=J_).sum())
                        - g["r1"].transform(lambda s: s.rolling(J_, min_periods=J_).sum()))
        if vw:
            mm["vol"] = (g["r0"].transform(
                lambda s: s.rolling(vw, min_periods=3).std()) * np.sqrt(12)
                ).groupby(mm["symbol"]).shift(1)
        m = mm
    rng = np.random.default_rng(seed) if seed is not None else None
    prev, out = {}, {}
    for ym, g in m.groupby("ym"):
        s = g[["symbol", "bm", "vol", "px_entry", "fwd"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < 6:
            continue
        sv = s["bm"]
        if rng is not None:
            sv = pd.Series(rng.permutation(sv.to_numpy()), index=sv.index)
        r = sv.rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        pnl = cost = 0.0
        held = {}
        for sym, wi, vol, px, fwd in zip(s["symbol"], w, s["vol"], s["px_entry"], s["fwd"]):
            inst = BY_SYMBOL[sym]
            dpm = inst.dollar_price_mult
            den = dpm * px * vol
            if den <= 0:
                continue
            n = float(np.round(wi * CAPITAL * VOL_TARGET * idm / den))
            held[sym] = n
            pnl += n * dpm * px * (np.exp(fwd) - 1.0)
            tr = abs(n - prev.get(sym, 0.0))
            if tr > 0:
                cost += tr * (inst.commission + abs(dpm) * px * bps / 1e4)
        for sym in set(prev) - set(held):
            cost += abs(prev[sym]) * BY_SYMBOL[sym].commission
        prev = held
        out[ym] = (pnl - cost) / CAPITAL
    return pd.Series(out).sort_index()


def sharpe(r):
    r = r.dropna()
    if len(r) < 48:
        return np.nan
    av = r.std(ddof=1) * np.sqrt(12)
    return (r.mean() * 12) / av if av > 0 else np.nan


# ----------------------------------------------------------------------------------

def ex1(ax, px):
    """
    The thesis in one image: hedge quality per regressor, ordered by how literally the
    hedging instrument is the same physical thing. Only the curve clears the cost hurdle.
    """
    tiers = [
        ("deferred contract\nsame commodity, later date", px["curve"], 1.0, 0.760, ACC),
        ("crush / crack\nsame commodity, transformed", px["chain"],
         px["chain_n"] if np.isfinite(px["chain_n"]) else 2.0, None, "#7a9e3f"),
        ("best other commodity\nchosen with hindsight", px["peer"], 1.0, None, MUTE),
        ("5 principal components", 0.686, 5.0, 0.273, MUTE),
        ("8 principal components", 0.841, 8.0, -0.027, MUTE),
        ("equal-weighted market", 0.187, 1.0, -0.208, MUTE),
    ]
    tiers = [t for t in tiers if np.isfinite(t[1])]
    tiers.sort(key=lambda t: t[1] / max(t[2], 1))
    y = np.arange(len(tiers))
    vals = [t[1] / max(t[2], 1) for t in tiers]
    ax.barh(y, vals, color=[t[4] for t in tiers], height=0.62, zorder=3,
            edgecolor=INK, linewidth=0.5)
    for i, t in enumerate(tiers):
        lab = f"R\u00b2 {t[1]:.2f}"
        if t[2] > 1:
            lab += f" / {t[2]:.0f} reg"
        if t[3] is not None:
            lab += f"   SR {t[3]:+.2f}"
        ax.text(vals[i] + 0.012, i, lab, va="center", fontsize=6.1, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in tiers], fontsize=6.0, linespacing=1.35)
    ax.set_xlabel("variance of the common component removed, per regressor", fontsize=7)
    ax.set_xlim(0, max(vals) * 1.62)
    ax.set_title("1 \u00b7 Hedge quality tracks economic proximity",
                 fontsize=8.5, loc="left", weight="bold")
    ax.text(0.985, 0.60,
            f"curve beats a hindsight-chosen\npeer in {px['wins']} of {px['n']} instruments",
            transform=ax.transAxes, fontsize=6.1, color=ACC, ha="right", va="bottom",
            linespacing=1.4)


def ex2(ax, r):
    eq = (1 + r).cumprod()
    x = eq.index.to_timestamp()
    ax.plot(x, eq.to_numpy(), color=ACC, lw=1.3)
    ax.set_ylabel("growth of $1, net", fontsize=7)
    ax.set_title("2 · Equity curve and drawdown, 2011–2026",
                 fontsize=8.5, loc="left", weight="bold")
    ax.axhline(1.0, color=MUTE, lw=0.6, ls=":")
    dd = (eq / eq.cummax() - 1)
    ax2 = ax.twinx()
    ax2.fill_between(x, dd.to_numpy() * 100, 0, color=NEG, alpha=0.20, lw=0)
    ax2.set_ylabel("drawdown (%)", fontsize=7, color=NEG)
    ax2.set_ylim(dd.min() * 100 * 2.4, 0)
    ax2.tick_params(axis="y", colors=NEG, labelsize=6.5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(NEG)
    ax.annotate("current drawdown\ndisclosed, not excluded",
                xy=(x[-1], eq.iloc[-1]), xytext=(-104, -34),
                textcoords="offset points", fontsize=6.2, color=NEG,
                ha="left",
                arrowprops=dict(arrowstyle="->", color=NEG, lw=0.7))


def ex3(ax, real_t, placebo_t):
    lo = min(placebo_t.min(), real_t) - 0.4
    hi = max(placebo_t.max(), real_t) + 0.4
    ax.hist(placebo_t, bins=np.linspace(lo, hi, 22), color=MUTE, alpha=0.7,
            edgecolor="white", lw=0.5)
    ax.set_xlim(lo, hi)
    ax.axvline(real_t, color=ACC, lw=1.8)
    ax.annotate(f"real signal\nt = {real_t:+.2f}", xy=(real_t, ax.get_ylim()[1] * 0.72),
                xytext=(-64, 0), textcoords="offset points", fontsize=6.6, color=ACC,
                arrowprops=dict(arrowstyle="->", color=ACC, lw=0.7))
    ax.set_xlabel("t-statistic", fontsize=7)
    ax.set_ylabel("shuffles", fontsize=7)
    ax.set_title("3 · Placebo: the signal, not the machinery",
                 fontsize=8.5, loc="left", weight="bold")
    ax.text(0.02, -0.235, f"{len(placebo_t)} shuffles of the signal across instruments; "
                          f"all other mechanics fixed",
            transform=ax.transAxes, fontsize=6.2, color=MUTE, va="top")


def ex4(ax, grid, costs):
    ks = sorted({k for k, _ in grid})
    vws = sorted({v for _, v in grid})
    Z = np.array([[grid[(k, v)] for v in vws] for k in ks])
    im = ax.imshow(Z, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(vws))); ax.set_xticklabels(vws, fontsize=6.5)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=6.5)
    ax.set_xlabel("volatility lookback (months)", fontsize=7)
    ax.set_ylabel("formation window (months)", fontsize=7)
    for i in range(len(ks)):
        for j in range(len(vws)):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                    color=INK)
    ax.set_title("4 · Parameter plateau, and cost sensitivity",
                 fontsize=8.5, loc="left", weight="bold")
    txt = "  ".join(f"{b}bp {s:.2f}" for b, s in costs)
    ax.text(0.0, -0.235, f"net Sharpe by cost per side:  {txt}",
            transform=ax.transAxes, fontsize=6.2, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_clean.parquet")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--out", default="EXHIBITS")
    a = ap.parse_args()

    m = load(a.prices)
    idm = idm_of(m)
    r = book(m, idm)
    yrs = len(r) / 12
    sr = sharpe(r)
    print(f"  strategy: Sharpe {sr:+.3f}  t {sr*np.sqrt(yrs):+.2f}  "
          f"{len(r)} months  {m['symbol'].nunique()} instruments")

    print("  running placebo...")
    pt = []
    for s in range(a.seeds):
        v = sharpe(book(m, idm, seed=s))
        if np.isfinite(v):
            pt.append(v * np.sqrt(yrs))
    print("  running parameter grid...")
    grid = {}
    for k in (6, 9, 12, 15):
        for vw in (3, 6, 12):
            grid[(k, vw)] = sharpe(book(m, idm, J_=k, vw=vw))
    costs = [(b, sharpe(book(m, idm, bps=b))) for b in (3, 10, 20, 40)]

    fig = plt.figure(figsize=(8.5, 10.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30,
                  left=0.175, right=0.945, top=0.885, bottom=0.135)
    print("  computing hedge proximity...")
    px = proximity(a.prices)
    ex1(fig.add_subplot(gs[0, 0]), px)
    ex2(fig.add_subplot(gs[0, 1]), r)
    ex3(fig.add_subplot(gs[1, 0]), sr * np.sqrt(yrs), np.array(pt))
    ex4(fig.add_subplot(gs[1, 1]), grid, costs)

    fig.suptitle("The Same Barrel — Curve-Residual Momentum in Commodity Futures",
                 fontsize=10.5, weight="bold", x=0.085, ha="left", y=0.978)
    fig.text(0.085, 0.943, "Supporting exhibits", fontsize=8, color=MUTE, ha="left")
    fig.text(0.085, 0.028,
             f"17 CME commodity futures, {m['ym'].min()} to {m['ym'].max()}, "
             f"{len(r)} monthly observations. All figures net of 3bp per side unless "
             f"stated. Positions are whole contracts on $450,000.\n"
             f"Exhibit 1: proximity tiers computed from daily returns; principal-component\n"
             f"and market tiers from the channel comparison. Sharpe shown where measured.",
             fontsize=6.1, color=MUTE)

    fig.savefig(f"{a.out}.pdf")
    fig.savefig(f"{a.out}.png", dpi=200)
    print(f"\n  -> {a.out}.pdf and {a.out}.png")
    print(f"  grid: {sum(1 for v in grid.values() if v > 0.35)} of {len(grid)} cells "
          f"above 0.35")
    print(f"  placebo: real t {sr*np.sqrt(yrs):+.2f}, placebo mean "
          f"{np.mean(pt):+.2f} sd {np.std(pt, ddof=1):.2f}")


if __name__ == "__main__":
    main()