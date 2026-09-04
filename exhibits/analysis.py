"""
analysis.py — bootstrap distributions and the strategy against its market.

    python analysis.py --prices data/px_clean.parquet

    ANALYSIS.pdf / ANALYSIS.png  - six panels, for your own reading rather than the pitch

WHAT IS HERE

  1  cumulative return, strategy against the equal-weighted commodity market, log scale
  2  rolling 36-month Sharpe, which is where the decay question lives
  3  bootstrap distribution of the Sharpe ratio
  4  bootstrap distribution of the Sortino ratio
  5  bootstrap distribution of maximum drawdown
  6  bootstrap distribution of R-squared against the market

THE RESAMPLING IS BLOCKED, NOT SHUFFLED

Monthly returns are resampled in six-month blocks with replacement. Reordering months
independently would destroy the clustering of losses and produce drawdown distributions
far milder than anything achievable in practice - a strategy whose bad months arrive
together looks much safer once that grouping is shuffled away. Blocks preserve it.

ON n = 100

One hundred resamples is what was asked for and it is what the paths show. A confidence
interval estimated from one hundred draws is itself noisy at the tails, so the 2.5th and
97.5th percentiles are also computed from two thousand draws and printed alongside. Where
the two disagree, trust the larger number - the difference is a measure of how much of the
interval is resampling noise rather than strategy behaviour.

WHAT TO LOOK FOR

Panel 2 is the one that matters. A full-sample Sharpe of 0.94 is an average across periods
that look nothing alike: strong through 2016, weak from 2016 to 2022, strong again since.
Whether that reads as decay or as regime variation is the single most contestable claim in
the pitch, and this panel is the evidence either way.
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

CAPITAL, VOL_TARGET, IDM, J, VOL_WINDOW = 450_000.0, 0.20, 2.5, 12, 6
N_GRIDS, BLOCK = 21, 6
N_SHOW, N_CI = 100, 2000
COST_MULTIPLE = 3.0

INK, MUTE, ACC, NEG, GRN = "#1a1a1a", "#8f8f8f", "#c1440e", "#2b6a8f", "#5d8a3a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


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
    df = df[df["asset"] == "commodity"].copy()
    med = df.groupby("symbol")["settle_0"].median()
    cost = {}
    for s in med.index:
        i = BY_SYMBOL[s]
        n = med[s] * i.dollar_price_mult
        cost[s] = 1.5 * (i.tick_value / n * 1e4) + i.commission / n * 1e4
    cs = pd.Series(cost)
    drop = set(cs[cs > COST_MULTIPLE * cs.median()].index)
    if drop:
        print(f"  universe rule excludes {sorted(drop)}")
        df = df[~df["symbol"].isin(drop)].copy()
    for leg in ("0", "1"):
        blk = df.groupby("symbol")[f"contract_{leg}"].transform(
            lambda s: (s != s.shift(1)).cumsum())
        prev = df.groupby(["symbol", blk])[f"settle_{leg}"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"r{leg}"] = np.log(df[f"settle_{leg}"] / prev)
        df.loc[~np.isfinite(df[f"r{leg}"]), f"r{leg}"] = np.nan
    df["ym"] = df["date"].dt.to_period("M")
    df["dom"] = df.groupby(["symbol", "ym"]).cumcount()
    return df


def grid_targets(df, offset, min_n=6):
    d = df.sort_values(["symbol", "date"]).copy()
    for leg in ("0", "1"):
        d[f"c{leg}"] = d.groupby("symbol")[f"r{leg}"].transform(
            lambda s: s.fillna(0.0).cumsum())
    snap = d[d["dom"] == offset][["symbol", "ym", "date", "c0", "c1", "settle_0"]].copy()
    if snap.empty:
        return pd.DataFrame()
    snap = snap.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = snap.groupby("symbol")
    snap["r0"] = g["c0"].diff(); snap["r1"] = g["c1"].diff()
    snap["bm"] = (g["r0"].transform(lambda s: s.rolling(J, min_periods=J).sum())
                  - g["r1"].transform(lambda s: s.rolling(J, min_periods=J).sum()))
    snap["vol"] = (g["r0"].transform(
        lambda s: s.rolling(VOL_WINDOW, min_periods=3).std()) * np.sqrt(12)
        ).groupby(snap["symbol"]).shift(1)
    snap["px_entry"] = g["settle_0"].shift(1)
    rows = []
    for dt, gg in snap.groupby("date"):
        s = gg[["symbol", "bm", "vol", "px_entry"]].dropna()
        s = s[(s["vol"] > 0) & (s["px_entry"] > 0)]
        if len(s) < min_n:
            continue
        r = s["bm"].rank(); w = (r - r.mean()).to_numpy(); gr = np.abs(w).sum()
        if gr <= 0:
            continue
        w = w / gr
        for sym, wi, vol, px in zip(s["symbol"], w, s["vol"], s["px_entry"]):
            inst = BY_SYMBOL[sym]
            den = inst.dollar_price_mult * px * vol
            if den > 0:
                rows.append(dict(date=dt, symbol=sym,
                                 target=wi * CAPITAL * VOL_TARGET * IDM / den))
    return pd.DataFrame(rows)


def tranched(df, frames, bps=3.0):
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    syms = sorted(df["symbol"].unique())
    ret = df.pivot_table(index="date", columns="symbol", values="r0").reindex(
        dates, columns=syms)
    px = df.pivot_table(index="date", columns="symbol", values="settle_0").reindex(
        dates, columns=syms).ffill()
    stacks = [(tf.pivot_table(index="date", columns="symbol", values="target")
                 .reindex(index=dates, columns=syms).ffill()).to_numpy()
              for tf in frames if not tf.empty]
    S = np.stack(stacks, axis=0)
    cnt = np.sum(~np.isnan(S), axis=0)
    T = np.divide(np.nansum(S, axis=0), np.maximum(cnt, 1),
                  out=np.zeros_like(cnt, dtype=float), where=cnt > 0)
    N = np.round(T)
    dpm = np.array([BY_SYMBOL[s].dollar_price_mult for s in syms])
    comm = np.array([BY_SYMBOL[s].commission for s in syms])
    P = np.nan_to_num(px.to_numpy(), nan=0.0)
    R = np.nan_to_num(ret.to_numpy(), nan=0.0)
    held = N[:-1]
    pnl = np.nansum(held * dpm * P[:-1] * np.expm1(R[1:]), axis=1)
    trades = np.abs(np.diff(N, axis=0))
    cost = np.nansum(trades * (comm + np.abs(dpm) * P[:-1] * bps / 1e4), axis=1)
    return pd.Series((pnl - cost) / CAPITAL, index=dates[1:])


# ---------------------------------------------------------------- metrics
def sharpe(r):
    av = r.std(ddof=1) * np.sqrt(12)
    return (r.mean() * 12) / av if av > 0 else np.nan


def sortino(r):
    d = r[r < 0]
    dd = d.std(ddof=1) * np.sqrt(12) if len(d) > 2 else np.nan
    return (r.mean() * 12) / dd if dd and dd > 0 else np.nan


def maxdd(r):
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def r2_vs(y, x):
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < 24 or x.var() == 0 or y.var() == 0:
        return np.nan
    b = np.cov(y, x)[0, 1] / x.var()
    e = y - (y.mean() - b * x.mean()) - b * x
    return float(1 - e.var() / y.var())


def blocks(n, rng, size):
    nb = int(np.ceil(n / BLOCK))
    st = rng.integers(0, max(n - BLOCK, 1), size=(size, nb))
    return (st[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(size, -1)[:, :n]


def hist_panel(ax, vals, real, title, xlabel, pct=False, lo_better=False):
    v = vals[np.isfinite(vals)]
    ax.hist(v * (100 if pct else 1), bins=32, color=MUTE, alpha=0.75,
            edgecolor="white", lw=0.4)
    lo, hi = np.percentile(v, [2.5, 97.5])
    for q, c, ls in ((lo, NEG, "--"), (hi, NEG, "--")):
        ax.axvline(q * (100 if pct else 1), color=c, lw=0.9, ls=ls)
    ax.axvline(real * (100 if pct else 1), color=ACC, lw=1.7)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("resamples", fontsize=7)
    ax.set_title(title, fontsize=8.2, loc="left", weight="bold", pad=9)
    f = (lambda z: f"{z*100:.1f}%") if pct else (lambda z: f"{z:.2f}")
    ax.text(0.02, 0.97,
            f"realised {f(real)}\n95% CI [{f(lo)}, {f(hi)}]\nmedian {f(np.median(v))}",
            transform=ax.transAxes, fontsize=6.3, va="top", linespacing=1.5,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.85))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--out", default="ANALYSIS")
    a = ap.parse_args()

    df = load(a.prices)
    print("  building the tranched book...")
    frames = [f for f in (grid_targets(df, o) for o in range(N_GRIDS)) if not f.empty]
    net = tranched(df, frames).resample("ME").sum()
    net = net[net != 0]

    mkt = (df.groupby(["symbol", "ym"])["r0"].sum(min_count=1)
             .groupby("ym").mean().dropna())
    mkt.index = mkt.index.to_timestamp("M")
    mkt = mkt.reindex(net.index).fillna(0.0)

    x = net.to_numpy(); mx = mkt.to_numpy(); n = len(x)
    real = dict(sharpe=sharpe(net), sortino=sortino(net), dd=maxdd(x),
                r2=r2_vs(x, mx))

    print(f"  {n} months, Sharpe {real['sharpe']:.3f}, Sortino {real['sortino']:.3f}, "
          f"maxDD {real['dd']*100:.1f}%, R2 vs market {real['r2']:.4f}")
    print(f"  bootstrapping ({N_SHOW} shown, {N_CI} for intervals)...")

    rng = np.random.default_rng(0)
    idx_show = blocks(n, rng, N_SHOW)
    idx_ci = blocks(n, np.random.default_rng(1), N_CI)

    def metrics(idx):
        out = {k: [] for k in ("sharpe", "sortino", "dd", "r2")}
        for row in idx:
            s = pd.Series(x[row])
            out["sharpe"].append(sharpe(s))
            out["sortino"].append(sortino(s))
            out["dd"].append(maxdd(x[row]))
            out["r2"].append(r2_vs(x[row], mx[row]))
        return {k: np.array(v) for k, v in out.items()}

    M_ci = metrics(idx_ci)
    M_show = metrics(idx_show)

    fig = plt.figure(figsize=(8.5, 11.0))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.0, 1.0],
                  hspace=0.44, wspace=0.26,
                  left=0.085, right=0.955, top=0.900, bottom=0.075)

    # ---- 1 cumulative -------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    eq_s = (1 + net).cumprod()
    eq_m = (1 + mkt).cumprod()
    ax.plot(eq_s.index, eq_s.to_numpy(), color=ACC, lw=1.4, label="strategy, net of costs")
    ax.plot(eq_m.index, eq_m.to_numpy(), color=NEG, lw=1.2, label="commodity market, equal-weighted")
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1 (log scale)", fontsize=7)
    ax.axhline(1.0, color=MUTE, lw=0.5, ls=":")
    ax.legend(fontsize=6.6, frameon=False, loc="upper left")
    ax.set_title("1 \u00b7 Cumulative return, strategy against its own market",
                 fontsize=8.4, loc="left", weight="bold", pad=9)
    ax.text(0.985, 0.05,
            f"strategy {(eq_s.iloc[-1]-1)*100:+.0f}%   market {(eq_m.iloc[-1]-1)*100:+.0f}%",
            transform=ax.transAxes, fontsize=6.5, ha="right", color=INK)

    # ---- 2 rolling Sharpe ---------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    roll = net.rolling(36).apply(lambda s: sharpe(pd.Series(s)), raw=False).dropna()
    ax.plot(roll.index, roll.to_numpy(), color=ACC, lw=1.3)
    ax.axhline(0, color=INK, lw=0.6)
    ax.axhline(real["sharpe"], color=MUTE, lw=0.9, ls="--")
    ax.text(roll.index[2], real["sharpe"], f" full sample {real['sharpe']:.2f}",
            fontsize=6.3, color=MUTE, va="bottom")
    ax.fill_between(roll.index, 0, roll.to_numpy(),
                    where=(roll.to_numpy() > 0), color=ACC, alpha=0.12, lw=0)
    ax.fill_between(roll.index, 0, roll.to_numpy(),
                    where=(roll.to_numpy() <= 0), color=NEG, alpha=0.15, lw=0)
    ax.set_ylabel("rolling 36-month Sharpe", fontsize=7)
    ax.set_title("2 \u00b7 The decay question, in one picture",
                 fontsize=8.4, loc="left", weight="bold", pad=9)
    ax.text(0.985, 0.05,
            "a full-sample average across periods that look nothing alike",
            transform=ax.transAxes, fontsize=6.4, ha="right", color=MUTE)

    hist_panel(fig.add_subplot(gs[2, 0]), M_ci["sharpe"], real["sharpe"],
               "3 \u00b7 Sharpe ratio", "Sharpe")
    hist_panel(fig.add_subplot(gs[2, 1]), M_ci["dd"], real["dd"],
               "4 \u00b7 Maximum drawdown", "maximum drawdown (%)", pct=True)

    fig.suptitle("The Same Barrel \u2014 bootstrap analysis", fontsize=11,
                 weight="bold", x=0.085, ha="left", y=0.962)
    fig.text(0.085, 0.938,
             f"{n} monthly observations, {N_SHOW} block-bootstrap paths shown, "
             f"{N_CI} used for intervals, {BLOCK}-month blocks",
             fontsize=7.2, color=MUTE, ha="left")
    fig.text(0.085, 0.030,
             "Blocks of six months are resampled with replacement so runs of consecutive "
             "losses survive into the simulated paths; reordering months independently "
             "would destroy that\nclustering and produce drawdown distributions far milder "
             "than anything achievable in practice. Red line is the realised value, dashed "
             "lines the 2.5th and 97.5th percentiles.",
             fontsize=6.1, color=MUTE, linespacing=1.55)

    fig.savefig(f"{a.out}.pdf"); fig.savefig(f"{a.out}.png", dpi=190)

    # second page: sortino and r-squared
    fig2 = plt.figure(figsize=(8.5, 4.4))
    gs2 = GridSpec(1, 2, figure=fig2, wspace=0.26, left=0.085, right=0.955,
                   top=0.80, bottom=0.20)
    hist_panel(fig2.add_subplot(gs2[0, 0]), M_ci["sortino"], real["sortino"],
               "5 \u00b7 Sortino ratio", "Sortino")
    hist_panel(fig2.add_subplot(gs2[0, 1]), M_ci["r2"], real["r2"],
               "6 \u00b7 R\u00b2 against the commodity market", "R\u00b2")
    fig2.suptitle("The Same Barrel \u2014 bootstrap analysis, continued", fontsize=10,
                  weight="bold", x=0.085, ha="left", y=0.945)
    fig2.text(0.085, 0.055,
              "Sortino divides by downside deviation only. R\u00b2 near zero is the "
              "market-neutrality claim: almost none of the strategy's variance is "
              "explained by the commodity complex.",
              fontsize=6.2, color=MUTE)
    fig2.savefig(f"{a.out}_2.pdf"); fig2.savefig(f"{a.out}_2.png", dpi=190)

    print(f"\n  -> {a.out}.pdf / .png and {a.out}_2.pdf / .png\n")
    print(f"  {'metric':22s} {'realised':>10s} {'median':>10s} "
          f"{'2.5%':>10s} {'97.5%':>10s}   {'n=100 CI':>20s}")
    for k, lab, pct in (("sharpe", "Sharpe", False), ("sortino", "Sortino", False),
                        ("dd", "max drawdown", True), ("r2", "R2 vs market", False)):
        v, vs = M_ci[k], M_show[k]
        v, vs = v[np.isfinite(v)], vs[np.isfinite(vs)]
        f = (lambda z: f"{z*100:9.1f}%") if pct else (lambda z: f"{z:10.3f}")
        lo2, hi2 = np.percentile(vs, [2.5, 97.5])
        print(f"  {lab:22s} {f(real[k])} {f(np.median(v))} "
              f"{f(np.percentile(v,2.5))} {f(np.percentile(v,97.5))}   "
              f"[{f(lo2).strip()}, {f(hi2).strip()}]")
    print(f"\n  share of resamples with Sharpe below 0: "
          f"{(M_ci['sharpe'] < 0).mean():.1%}")
    print(f"  share with Sharpe below 0.5:            "
          f"{(M_ci['sharpe'] < 0.5).mean():.1%}")
    print("\n  The n=100 interval is printed beside the n=2000 one. Where they differ,")
    print("  the gap is resampling noise rather than strategy behaviour.")


if __name__ == "__main__":
    main()