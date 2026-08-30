"""
riskfigs.py — the two exhibits the fund's own house style leads with.

    python riskfigs.py --prices px_clean.parquet

WHY THESE TWO

The fund's example strategy document reports its evidence in a particular vocabulary. Its
headline exhibit is a trade-statistics table: number of trades, win rate, win/loss ratio,
profit factor, average trade before and after slippage, slippage as a share of gross profit,
and time in market. Its Monte Carlo exhibits show the distribution of drawdowns the strategy
COULD have produced, with the realised path drawn against them.

This project measured different things - placebo tests, statistical power, multiple-testing
burden - which are more rigorous but written in a different language. These two figures
translate the same result into the vocabulary the reader already uses. Nothing here is a new
claim; it is the existing return series described differently.

    FIGURE A   trade statistics, monthly, gross and net of cost
    FIGURE B   block-bootstrap distribution of maximum drawdown, with the realised
               drawdown marked and its percentile stated

THE MONTE CARLO IS A BLOCK BOOTSTRAP, NOT A RESHUFFLE

Reordering months independently would destroy the autocorrelation in the return series and
understate the drawdowns that are actually possible - a strategy whose losses cluster looks
far safer than it is once the clustering is shuffled away. Six-month blocks are resampled
with replacement so that runs of consecutive losses survive into the simulated paths. The
question the figure answers is: given returns that behave like these, how bad could the
worst peak-to-trough decline have been?
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
N_GRIDS = 21
BLOCK = 6
N_PATHS = 5000

INK, MUTE, ACC, NEG = "#1a1a1a", "#8a8a8a", "#c1440e", "#2b6a8f"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": INK, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ----------------------------------------------------------------------------------
# strategy — the frozen tranched specification, marked daily
# ----------------------------------------------------------------------------------

def load_daily(path: str) -> pd.DataFrame:
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
    df["dom"] = df.groupby(["symbol", "ym"]).cumcount()
    return df


def grid_targets(df: pd.DataFrame, offset: int, min_n: int = 6) -> pd.DataFrame:
    d = df.sort_values(["symbol", "date"]).copy()
    for leg in ("0", "1"):
        d[f"c{leg}"] = d.groupby("symbol")[f"r{leg}"].transform(
            lambda s: s.fillna(0.0).cumsum())
    snap = d[d["dom"] == offset][["symbol", "ym", "date", "c0", "c1", "settle_0"]].copy()
    if snap.empty:
        return pd.DataFrame()
    snap = snap.sort_values(["symbol", "ym"]).reset_index(drop=True)
    g = snap.groupby("symbol")
    snap["r0"] = g["c0"].diff()
    snap["r1"] = g["c1"].diff()
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
        r = s["bm"].rank()
        w = (r - r.mean()).to_numpy()
        gr = np.abs(w).sum()
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


def daily_book(df: pd.DataFrame, frames: list[pd.DataFrame], bps: float = 3.0):
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    syms = sorted(df["symbol"].unique())
    ret = df.pivot_table(index="date", columns="symbol", values="r0").reindex(
        dates, columns=syms)
    px = df.pivot_table(index="date", columns="symbol", values="settle_0").reindex(
        dates, columns=syms).ffill()
    stacks = []
    for tf in frames:
        if tf.empty:
            continue
        w = (tf.pivot_table(index="date", columns="symbol", values="target")
               .reindex(index=dates, columns=syms).ffill())
        stacks.append(w.to_numpy())
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
    gross = pd.Series(pnl / CAPITAL, index=dates[1:])
    net = pd.Series((pnl - cost) / CAPITAL, index=dates[1:])
    in_mkt = (np.abs(held).sum(axis=1) > 0)
    return net, gross, dict(in_market=float(in_mkt.mean()),
                            trades_per_month=float(trades.sum(axis=1).mean() * 21))


def max_dd(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="px_clean.parquet")
    ap.add_argument("--out", default="RISK_FIGURES")
    a = ap.parse_args()

    df = load_daily(a.prices)
    print("  building grids...")
    frames = [f for f in (grid_targets(df, o) for o in range(N_GRIDS)) if not f.empty]
    net_d, gross_d, aux = daily_book(df, frames)

    net = net_d.resample("ME").sum(); net = net[net != 0]
    gross = gross_d.resample("ME").sum(); gross = gross.reindex(net.index)

    # ---------- Figure A: trade statistics ----------
    wins, losses = net[net > 0], net[net < 0]
    gp, gl = wins.sum(), abs(losses.sum())
    yrs = len(net) / 12
    ann = net.mean() * 12
    vol = net.std(ddof=1) * np.sqrt(12)
    realised_dd = max_dd(net.to_numpy())
    cost_share = (gross.sum() - net.sum()) / gross.sum() if gross.sum() > 0 else np.nan

    # longest drawdown, in months
    eq = (1 + net).cumprod()
    peak = eq.cummax()
    under = (eq < peak * (1 - 1e-12)).to_numpy()
    longest, run = 0, 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)

    stats = [
        ("Rebalances (months)", f"{len(net)}"),
        ("Winning months", f"{(net > 0).mean():.0%}"),
        ("Average winning month", f"{wins.mean()*100:+.2f}%"),
        ("Average losing month", f"{losses.mean()*100:+.2f}%"),
        ("Win / loss ratio", f"{wins.mean()/abs(losses.mean()):.2f}"),
        ("Profit factor", f"{gp/gl:.2f}"),
        ("Annual return, gross of cost", f"{gross.mean()*12*100:+.2f}%"),
        ("Annual return, net of cost", f"{ann*100:+.2f}%"),
        ("Cost as share of gross profit", f"{cost_share:.1%}"),
        ("Annualised volatility", f"{vol*100:.1f}%"),
        ("Sharpe ratio, net", f"{ann/vol:.2f}"),
        ("Maximum drawdown", f"{realised_dd*100:.1f}%"),
        ("Longest drawdown (months)", f"{longest}"),
        ("Best month / worst month", f"{net.max()*100:+.1f}% / {net.min()*100:+.1f}%"),
        ("Contracts traded per month", f"{aux['trades_per_month']:.0f}"),
        ("Days holding a position", f"{aux['in_market']:.0%}"),
    ]

    # ---------- Figure B: block bootstrap of maximum drawdown ----------
    print("  bootstrapping drawdowns...")
    x = net.to_numpy()
    n = len(x)
    rng = np.random.default_rng(0)
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(N_PATHS, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_PATHS, -1)[:, :n]
    paths = x[idx]
    dds = np.array([max_dd(p) for p in paths])
    pct = float((dds < realised_dd).mean())

    # ---------- render ----------
    fig = plt.figure(figsize=(8.5, 5.9))
    gs = GridSpec(1, 2, figure=fig, wspace=0.24, left=0.055, right=0.965,
                  top=0.855, bottom=0.185)

    axA = fig.add_subplot(gs[0, 0]); axA.axis("off")
    axA.set_title("A \u00b7 Trade statistics, monthly", fontsize=9, loc="left",
                  weight="bold", pad=16)
    tbl = axA.table(cellText=[[k, v] for k, v in stats],
                    colWidths=[0.66, 0.34], loc="upper center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.4); tbl.scale(1, 1.32)
    for (i, j), c in tbl.get_celld().items():
        c.set_edgecolor("#cccccc"); c.set_linewidth(0.5)
        if j == 1:
            c.get_text().set_ha("right")
        if i % 2 == 1:
            c.set_facecolor("#f6f6f6")
        if stats[i][0] in ("Sharpe ratio, net", "Profit factor", "Maximum drawdown"):
            c.get_text().set_weight("bold")

    axB = fig.add_subplot(gs[0, 1])
    axB.hist(dds * 100, bins=55, color=MUTE, alpha=0.75, edgecolor="white", lw=0.4)
    axB.axvline(realised_dd * 100, color=ACC, lw=1.8)
    axB.axvline(np.percentile(dds, 5) * 100, color=NEG, lw=1.0, ls="--")
    ymax = axB.get_ylim()[1]
    axB.annotate(f"realised\n{realised_dd*100:.1f}%",
                 xy=(realised_dd * 100, ymax * 0.80), xytext=(28, 0),
                 textcoords="offset points", fontsize=7, color=ACC,
                 arrowprops=dict(arrowstyle="->", color=ACC, lw=0.8))
    axB.annotate(f"5th percentile\n{np.percentile(dds,5)*100:.1f}%",
                 xy=(np.percentile(dds, 5) * 100, ymax * 0.42), xytext=(-72, 0),
                 textcoords="offset points", fontsize=7, color=NEG,
                 arrowprops=dict(arrowstyle="->", color=NEG, lw=0.8))
    axB.set_xlabel("maximum drawdown (%)", fontsize=8)
    axB.set_ylabel("simulated paths", fontsize=8)
    axB.set_title("B \u00b7 Drawdowns the strategy could have produced",
                  fontsize=9, loc="left", weight="bold", pad=16)
    axB.text(0.02, 0.985,
             f"{N_PATHS:,} block-bootstrap paths, {BLOCK}-month blocks\n"
             f"median {np.median(dds)*100:.1f}%   worst {dds.min()*100:.1f}%\n"
             f"the realised path was worse than {pct:.0%} of simulations",
             transform=axB.transAxes, fontsize=6.8, color=INK, va="top")

    fig.suptitle("The Same Barrel \u2014 risk exhibits", fontsize=10.5, weight="bold",
                 x=0.055, ha="left", y=0.968)
    fig.text(0.055, 0.030,
             "Monthly returns of the tranched strategy, net of three basis points per side. "
             "Blocks of six months are resampled with replacement so that runs of consecutive "
             "losses survive into the\nsimulated paths; reshuffling months independently would "
             "destroy that clustering and understate the drawdowns actually possible.",
             fontsize=6.2, color=MUTE)

    fig.savefig(f"{a.out}.pdf"); fig.savefig(f"{a.out}.png", dpi=200)
    print(f"\n  -> {a.out}.pdf and {a.out}.png\n")

    print("  FIGURE A — trade statistics")
    for k, v in stats:
        print(f"    {k:34s} {v:>18s}")
    print(f"\n  FIGURE B — bootstrap drawdowns")
    print(f"    realised                     {realised_dd*100:>8.1f}%")
    print(f"    median simulated             {np.median(dds)*100:>8.1f}%")
    print(f"    5th percentile               {np.percentile(dds,5)*100:>8.1f}%")
    print(f"    worst of {N_PATHS:,} paths        {dds.min()*100:>8.1f}%")
    print(f"    realised was worse than      {pct:>8.0%} of paths")
    print()
    if pct < 0.5:
        print("  The realised drawdown is milder than the median simulation, so the")
        print("  historical path was not unusually kind. Quote the 5th percentile as the")
        print("  drawdown to plan for: it is the number a risk limit should be set against,")
        print("  not the one that happened to occur.")
    else:
        print("  The realised drawdown is worse than the median simulation. Say so plainly")
        print("  and quote it as the planning figure.")


if __name__ == "__main__":
    main()