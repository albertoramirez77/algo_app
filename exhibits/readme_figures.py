"""
readme_figures.py — the two overview figures on the front page of the README.

    python exhibits/readme_figures.py          (or: make figures)

Reads ONLY the committed files in data/derived/. No vendor data, no API key, no
Databento subscription. Anyone who clones this repository can regenerate both figures
and confirm they follow from the same series that `make verify` checks.

Writes:
    docs/figures/strategy_overview.png   what the strategy earned, and where
    docs/figures/how_it_works.png        how a position is chosen
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DERIVED = Path("data/derived")
OUT = Path("docs/figures")

NAVY, ORANGE, GREY, PALE = "#1F3864", "#E2571E", "#8A8A8A", "#C9D3E4"
POS, NEG = "#2E7D32", "#B23A2E"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#BDBDBD", "axes.linewidth": 0.7,
    "xtick.color": "#555", "ytick.color": "#555", "axes.labelcolor": "#333",
})

# Regime boundaries are a labelling choice made in the pitch; every number shown
# inside them is computed from the committed series.
REGIMES = [
    ("2011-06", "2011-12", "2011\npeak"),
    ("2012-01", "2014-06", "2012–14\ndecline"),
    ("2014-07", "2016-02", "2014–16\noil collapse"),
    ("2016-03", "2020-01", "2016–20\nrange-bound"),
    ("2020-02", "2020-12", "2020\nCOVID"),
    ("2021-01", "2022-06", "2021–22\ninflation"),
    ("2022-07", "2026-08", "2022–26\nnormalisation"),
]

NAME = {"ZS": "Soybeans", "SIL": "Silver", "ZW": "Wheat", "ZC": "Corn", "ZL": "Soy oil",
        "KE": "KC wheat", "LE": "Live cattle", "MGC": "Gold", "MHG": "Copper",
        "ZM": "Soy meal", "MCL": "Crude", "HE": "Lean hogs", "PL": "Platinum",
        "PA": "Palladium", "RB": "RBOB", "HO": "Heating oil", "QG": "Nat gas"}


def load():
    pnl = pd.read_csv(DERIVED / "monthly_pnl.csv", index_col=0, parse_dates=True)
    pnl.index = pnl.index.to_period("M")
    bench = pd.read_csv(DERIVED / "benchmarks.csv", index_col=0)
    bench.index = pd.PeriodIndex(bench.index, freq="M")
    costs = pd.read_csv(DERIVED / "cost_table.csv")
    sig = pd.read_csv(DERIVED / "signal_ranks.csv")
    contrib = pd.read_csv(DERIVED / "pnl_by_instrument.csv")
    return pnl, bench, costs, sig, contrib


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    v = r.std(ddof=1) * np.sqrt(12)
    return float((r.mean() * 12) / v) if v > 0 else np.nan


def longest_underwater(equity: pd.Series):
    """Return (n_months, start, end) of the longest run below a previous peak."""
    under = equity < equity.cummax()
    best = cur = 0
    best_end = cur_start = None
    for i, flag in enumerate(under):
        if flag:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best:
                best, best_end, best_start = cur, i, cur_start
        else:
            cur = 0
    if best == 0:
        return 0, None, None
    return best, equity.index[best_start], equity.index[best_end]


# ======================================================================== FIG 1
def figure_one(pnl, bench):
    r = pnl["net"]
    eq = (1 + r).cumprod()
    mkt = bench["equal_weighted_complex"].reindex(r.index).fillna(0.0)
    eqm = (1 + mkt).cumprod()
    x = r.index.to_timestamp()

    fig = plt.figure(figsize=(11.0, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.52, wspace=0.24,
                          left=0.065, right=0.975, top=0.815, bottom=0.155)

    # --- A: growth of $1 ------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.plot(x, eq.values, color=NAVY, lw=1.9, label="The Same Barrel, net of costs")
    ax.plot(x, eqm.values, color=GREY, lw=1.4, ls=(0, (4, 2)),
            label="equal-weighted commodity market")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("growth of $1  (log scale)")
    ax.axhline(1, color="#DDD", lw=0.8)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("A · $1 invested, against the market it trades",
                 loc="left", fontsize=10, color=NAVY, fontweight="bold", pad=6)
    ax.annotate(f"${eq.iloc[-1]:,.2f}", xy=(x[-1], eq.iloc[-1]),
                xytext=(-2, 8), textcoords="offset points",
                color=NAVY, fontweight="bold", fontsize=9, ha="right")
    ax.annotate(f"${eqm.iloc[-1]:,.2f}", xy=(x[-1], eqm.iloc[-1]),
                xytext=(-2, -14), textcoords="offset points",
                color=GREY, fontsize=8.5, ha="right")

    # --- B: underwater --------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    dd = (eq / eq.cummax() - 1) * 100
    ax2.fill_between(x, dd.values, 0, color=NAVY, alpha=0.20, lw=0)
    ax2.plot(x, dd.values, color=NAVY, lw=1.1)
    n_uw, uw_start, uw_end = longest_underwater(eq)
    if uw_start is not None:
        ax2.axvspan(uw_start.to_timestamp(), uw_end.to_timestamp(),
                    color=ORANGE, alpha=0.13, lw=0)
        mid = x[(x >= uw_start.to_timestamp()) & (x <= uw_end.to_timestamp())]
        ax2.annotate(f"{n_uw} months underwater",
                     xy=(mid[len(mid) // 2], dd.min() * 0.30),
                     ha="center", va="center", fontsize=8.2, color=ORANGE,
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                               edgecolor="none", alpha=0.88))
    ax2.set_ylabel("drawdown (%)")
    ax2.set_ylim(dd.min() * 1.22, 2)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.set_title(f"B · Every loss, and the longest one  (worst {dd.min():.1f}%)",
                  loc="left", fontsize=10, color=NAVY, fontweight="bold", pad=6)

    # --- C: regimes -----------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    labs, s_ret, m_ret = [], [], []
    for a, b, lab in REGIMES:
        w = (r.index >= pd.Period(a, "M")) & (r.index <= pd.Period(b, "M"))
        if w.sum() == 0:
            continue
        labs.append(lab)
        s_ret.append(r[w].mean() * 12 * 100)
        m_ret.append(mkt[w].mean() * 12 * 100)
    idx = np.arange(len(labs))
    ax3.bar(idx - 0.19, s_ret, width=0.38, color=NAVY, label="strategy")
    ax3.bar(idx + 0.19, m_ret, width=0.38, color=PALE, label="commodity market")
    ax3.axhline(0, color="#999", lw=0.8)
    ax3.set_xticks(idx)
    ax3.set_xticklabels(labs, fontsize=6.6, linespacing=1.15)
    ax3.set_ylabel("annualised return (%)")
    lo, hi = min(min(s_ret), min(m_ret)), max(max(s_ret), max(m_ret))
    ax3.set_ylim(lo * 1.18, hi * 1.34)
    ax3.legend(frameon=False, fontsize=7.4, loc="upper center", ncols=2,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.3, columnspacing=1.2)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    ax3.set_title("C · The two best regimes are the market's two worst",
                  loc="left", fontsize=10, color=NAVY, fontweight="bold", pad=6)

    # --- header ---------------------------------------------------------------
    eq_full = eq
    stats = [("Sharpe, net", f"{sharpe(r):.2f}"),
             ("annual return", f"{r.mean() * 12 * 100:.1f}%"),
             ("volatility", f"{r.std(ddof=1) * np.sqrt(12) * 100:.1f}%"),
             ("max drawdown", f"{dd.min():.1f}%"),
             ("longest drawdown", f"{n_uw} mo"),
             ("months", f"{len(r)}")]
    fig.suptitle("The Same Barrel — curve-residual momentum in commodity futures",
                 x=0.065, ha="left", fontsize=13.5, color=NAVY, fontweight="bold",
                 y=0.972)
    fig.text(0.065, 0.930,
             f"Sixteen CME contracts · {r.index[0]} to {r.index[-1]} · $450,000 · whole "
             f"contracts · net of bottom-up costs",
             fontsize=8.4, color="#555", ha="left")
    for i, (k, v) in enumerate(stats):
        xp = 0.065 + i * 0.156
        fig.text(xp, 0.885, v, fontsize=13, color=NAVY, fontweight="bold", ha="left")
        fig.text(xp, 0.858, k, fontsize=7.6, color="#666", ha="left")

    fig.text(0.065, 0.018,
             "Computed from data/derived/monthly_pnl.csv and benchmarks.csv — the same "
             "committed series that `make verify` checks.\nRegime boundaries are the "
             "labels used in the pitch; every value inside them is computed from the data.",
             fontsize=6.9, color="#777", ha="left", linespacing=1.4, va="bottom")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "strategy_overview.png", dpi=200, facecolor="white")
    plt.close(fig)
    return dict(sharpe=sharpe(r), n_uw=n_uw, dd=dd.min(), final=eq_full.iloc[-1])


# ======================================================================== FIG 2
def figure_two(pnl, bench, costs, sig, contrib):
    fig = plt.figure(figsize=(11.0, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.02], wspace=0.36,
                          left=0.070, right=0.972, top=0.700, bottom=0.215)

    # --- A: the universe rule -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    c = costs.sort_values("cost_bp")
    med = c.loc[~c["excluded"], "cost_bp"].median()
    thr = 3 * med
    cols = [ORANGE if e else NAVY for e in c["excluded"]]
    y = np.arange(len(c))
    ax.barh(y, c["cost_bp"], color=cols, height=0.62)
    ax.axvline(thr, color=NEG, ls=(0, (3, 2)), lw=1.1)
    ax.text(thr, 0.4, f"  3× median = {thr:.1f}bp",
            color=NEG, fontsize=7.4, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([NAME.get(s, s) for s in c["symbol"]], fontsize=7.2)
    ax.set_xlabel("round-trip cost (bp of notional)", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("A · The universe rule, set before any return",
                 loc="left", fontsize=9.6, color=NAVY, fontweight="bold", pad=6)
    ex = c[c["excluded"]]
    if len(ex):
        ax.annotate("EXCLUDED", xy=(ex["cost_bp"].iloc[0], y[-1]),
                    xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7.6,
                    color="white", fontweight="bold")

    # --- B: one month's cross-section ----------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    q = sig.groupby("month")["signal"].quantile([0.25, 0.75]).unstack()
    month = (q[0.75] - q[0.25]).idxmax()
    s = sig[sig["month"] == month].sort_values("signal", ascending=False)
    cols = [ORANGE if w > 0 else NAVY for w in s["weight"]]
    xs = np.arange(len(s))
    ax2.bar(xs, s["signal"] * 100, color=cols, width=0.68)
    ax2.axhline(0, color="#999", lw=0.8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([NAME.get(x, x) for x in s["symbol"]],
                        rotation=90, fontsize=6.8)
    ax2.set_ylabel("12-month change in the\nnear-to-far gap (%)", fontsize=8,
                   linespacing=1.25)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.set_title(f"B · One month's ranking  ({month})",
                  loc="left", fontsize=9.6, color=NAVY, fontweight="bold", pad=6)
    lo2, hi2 = (s["signal"] * 100).min(), (s["signal"] * 100).max()
    ax2.set_ylim(lo2 * 1.55 if lo2 < 0 else -1, hi2 * 1.30)
    ax2.text(0.02, 0.97, "LONG  the top half by rank", transform=ax2.transAxes,
             color=ORANGE, fontsize=7.6, fontweight="bold", va="top")
    ax2.text(0.98, 0.62, "SHORT  the bottom half", transform=ax2.transAxes,
             color=NAVY, fontsize=7.6, fontweight="bold", ha="right", va="bottom")

    # --- C: breadth -----------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    tot = (contrib.groupby("symbol")["contribution"].sum() * 100).sort_values()
    cols = [NEG if v < 0 else POS for v in tot]
    y3 = np.arange(len(tot))
    ax3.barh(y3, tot.values, color=cols, height=0.62, alpha=0.85)
    ax3.axvline(0, color="#999", lw=0.8)
    ax3.set_yticks(y3)
    ax3.set_yticklabels([NAME.get(s, s) for s in tot.index], fontsize=7.2)
    ax3.set_xlabel("cumulative return with the instrument\nminus without it (percentage points)",
                   fontsize=7.6, linespacing=1.3)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    ax3.tick_params(axis="y", length=0)
    ax3.set_title("C · Leave-one-out by instrument",
                  loc="left", fontsize=9.6, color=NAVY, fontweight="bold", pad=6)

    fig.suptitle("How a position gets chosen", x=0.070, ha="left",
                 fontsize=13.5, color=NAVY, fontweight="bold", y=0.980)
    fig.text(0.070, 0.872,
             "Each month: drop anything too expensive to trade, rank the rest on how far "
             "their near-to-far gap has moved over twelve months,\n"
             "buy the tightening curves and sell the loosening ones in whole contracts.",
             fontsize=8.4, color="#555", ha="left", linespacing=1.4)
    fig.text(0.070, 0.028,
             "Computed from data/derived/cost_table.csv, signal_ranks.csv and "
             "pnl_by_instrument.csv.\nPanel B shows the month with the widest "
             "interquartile spread of the signal.",
             fontsize=6.9, color="#777", ha="left", linespacing=1.4, va="bottom")
    fig.savefig(OUT / "how_it_works.png", dpi=200, facecolor="white")
    plt.close(fig)
    return month


def main() -> None:
    missing = [f for f in ("monthly_pnl.csv", "benchmarks.csv", "cost_table.csv",
                           "signal_ranks.csv", "pnl_by_instrument.csv")
               if not (DERIVED / f).exists()]
    if missing:
        raise SystemExit(f"missing committed files in {DERIVED}/: {missing}\n"
                         f"run `make derived PRICES=<path to px_clean.parquet>` first.")

    pnl, bench, costs, sig, contrib = load()
    s = figure_one(pnl, bench)
    month = figure_two(pnl, bench, costs, sig, contrib)

    print(f"  wrote {OUT}/strategy_overview.png")
    print(f"        Sharpe {s['sharpe']:.3f}, max drawdown {s['dd']:.1f}%, "
          f"{s['n_uw']} months underwater, $1 -> ${s['final']:.2f}")
    print(f"  wrote {OUT}/how_it_works.png   (cross-section panel: {month})")
    print("\n  Both figures are computed from data/derived/ alone — no vendor data.")


if __name__ == "__main__":
    main()
