"""
concept_figure.py — the figure that explains the idea: level versus shape.

    python exhibits/concept_figure.py

Writes docs/figures/concept.png — Figure 1 of the pitch.

Panels A and B are labelled schematics of a forward curve. They carry no data and
claim none; they exist to make the level-versus-shape distinction visible.

Panel C is real: the actual cross-section for one month, read from
data/derived/signal_ranks.csv, showing all sixteen traded instruments.

An earlier version of this figure was built from a narrow price file that carried only
twelve of the sixteen, while its title said sixteen. This version reads the committed
derived data, so the count in the title is the count on the axis, always.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DERIVED, OUT = Path("data/derived"), Path("docs/figures")
MONTH = "2014-12"          # mid oil-collapse; wide, readable dispersion

NAVY, ORANGE, GREY = "#1F3864", "#E2571E", "#8A8A8A"
NAME = {"ZS": "Soybeans", "SIL": "Silver", "ZW": "Wheat", "ZC": "Corn", "ZL": "Soy oil",
        "KE": "KC wheat", "LE": "Live cattle", "MGC": "Gold", "MHG": "Copper",
        "ZM": "Soy meal", "MCL": "Crude", "HE": "Lean hogs", "PL": "Platinum",
        "PA": "Palladium", "RB": "RBOB", "HO": "Heating oil"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.edgecolor": "#BDBDBD", "axes.linewidth": 0.7,
                     "xtick.color": "#555", "ytick.color": "#555"})

M = np.arange(1, 7)
OLD = np.array([100, 102, 104, 105.5, 106.5, 107.0])


def curve_panel(ax, new, title, colour, note, readout):
    ax.plot(M, OLD, color=GREY, lw=1.4, ls=(0, (4, 2)), marker="o", ms=3.4,
            mfc="white", label="curve 12 months ago")
    ax.plot(M, new, color=colour, lw=1.8, marker="o", ms=3.4, label="curve today")
    for xi, y0, y1 in ((1, OLD[0], new[0]), (2, OLD[1], new[1])):
        ax.annotate("", xy=(xi, y1), xytext=(xi, y0),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.3,
                                    shrinkA=2, shrinkB=2))
    ax.scatter([1, 2], [new[0], new[1]], s=58, facecolor="none",
               edgecolor=colour, lw=1.5, zorder=5)
    ax.set_xticks(M)
    ax.set_xticklabels(["M1", "M2", "M3", "M4", "M5", "M6"], fontsize=7.5)
    ax.set_xlabel("delivery month", fontsize=8, color="#777", labelpad=1)
    ax.set_ylim(77, 116)
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=9.5, color=NAVY,
                 fontweight="bold", pad=6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left", handlelength=1.6,
              borderpad=0.1, labelspacing=0.25)
    ax.text(3.55, 78.5, readout, fontsize=8.4, color=colour, fontweight="bold",
            ha="left", va="bottom", linespacing=1.4)
    ax.text(0.5, -0.27, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.2, color="#444", linespacing=1.35)


def main() -> None:
    path = DERIVED / "signal_ranks.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `make derived` first.")

    sig = pd.read_csv(path)
    snap = sig[sig["month"] == MONTH].sort_values("signal", ascending=False)
    if snap.empty:
        raise SystemExit(f"no rows for {MONTH} in {path}")
    n_inst = len(snap)

    fig = plt.figure(figsize=(10.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.70, wspace=0.22,
                          left=0.075, right=0.975, top=0.795, bottom=0.165)

    curve_panel(fig.add_subplot(gs[0, 0]), OLD - 12,
                "A · The whole curve moves", NAVY,
                "Only the first two contracts enter the signal.\n"
                "Here both fell alike, so the signal is zero.",
                "M1  −12%\nM2  −12%\nsignal   0")

    curve_panel(fig.add_subplot(gs[0, 1]),
                np.array([106, 98, 96, 95, 94.5, 94.0]),
                "B · The curve changes shape", ORANGE,
                "The near contract rose while the far one fell —\n"
                "the market is short of the physical. That is a signal.",
                "M1   +6%\nM2   −4%\nsignal  +10")

    ax = fig.add_subplot(gs[1, :])
    colours = [ORANGE if w > 0 else NAVY for w in snap["weight"]]
    xs = np.arange(n_inst)
    ax.bar(xs, snap["signal"] * 100, color=colours, width=0.66)
    ax.axhline(0, color="#999", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([NAME.get(s, s) for s in snap["symbol"]], fontsize=7.4,
                       rotation=32, ha="right", rotation_mode="anchor")
    ax.set_ylabel("12-month change in the\nnear-to-far gap (%)", fontsize=8.2,
                  color="#555", linespacing=1.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    lo, hi = (snap["signal"] * 100).min(), (snap["signal"] * 100).max()
    ax.set_ylim(min(lo * 1.5, -2), hi * 1.42)
    n_long = int((snap["weight"] > 0).sum())
    ax.set_title(f"C · Every month we rank all {n_inst} and trade the spread "
                 f"between them — {MONTH}",
                 loc="left", fontsize=9.5, color=NAVY, fontweight="bold", pad=8)
    ax.text(0.015, 0.95, f"LONG  the top {n_long} by rank", transform=ax.transAxes,
            color=ORANGE, fontsize=8.6, fontweight="bold", va="top")
    ax.text(0.985, 0.95, f"SHORT  the bottom {n_inst - n_long}",
            transform=ax.transAxes, color=NAVY, fontsize=8.6,
            fontweight="bold", ha="right", va="top")

    fig.suptitle("We trade the shape of the curve, not the level of the price",
                 x=0.075, ha="left", fontsize=13, color=NAVY, fontweight="bold",
                 y=0.965)
    fig.text(0.075, 0.905,
             "Panels A and B are schematics of a forward curve. Panel C is the real "
             "cross-section, read from data/derived/signal_ranks.csv.",
             fontsize=8.2, color="#555", ha="left")
    fig.text(0.075, 0.022,
             "Position size comes from each commodity's RANK, not the height of its "
             "bar, and the ranks sum to zero — so the book is long and short at all "
             "times.",
             fontsize=8.0, color="#444", ha="left", va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "concept.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}/concept.png  ({n_inst} instruments, {MONTH}, "
          f"{n_long} long / {n_inst - n_long} short)")


if __name__ == "__main__":
    main()
