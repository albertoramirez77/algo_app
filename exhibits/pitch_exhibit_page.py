"""
pitch_exhibit_page.py — the one appendix page of figures for the AlgoGators submission.

    python exhibits/pitch_exhibit_page.py        (or: make exhibit-page)

Produces docs/figures/EXHIBIT_PAGE.pdf and .png, sized for US Letter portrait so it
drops straight into the pitch as the single additional page the guidelines allow.

EVERY panel is computed from data/derived/*.csv — the same committed series that
`make verify` checks — with one exception, stated on the page itself: panel G quotes the
channel comparison from docs/FINAL_NUMBERS.txt, because that analysis needs the price
file and is not reproducible from the derived series alone.

Nothing here is estimated, illustrative, or drawn by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DERIVED, OUT = Path("data/derived"), Path("docs/figures")

NAVY, ORANGE, GREY, PALE = "#1F3864", "#E2571E", "#8A8A8A", "#C9D3E4"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.4,
    "axes.edgecolor": "#BDBDBD", "axes.linewidth": 0.6,
    "xtick.color": "#555", "ytick.color": "#555", "axes.labelcolor": "#333",
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
})

REGIMES = [("2011-06", "2011-12", "2011\npeak"),
           ("2012-01", "2014-06", "2012–14\ndecline"),
           ("2014-07", "2016-02", "2014–16\noil"),
           ("2016-03", "2020-01", "2016–20\nquiet"),
           ("2020-02", "2020-12", "2020\nCOVID"),
           ("2021-01", "2022-06", "2021–22\ninflat."),
           ("2022-07", "2026-08", "2022–26\nnormal.")]

# From docs/FINAL_NUMBERS.txt, section 4. Needs the price file; not reproducible from
# the derived series, so it is quoted rather than recomputed, and labelled as such.
CHANNELS = [("Deferred contract,\nsame commodity", 93.6, 1),
            ("8 principal comps,\nother commodities", 47.5, 8),
            ("Sector peers", 44.3, 1),
            ("5 principal comps,\nother commodities", 41.4, 5),
            ("Equal-weighted\ncommodity market", 18.5, 1)]

BLOCK, N_BOOT, SEED = 6, 2000, 42


def sharpe(x: np.ndarray) -> float:
    v = x.std(ddof=1) * np.sqrt(12)
    return float(x.mean() * 12 / v) if v > 0 else np.nan


def max_dd(x: np.ndarray) -> float:
    eq = np.cumprod(1 + x)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def longest_underwater(eq: pd.Series):
    under = eq < eq.cummax()
    best = cur = 0
    start = best_start = best_end = None
    for i, f in enumerate(under):
        if f:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start, best_end = cur, start, i
        else:
            cur = 0
    return (best, eq.index[best_start], eq.index[best_end]) if best else (0, None, None)


def block_bootstrap(r: np.ndarray):
    rng = np.random.default_rng(SEED)
    n, nb = len(r), int(np.ceil(len(r) / BLOCK))
    srs, dds = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK, size=nb)
        path = np.concatenate([r[s:s + BLOCK] for s in starts])[:n]
        srs[i], dds[i] = sharpe(path), max_dd(path)
    return srs, dds


def main() -> None:
    pnl = pd.read_csv(DERIVED / "monthly_pnl.csv", index_col=0, parse_dates=True)
    pnl.index = pnl.index.to_period("M")
    bench = pd.read_csv(DERIVED / "benchmarks.csv", index_col=0)
    bench.index = pd.PeriodIndex(bench.index, freq="M")

    r = pnl["net"]
    rv = r.to_numpy()
    eq = (1 + r).cumprod()
    mkt = bench["equal_weighted_complex"].reindex(r.index).fillna(0.0)
    eqm = (1 + mkt).cumprod()
    x = r.index.to_timestamp()
    dd = (eq / eq.cummax() - 1) * 100
    n_uw, uw0, uw1 = longest_underwater(eq)
    srs, dds = block_bootstrap(rv)

    fig = plt.figure(figsize=(8.0, 10.3))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.30, 0.85, 1.05, 1.05, 0.90],
                          hspace=0.68, wspace=0.26,
                          left=0.085, right=0.965, top=0.845, bottom=0.118)

    # ---------- A: equity ------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.plot(x, eq.values, color=NAVY, lw=1.6, label="The Same Barrel, net of costs")
    ax.plot(x, eqm.values, color=GREY, lw=1.2, ls=(0, (4, 2)),
            label="equal-weighted commodity market")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("growth of $1 (log)")
    ax.axhline(1, color="#DDD", lw=0.7)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("A · Cumulative return against the market it trades",
                 loc="left", fontsize=8.6, color=NAVY, fontweight="bold", pad=4)
    ax.annotate(f"${eq.iloc[-1]:,.2f}", xy=(x[-1], eq.iloc[-1]), xytext=(-2, 6),
                textcoords="offset points", ha="right", color=NAVY,
                fontweight="bold", fontsize=7.4)
    ax.annotate(f"${eqm.iloc[-1]:,.2f}", xy=(x[-1], eqm.iloc[-1]), xytext=(-2, -11),
                textcoords="offset points", ha="right", color=GREY, fontsize=7)

    # ---------- B: underwater --------------------------------------------
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(x, dd.values, 0, color=NAVY, alpha=0.20, lw=0)
    ax2.plot(x, dd.values, color=NAVY, lw=0.9)
    if uw0 is not None:
        ax2.axvspan(uw0.to_timestamp(), uw1.to_timestamp(), color=ORANGE,
                    alpha=0.13, lw=0)
        mid = x[(x >= uw0.to_timestamp()) & (x <= uw1.to_timestamp())]
        ax2.annotate(f"{n_uw} months underwater",
                     xy=(mid[len(mid) // 2], dd.min() * 0.34), ha="center",
                     va="center", fontsize=7, color=ORANGE, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                               edgecolor="none", alpha=0.9))
    ax2.set_ylabel("drawdown (%)")
    ax2.set_ylim(dd.min() * 1.18, 2)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.set_title(f"B · Every drawdown, and the longest one  "
                  f"(worst {dd.min():.1f}%, longest {n_uw} months)",
                  loc="left", fontsize=8.6, color=NAVY, fontweight="bold", pad=4)

    # ---------- C: regimes ------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    labs, sr, mr = [], [], []
    for a, b, lab in REGIMES:
        w = (r.index >= pd.Period(a, "M")) & (r.index <= pd.Period(b, "M"))
        if w.sum():
            labs.append(lab); sr.append(r[w].mean() * 1200); mr.append(mkt[w].mean() * 1200)
    i3 = np.arange(len(labs))
    ax3.bar(i3 - 0.19, sr, 0.38, color=NAVY, label="strategy")
    ax3.bar(i3 + 0.19, mr, 0.38, color=PALE, label="market")
    ax3.axhline(0, color="#999", lw=0.7)
    ax3.set_xticks(i3); ax3.set_xticklabels(labs, fontsize=5.5, linespacing=1.1)
    ax3.set_ylabel("annualised return (%)")
    ax3.set_ylim(min(min(sr), min(mr)) * 1.2, max(max(sr), max(mr)) * 1.40)
    ax3.legend(frameon=False, fontsize=6.2, ncols=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.03), handlelength=1.1, columnspacing=1.0)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    ax3.set_title("C · Return by regime", loc="left", fontsize=8.6,
                  color=NAVY, fontweight="bold", pad=4)

    # ---------- D: market neutrality --------------------------------------
    ax4 = fig.add_subplot(gs[2, 1])
    j = pd.DataFrame({"s": r, "m": mkt}).dropna()
    j = j[j["m"] != 0]
    ax4.scatter(j["m"] * 100, j["s"] * 100, s=8, color=NAVY, alpha=0.45, lw=0)
    ax4.axhline(0, color="#CCC", lw=0.7); ax4.axvline(0, color="#CCC", lw=0.7)
    b1, b0 = np.polyfit(j["m"], j["s"], 1)
    xs = np.linspace(j["m"].min(), j["m"].max(), 20)
    ax4.plot(xs * 100, (b0 + b1 * xs) * 100, color=ORANGE, lw=1.3)
    r2 = np.corrcoef(j["m"], j["s"])[0, 1] ** 2
    ax4.set_xlabel("commodity market, monthly (%)", fontsize=6.8)
    ax4.set_ylabel("strategy, monthly (%)", fontsize=6.8)
    for s in ("top", "right"):
        ax4.spines[s].set_visible(False)
    ax4.set_title("D · Market neutrality", loc="left", fontsize=8.6,
                  color=NAVY, fontweight="bold", pad=4)
    ax4.text(0.03, 0.96, f"R² = {r2:.2f}\n{len(j)} months", transform=ax4.transAxes,
             va="top", fontsize=6.8, color="#444", linespacing=1.4)

    # ---------- E / F: bootstrap ------------------------------------------
    real_sr, real_dd = sharpe(rv), max_dd(rv) * 100
    for k, (ax_, data, real, lab, ttl) in enumerate([
            (fig.add_subplot(gs[3, 0]), srs, real_sr, "Sharpe ratio",
             "E · Bootstrap: Sharpe ratio"),
            (fig.add_subplot(gs[3, 1]), dds * 100, real_dd, "maximum drawdown (%)",
             "F · Bootstrap: maximum drawdown")]):
        ax_.hist(data, bins=45, color=PALE, edgecolor="white", linewidth=0.3)
        lo, hi = np.percentile(data, [2.5, 97.5])
        ax_.axvline(real, color=ORANGE, lw=1.6)
        for q in (lo, hi):
            ax_.axvline(q, color=NAVY, lw=0.9, ls=(0, (3, 2)))
        ax_.set_xlabel(lab, fontsize=6.8)
        ax_.set_ylabel("resampled paths", fontsize=6.8)
        for s in ("top", "right", "left"):
            ax_.spines[s].set_visible(False)
        ax_.set_yticks([])
        ax_.set_title(ttl, loc="left", fontsize=8.6, color=NAVY,
                      fontweight="bold", pad=4)
        note = (f"realised {real:.2f}\nmedian {np.median(data):.2f}\n"
                f"95% [{lo:.2f}, {hi:.2f}]") if k == 0 else \
               (f"realised {real:.1f}%\nmedian {np.median(data):.1f}%\n"
                f"5th pct {np.percentile(data, 5):.1f}%")
        ax_.text(0.97 if k == 0 else 0.03, 0.96, note, transform=ax_.transAxes,
                 ha="right" if k == 0 else "left", va="top", fontsize=6.6,
                 color="#444", linespacing=1.45)

    # ---------- G: channels ------------------------------------------------
    ax5 = fig.add_subplot(gs[4, :])
    labels = [c[0] for c in CHANNELS][::-1]
    vals = [c[1] for c in CHANNELS][::-1]
    regs = [c[2] for c in CHANNELS][::-1]
    cols = [ORANGE if v == 93.6 else PALE for v in vals]
    y5 = np.arange(len(vals))
    ax5.barh(y5, vals, color=cols, height=0.60)
    for yi, v, g in zip(y5, vals, regs):
        ax5.text(v + 1.2, yi, f"{v:.1f}%   ({g} regressor{'s' if g > 1 else ''})",
                 va="center", fontsize=6.6,
                 color=NAVY if v == 93.6 else "#555",
                 fontweight="bold" if v == 93.6 else "normal")
    ax5.set_yticks(y5); ax5.set_yticklabels(labels, fontsize=6.0, linespacing=1.15)
    ax5.get_yticklabels()[-1].set_color(NAVY)
    ax5.get_yticklabels()[-1].set_fontweight("bold")
    ax5.set_xlim(0, 128); ax5.set_xticks([0, 25, 50, 75, 100])
    ax5.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax5.set_xlabel("variance of the common component removed, leave-one-out",
                   fontsize=6.8)
    for s in ("top", "right", "left"):
        ax5.spines[s].set_visible(False)
    ax5.tick_params(axis="y", length=0)
    ax5.set_title("G · Why the deferred contract is the right hedge",
                  loc="left", fontsize=8.6, color=NAVY, fontweight="bold", pad=4)
    pos = ax5.get_position()
    ax5.set_position([0.175, pos.y0, pos.x1 - 0.175, pos.height])

    # ---------- header / footer -------------------------------------------
    fig.suptitle("The Same Barrel — supporting exhibits", x=0.085, ha="left",
                 fontsize=13, color=NAVY, fontweight="bold", y=0.977)
    fig.text(0.085, 0.949,
             f"Sixteen CME commodity futures · {r.index[0]} to {r.index[-1]} "
             f"({len(r)} months) · $450,000 · whole contracts · net of bottom-up costs",
             fontsize=7.2, color="#555", ha="left")
    stats = [("Sharpe, net", f"{real_sr:.2f}"),
             ("annual return", f"{r.mean() * 1200:.1f}%"),
             ("volatility", f"{r.std(ddof=1) * np.sqrt(12) * 100:.1f}%"),
             ("max drawdown", f"{real_dd:.1f}%"),
             ("longest DD", f"{n_uw} mo"),
             ("R² vs market", f"{r2:.2f}")]
    for i, (k, v) in enumerate(stats):
        xp = 0.085 + i * 0.149
        fig.text(xp, 0.907, v, fontsize=10.5, color=NAVY, fontweight="bold", ha="left")
        fig.text(xp, 0.890, k, fontsize=6.4, color="#666", ha="left")

    fig.text(0.085, 0.016,
             "Panels A–F are computed from data/derived/monthly_pnl.csv and "
             "benchmarks.csv, the committed series that `make verify` re-checks against "
             "the headline table.\nPanels E and F resample 6-month blocks 2,000 times, "
             "preserving runs of consecutive losing months; reshuffling months "
             "independently would understate the drawdowns actually possible.\n"
             "Panel G is quoted from docs/FINAL_NUMBERS.txt section 4: it requires the "
             "price file and is not reproducible from the derived series alone.\n"
             "Regime boundaries in panel C are the labels used in the text; every value "
             "inside them is computed. Repository: github.com/albertoramirez77/algo_app",
             fontsize=5.9, color="#777", ha="left", va="bottom", linespacing=1.55)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "EXHIBIT_PAGE.png", dpi=220, facecolor="white")
    fig.savefig(OUT / "EXHIBIT_PAGE.pdf", facecolor="white")
    plt.close(fig)

    print(f"  wrote {OUT}/EXHIBIT_PAGE.png and .pdf")
    print(f"    Sharpe {real_sr:.3f} · max DD {real_dd:.1f}% · {n_uw} months underwater")
    print(f"    bootstrap Sharpe 95% [{np.percentile(srs, 2.5):.2f}, "
          f"{np.percentile(srs, 97.5):.2f}], median max DD "
          f"{np.median(dds) * 100:.1f}%, 5th pct {np.percentile(dds, 5) * 100:.1f}%")
    print(f"    R² vs commodity market {r2:.3f}")


if __name__ == "__main__":
    main()
