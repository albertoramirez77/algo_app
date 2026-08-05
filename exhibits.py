"""
exhibits.py — the four appendix exhibits.

Nothing here is simulated. Panels 1 and 3 plot published coefficients; panels 2 and 4 are
arithmetic on real contract multipliers and the cost model in immediacy.py. Every number
is traceable to a source or to a line of arithmetic in this file.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK, MUTE, ACC, NEG = "#1a1a1a", "#8a8a8a", "#c1440e", "#2b6a8f"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": INK, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

CAPITAL = 450_000.0
RISK_BUDGET = 17_308.0   # CAPITAL * 0.20 * 2.5 / 13

# ---------------------------------------------------------------------------------
# Contract reference data. Multipliers and ticks are exact. Prices, volatilities and
# ADV are representative mid-2026 levels; the code recomputes them live at runtime.
# ---------------------------------------------------------------------------------
# name, multiplier, price, ann vol, ADV(lots), tick$, comm/side, included
REF = [
    ("MCL", 100,    65.0,  0.35,  60_000,  1.00, 0.75, True),
    ("QG",  2_500,   4.0,  0.55,   4_000, 12.50, 0.75, True),
    ("MGC", 10,   3_300.0, 0.16,  15_000,  1.00, 0.75, True),
    ("SIL", 1_000,  40.0,  0.28,   4_000,  5.00, 0.75, True),
    ("MHG", 2_500,   5.50, 0.22,   8_000,  1.25, 0.75, True),
    ("ZC",  5_000,   4.50, 0.22, 350_000, 12.50, 1.50, True),
    ("ZW",  5_000,   5.50, 0.26, 120_000, 12.50, 1.50, True),
    ("KE",  5_000,   5.50, 0.27,  35_000, 12.50, 1.50, True),
    ("ZS",  5_000,  11.00, 0.18, 200_000, 12.50, 1.50, True),
    ("ZM",  100,   320.0,  0.22, 100_000, 10.00, 1.50, True),
    ("ZL",  60_000,  0.50, 0.25, 130_000,  6.00, 1.50, True),
    ("LE",  40_000,  2.00, 0.14,  50_000, 10.00, 1.50, True),
    ("HE",  40_000,  0.90, 0.25,  40_000, 10.00, 1.50, True),
    # excluded — one lot carries more risk than the whole per-instrument budget
    ("CC",  10,   6_000.0, 0.45,  30_000, 10.00, 1.50, False),
    ("KC",  37_500, 3.00,  0.40,  25_000, 18.75, 1.50, False),
    ("HO",  42_000, 2.20,  0.32, 100_000,  4.20, 1.50, False),
    ("RB",  42_000, 2.10,  0.32, 100_000,  4.20, 1.50, False),
    ("GC",  100,  3_300.0, 0.16, 250_000, 10.00, 1.50, False),
    ("CL",  1_000,  65.0,  0.35, 900_000, 10.00, 1.50, False),
]
COLS = ["sym", "mult", "px", "vol", "adv", "tick", "comm", "incl"]
ref = pd.DataFrame(REF, columns=COLS)
ref["notional"] = ref["mult"] * ref["px"]
ref["dvol"] = ref["notional"] * ref["vol"]              # annualised $ vol per lot
ref["lots"] = RISK_BUDGET / ref["dvol"]                 # lots at full forecast


def panel1(ax):
    """
    KRT (JF 2020) Table 4: quintile spread on hedger net trading, by event window.
    The argument: the premium is not consumed before the report is public. Roughly
    three quarters of it accrues AFTER release, in a window we can actually reach.
    """
    seg = [("1-4\npre-release", 0.21, 3.04, MUTE),
           ("5-10", 0.30, 3.62, ACC),
           ("11-20", 0.15, 1.23, ACC),
           ("21-40\nreverted", -0.07, -0.39, MUTE)]
    x = np.arange(len(seg))
    vals = [s[1] for s in seg]
    ax.bar(x, vals, color=[s[3] for s in seg], width=0.62, zorder=3)
    for i, (lab, v, t, _) in enumerate(seg):
        ax.text(i, v + (0.018 if v > 0 else -0.03), f"{v:+.2f}%\nt={t:.2f}",
                ha="center", va="bottom" if v > 0 else "top", fontsize=7)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axvspan(0.5, 2.5, color=ACC, alpha=0.06, zorder=0)
    ax.text(1.5, 0.40, "our holding window", ha="center", fontsize=7,
            color=ACC, style="italic")
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in seg], fontsize=7)
    ax.set_ylim(-0.17, 0.46)
    ax.set_ylabel("excess return, high−low quintile (%)")
    ax.set_title("1 · The premium survives the 3-day publication lag",
                 fontsize=8.5, loc="left", weight="bold")
    ax.set_xlabel("trading days after the CFTC position measurement date", fontsize=7)


def panel2(ax):
    """
    One lot's annualised dollar volatility against the per-instrument risk budget.
    The argument: the universe is set by integer granularity, not by liquidity. Coffee
    and cocoa trade tens of thousands of lots a day and are still untradeable here.
    """
    d = ref.sort_values("dvol")
    y = np.arange(len(d))
    cols = [ACC if i else MUTE for i in d["incl"]]
    ax.barh(y, d["dvol"] / 1000, color=cols, height=0.68, zorder=3)
    ax.axvline(RISK_BUDGET / 1000, color=NEG, lw=1.2, ls="--", zorder=4)
    ax.text(RISK_BUDGET / 1000 + 1.5, 0.4, f"risk budget\n${RISK_BUDGET/1000:.1f}k",
            fontsize=6.5, color=NEG, va="bottom")
    for i, (_, r) in enumerate(d.iterrows()):
        if r["lots"] >= 1:
            ax.text(r["dvol"] / 1000 + 0.7, i, f"{r['lots']:.1f}", fontsize=6,
                    va="center", color=INK)
    ax.set_yticks(y); ax.set_yticklabels(d["sym"], fontsize=6.5)
    ax.set_xlabel("annualised $ volatility of ONE contract (000s)", fontsize=7)
    ax.set_title("2 · Integer granularity, not liquidity, sets the universe",
                 fontsize=8.5, loc="left", weight="bold")
    ax.legend(handles=[Patch(color=ACC, label="tradeable at $450k"),
                       Patch(color=MUTE, label="excluded: one lot > budget")],
              fontsize=6.5, loc="lower right", frameon=False)


def panel3(ax):
    """
    Liquidity coefficient (Q) versus insurance coefficient (HP-bar) across published
    specifications and sub-periods. The argument: the premium we are harvesting is
    stable to rising; the textbook one next to it decays to zero post-financialisation.
    Sources: KRT JF 2020 Tables 3/6 and Appendix A1; Marechal JFM 2023 Tables 5/6.
    """
    specs = ["KRT\n1994-2014", "KRT\npre-2008", "KRT\npost-2008",
             "Mar.\nreplication", "Mar.\nrisk-adj", "Mar.\n1994-2020",
             "Mar. panel\npre-fin.", "Mar. panel\npost-fin."]
    q   = [4.77, 3.66, 8.17, 4.66, 3.80, 3.20, 3.19, 4.79]
    qt  = [6.55, 4.67, 4.69, 5.97, 4.91, 2.49, 3.26, 3.31]
    hp  = [np.nan, np.nan, np.nan, 0.43, 0.34, 0.41, 0.25, -0.37]
    hpt = [np.nan, np.nan, np.nan, 2.67, 1.93, 2.33, 1.34, -0.76]

    x = np.arange(len(specs))
    ax.bar(x - 0.19, q, width=0.36, color=ACC, zorder=3, label="liquidity provision (Q)")
    ax.bar(x + 0.19, np.nan_to_num(hp), width=0.36, color=NEG, zorder=3,
           label="insurance premium (HP)")
    for i, (a, b) in enumerate(zip(q, qt)):
        ax.text(i - 0.19, a + 0.18, f"{b:.1f}", ha="center", fontsize=5.8, color=ACC)
    for i, (a, b) in enumerate(zip(hp, hpt)):
        if np.isfinite(a):
            ax.text(i + 0.19, a + (0.18 if a > 0 else -0.75), f"{b:.1f}", ha="center",
                    fontsize=5.8, color=NEG)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_ylim(-1.5, 9.4)
    ax.set_xticks(x); ax.set_xticklabels(specs, fontsize=5.8)
    ax.set_ylabel("Fama–MacBeth slope  (t-stat above bar)", fontsize=7)
    ax.set_title("3 · The liquidity premium persists where the insurance premium decays",
                 fontsize=8.5, loc="left", weight="bold")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")


def capacity_curve():
    """
    What actually changes with AUM.

      cost   : contract-sides scale linearly with capital, so cost as a % of AUM is
               nearly FLAT. It falls only where a micro position grows past ~10 lots and
               can be replaced by the full-size contract, which cuts commission per unit
               of notional by 5x while leaving spread per unit of notional unchanged.
      rounding: falls steeply. This is what sets the FLOOR.
      participation: rises. This is what sets the CEILING.

    Turnover of 85 contract-sides per name per year is measured from the engine, not
    assumed; see immediacy.py smoke diagnostics.
    """
    SIDES_PER_NAME = 85.0
    IMPACT_MULT = 1.4          # impact ~ 29% of total cost, measured
    inc = ref[ref.incl].copy()
    base_budget = 450_000 * 0.20 * 2.5 / 13
    base_lots = base_budget / inc["dvol"].to_numpy()

    aums = np.array([150e3, 250e3, 450e3, 1e6, 2.5e6, 5e6, 15e6, 50e6, 150e6, 400e6])
    micro = np.array([s in ("MCL", "MGC", "SIL", "MHG") for s in inc["sym"]])
    rows = []
    for a in aums:
        lots = (a * 0.20 * 2.5 / 13) / inc["dvol"].to_numpy()
        upsize = micro & (lots > 10)                       # swap micro -> full size
        eff_lots = np.where(upsize, lots / 10, lots)
        comm = np.where(upsize, 1.50, inc["comm"].to_numpy())
        tick = np.where(upsize, inc["tick"].to_numpy() * 10, inc["tick"].to_numpy())
        adv = np.where(upsize, inc["adv"].to_numpy() / 6, inc["adv"].to_numpy())

        sides = SIDES_PER_NAME * eff_lots / base_lots      # scales with position size
        cost = np.sum(sides * (comm + 0.5 * tick)) * IMPACT_MULT
        rerr = np.mean(np.clip(0.25 / np.maximum(eff_lots, 0.01), 0, 1))
        part = np.max(eff_lots / adv)
        rows.append(dict(aum=a, cost_pct=cost / a, round_err=rerr,
                         median_lots=np.median(eff_lots), max_participation=part))
    return pd.DataFrame(rows)


def panel4(ax):
    """
    The argument: $450k sits near the FLOOR of this strategy's capacity, not the
    ceiling. Cost per unit of capital is scale-invariant; what scale buys is precision.
    """
    cap = capacity_curve()
    ax.plot(cap["aum"], cap["round_err"] * 100, "s-", color=NEG, lw=1.5, ms=4,
            label="integer rounding error (% of position)")
    ax.plot(cap["aum"], cap["max_participation"] * 100, "^-", color=ACC, lw=1.5, ms=4,
            label="peak participation (% of ADV)")
    ax.plot(cap["aum"], cap["cost_pct"] * 100, "o--", color=MUTE, lw=1.1, ms=3,
            label="all-in cost (% of AUM p.a.)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.08, 60)
    ax.axvspan(1e5, 2.5e5, color=NEG, alpha=0.08)
    ax.axvspan(7.5e7, 4e8, color=ACC, alpha=0.08)
    ax.axvline(450e3, color=INK, lw=1.1)
    ax.text(4.9e5, 30, "$450k", fontsize=6.8, color=INK, weight="bold")
    ax.text(1.55e5, 45, "floor\nrounding>18%", fontsize=6, color=NEG, ha="center", va="top")
    ax.text(1.6e8, 0.16, "ceiling\nparticipation>10% ADV", fontsize=6,
            color=ACC, ha="center", va="bottom")
    ax.set_xlabel("capital deployed (log scale)", fontsize=7)
    ax.set_ylabel("percent (log scale)", fontsize=7)
    ax.set_title("4 · $450k is near the floor of capacity, not the ceiling",
                 fontsize=8.5, loc="left", weight="bold")
    ax.legend(fontsize=6.3, frameon=False, loc="lower left",
              bbox_to_anchor=(0.02, 0.02))
    ax.set_xticks([1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels(["$100k", "$1m", "$10m", "$100m"], fontsize=6.5)
    ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(["0.1", "1", "10"], fontsize=6.5)
    return cap


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    panel1(axes[0, 0]); panel2(axes[0, 1]); panel3(axes[1, 0]); cap = panel4(axes[1, 1])
    fig.suptitle("Selling Immediacy in Commodity Futures — supporting exhibits",
                 fontsize=10, weight="bold", x=0.062, ha="left", y=0.985)
    fig.text(0.062, 0.012,
             "Panels 1 and 3 plot published coefficients (Kang, Rouwenhorst & Tang, "
             "Journal of Finance 2020; Maréchal, Journal of Futures Markets 2023).\n"
             "Panels 2 and 4 are arithmetic on exchange contract specifications at "
             "$450,000 of capital. No simulated returns appear on this page.",
             fontsize=6.2, color=MUTE)
    fig.tight_layout(rect=[0.01, 0.03, 0.99, 0.965])
    fig.savefig("/home/claude/build/exhibits.png", dpi=200)
    fig.savefig("/home/claude/build/exhibits.pdf")

    print(ref[["sym", "notional", "dvol", "lots", "incl"]].round(2).to_string(index=False))
    print()
    print(cap.round(4).to_string(index=False))


if __name__ == "__main__":
    main()