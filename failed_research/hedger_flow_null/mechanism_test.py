"""
mechanism_test.py — does hedger net trading predict returns in YOUR data?

    python mechanism_test.py --cot cot.parquet --prices data/px_clean.parquet

This bypasses the entire portfolio construction — no sizing, no buffering, no integer
contracts, no costs, no eligibility filter. It runs the cross-sectional regression that
the published papers run, so the number is directly comparable:

    Kang, Rouwenhorst & Tang (JF 2020)   slope 4.77  (t = 6.55)
    Marechal (JFM 2023), risk-adjusted   slope 3.80  (t = 4.91)
    Marechal, extended to 2020           slope 3.20  (t = 2.49)

WHY THIS MATTERS

A backtest that makes nothing has two possible causes and they need opposite responses:

    the mechanism is absent in this data  -> the strategy is dead; report that honestly
    the mechanism is there but the portfolio construction is throwing it away
                                          -> find the construction bug

Only a direct test of the mechanism separates them. If the slope here is positive and
significant while the backtest earns nothing, the fault is downstream of the signal.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import immediacy as m

HORIZON = 15   # trading days held from entry; KRT's days 5-20 window


def forward_returns(front: pd.DataFrame, h: int = HORIZON) -> pd.DataFrame:
    """
    Cumulative return over the next h trading days, per symbol.

    Roll days carry a NaN return by construction (a roll starts a new contract's series),
    and are treated as zero price change here — which is what a rolled position actually
    experiences, with the roll's cost charged separately.
    """
    # pivot_table's default aggfunc is 'mean', so a duplicate (date, symbol) row would
    # be averaged into its neighbour rather than raising — a silent halving of that
    # day's return. build_front_series emits one row per pair by construction, so this
    # never fires; it is here because the failure mode is invisible if it ever does.
    dup = front.duplicated(["date", "symbol"]).sum()
    if dup:
        raise ValueError(
            f"{dup} duplicate (date, symbol) rows in the front series; pivot_table "
            f"would silently average them into a single return."
        )
    piv = front.pivot_table(index="date", columns="symbol", values="ret")
    cum = np.log1p(piv.fillna(0.0)).cumsum()
    return np.expm1(cum.shift(-h) - cum)


def fm_slope(panel: pd.DataFrame, xcol: str, ycol: str = "fwd") -> dict:
    """
    Fama-MacBeth: one cross-sectional slope per week, then a t-test on the time series
    of slopes. Weeks with fewer than 4 usable names are dropped.
    """
    slopes = []
    for _, g in panel.groupby("report_date"):
        g = g[[xcol, ycol]].dropna()
        if len(g) < 4 or g[xcol].std() == 0:
            continue
        x, y = g[xcol].to_numpy(), g[ycol].to_numpy()
        b = np.polyfit(x, y, 1)[0]
        if np.isfinite(b):
            slopes.append(b)
    if len(slopes) < 30:
        return dict(n=len(slopes))
    s = np.array(slopes)
    return dict(n=len(s), slope=s.mean(), t=s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))


def quintile_spread(panel: pd.DataFrame, xcol: str) -> dict:
    """KRT's headline construction: top minus bottom quintile, equal weighted."""
    hi, lo = [], []
    for _, g in panel.groupby("report_date"):
        g = g[[xcol, "fwd"]].dropna()
        if len(g) < 5:
            continue
        k = max(1, len(g) // 5)
        g = g.sort_values(xcol)
        lo.append(g["fwd"].head(k).mean())
        hi.append(g["fwd"].tail(k).mean())
    if len(hi) < 30:
        return dict(n=len(hi))
    d = np.array(hi) - np.array(lo)
    return dict(n=len(d), spread=d.mean(), t=d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def report(label: str, r: dict, unit: str = "") -> None:
    if "slope" in r:
        print(f"  {label:38s} {r['slope']:>8.3f}  t={r['t']:>6.2f}   n={r['n']:,}")
    elif "spread" in r:
        print(f"  {label:38s} {r['spread']*100:>7.3f}%  t={r['t']:>6.2f}   n={r['n']:,}")
    else:
        print(f"  {label:38s} too few weeks ({r.get('n', 0)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", default="cot.parquet")
    ap.add_argument("--prices", default="data/px_clean.parquet")
    ap.add_argument("--oos-cut", default=None,
                    help="exclude data on/after this date (keeps OOS sealed)")
    a = ap.parse_args()

    px = pd.read_parquet(a.prices); cot = pd.read_parquet(a.cot)
    px["date"] = pd.to_datetime(px["date"]); px["expiry"] = pd.to_datetime(px["expiry"])
    cot["report_date"] = pd.to_datetime(cot["report_date"])

    front = m.build_front_series(px)
    dates = pd.DatetimeIndex(sorted(front["date"].unique()))
    cut = pd.Timestamp(a.oos_cut) if a.oos_cut else m.split_oos(dates)
    print(f"in-sample only, through {cut:%Y-%m-%d}   (OOS stays sealed)\n")

    wk = m.weekly_returns_from_front(front, pd.DatetimeIndex(sorted(cot["report_date"].unique())))
    sig = m.compute_signals(cot, wk)

    fwd = forward_returns(front)
    trade_map = {pd.Timestamp(rd): m.first_tradeable_date(pd.Timestamp(rd), dates)
                 for rd in sig["report_date"].unique()}
    sig["trade_date"] = sig["report_date"].map(trade_map)
    sig = sig.dropna(subset=["trade_date"])
    sig = sig[sig["trade_date"] < cut]

    sig["fwd"] = [fwd.at[t, s] if (t in fwd.index and s in fwd.columns) else np.nan
                  for t, s in zip(sig["trade_date"], sig["symbol"])]
    panel = sig.dropna(subset=["fwd", "Q"]).copy()
    print(f"{len(panel):,} name-weeks, {panel['report_date'].nunique():,} weeks, "
          f"{panel['symbol'].nunique()} contracts\n")

    print("=" * 72)
    print("A. THE MECHANISM — does hedger net trading predict the next 3 weeks?")
    print("=" * 72)
    report("slope on raw Q  (vs KRT 4.77)", fm_slope(panel, "Q"))
    report("slope on cross-sectional z", fm_slope(panel, "z"))
    report("quintile spread on Q  (vs KRT 0.45%)", quintile_spread(panel, "Q"))

    print("\n" + "=" * 72)
    print("B. IS THE CONDITIONING EARNING ITS KEEP?")
    print("=" * 72)
    med = panel.groupby("report_date")["HPbar"].transform("median")
    for lab, sub in (("hedging pressure ABOVE median", panel[panel["HPbar"] > med]),
                     ("hedging pressure BELOW median", panel[panel["HPbar"] <= med]),
                     ("hedger stress gate ON", panel[panel["g"] > 1]),
                     ("hedger stress gate OFF", panel[panel["g"] <= 1])):
        report(lab, fm_slope(sub, "Q"))
    print("\n  If the below-median slope is as strong as the above-median one, the")
    print("  eligibility filter is discarding half the universe for nothing. With")
    print("  producer-only data that is plausible: 96.9% of weeks have positive hedging")
    print("  pressure, so a median split is cutting on noise, not on economics.")

    print("\n" + "=" * 72)
    print("C. DOES SECTOR DEMEANING HELP OR HURT?")
    print("=" * 72)
    report("slope on raw Q", fm_slope(panel, "Q"))
    report("slope on sector-demeaned Q", fm_slope(panel, "Q_tilde")
           if "Q_tilde" in panel.columns else dict(n=0))

    print("\n" + "=" * 72)
    print("D. SUBPERIODS")
    print("=" * 72)
    yr = panel["report_date"].dt.year
    for lab, sub in (("2010-2014", panel[yr <= 2014]),
                     ("2015-2018", panel[(yr >= 2015) & (yr <= 2018)]),
                     ("2019-cut  ", panel[yr >= 2019])):
        report(lab, fm_slope(sub, "Q"))

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("  Section A slope clearly positive with t > 2")
    print("      the mechanism is present. A backtest earning nothing then has a")
    print("      construction bug, not a dead premise.")
    print("  Section A slope near zero or negative")
    print("      the mechanism is absent in this universe over this sample. That is a")
    print("      real finding and belongs in the pitch. It does not necessarily")
    print("      contradict KRT — they used 26 commodities from 1994 and the legacy")
    print("      commercial category, we use 13 from 2010 and producers only.")


if __name__ == "__main__":
    main()