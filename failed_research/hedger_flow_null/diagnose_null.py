"""
diagnose_null.py — is the null real, or is the return data broken?

    python diagnose_null.py --cot cot.parquet --prices data/px_clean.parquet

mechanism_test.py found a slope of 0.003 (t=0.08) on hedger net trading against KRT's
published 4.77. That is absence, not attenuation. Two very different things produce it:

    the premium is genuinely not in this universe/sample
        -> the strategy is dead as specified and the pitch has to change

    the forward returns or the alignment are broken
        -> nothing would show up, including effects that certainly exist

The only way to separate them is a positive control: test whether a WELL-DOCUMENTED
commodity effect appears in exactly the same panel, using exactly the same forward
returns and the same alignment. If momentum and short-term reversal show up and hedger
flow does not, the null is real. If nothing shows up, the plumbing is at fault.

It also tests the OTHER side of the mechanism. KRT decompose by trader category and find
producers at +8.83 and money managers at -6.91 — the liquidity supplier and the liquidity
consumer. If producers are flat but money managers are strongly negative, the mechanism
exists and we are measuring the wrong side of it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import immediacy as m
from mechanism_test import fm_slope, forward_returns, quintile_spread


def build_panel(cot: pd.DataFrame, px: pd.DataFrame, cut: pd.Timestamp,
                horizon: int) -> pd.DataFrame:
    front = m.build_front_series(px)
    dates = pd.DatetimeIndex(sorted(front["date"].unique()))
    wk = m.weekly_returns_from_front(
        front, pd.DatetimeIndex(sorted(cot["report_date"].unique())))
    sig = m.compute_signals(cot, wk)

    # extra predictors, all built from data available at the report date
    extra = cot.sort_values(["symbol", "report_date"]).copy()
    extra["mm_net"] = extra["mm_long"] - extra["mm_short"]
    extra["oi_lag"] = extra.groupby("symbol")["open_interest"].shift(1)
    extra["Q_mm"] = (extra["mm_net"] - extra.groupby("symbol")["mm_net"].shift(1)) \
                    / extra["oi_lag"]
    sig = sig.merge(extra[["report_date", "symbol", "Q_mm"]],
                    on=["report_date", "symbol"], how="left")
    sig = sig.merge(wk, on=["report_date", "symbol"], how="left")

    # trailing 12-month and 1-week price signals, as of the report date
    piv = front.pivot_table(index="date", columns="symbol", values="settle")
    rd = pd.DatetimeIndex(sorted(sig["report_date"].unique()))
    lvl = piv.reindex(piv.index.union(rd)).ffill().reindex(rd)
    mom12 = (lvl / lvl.shift(52) - 1).stack().reset_index()
    mom12.columns = ["report_date", "symbol", "mom12"]
    sig = sig.merge(mom12, on=["report_date", "symbol"], how="left")

    fwd = forward_returns(front, horizon)
    tmap = {pd.Timestamp(r): m.first_tradeable_date(pd.Timestamp(r), dates)
            for r in sig["report_date"].unique()}
    sig["trade_date"] = sig["report_date"].map(tmap)
    sig = sig.dropna(subset=["trade_date"])
    sig = sig[sig["trade_date"] < cut]
    sig["fwd"] = [fwd.at[t, s] if (t in fwd.index and s in fwd.columns) else np.nan
                  for t, s in zip(sig["trade_date"], sig["symbol"])]
    return sig.dropna(subset=["fwd"])


def show(label: str, r: dict, expect: str = "") -> None:
    if "slope" in r:
        flag = ""
        if abs(r["t"]) > 2:
            flag = "  <-- significant"
        print(f"  {label:34s} {r['slope']:>9.3f}  t={r['t']:>6.2f}   "
              f"n={r['n']:>4,}  {expect}{flag}")
    else:
        print(f"  {label:34s} too few weeks ({r.get('n', 0)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", default="cot.parquet")
    ap.add_argument("--prices", default="data/px_clean.parquet")
    a = ap.parse_args()

    px = pd.read_parquet(a.prices); cot = pd.read_parquet(a.cot)
    px["date"] = pd.to_datetime(px["date"]); px["expiry"] = pd.to_datetime(px["expiry"])
    cot["report_date"] = pd.to_datetime(cot["report_date"])
    front = m.build_front_series(px)
    cut = m.split_oos(pd.DatetimeIndex(sorted(front["date"].unique())))
    print(f"in-sample only, through {cut:%Y-%m-%d}\n")

    p = build_panel(cot, px, cut, 15)

    print("=" * 78)
    print("1. ARE THE FORWARD RETURNS EVEN SANE?")
    print("=" * 78)
    f = p["fwd"]
    print(f"  n = {len(f):,}   mean {f.mean()*100:+.3f}%   sd {f.std()*100:.2f}%   "
          f"min {f.min()*100:+.1f}%   max {f.max()*100:+.1f}%")
    print(f"  implied annualised vol of a single name: "
          f"{f.std()*np.sqrt(252/15)*100:.1f}%   (expect roughly 20-40%)")
    if f.std() < 0.01:
        print("  ^^ FAR too small. The forward returns are broken; stop here.")

    print("\n" + "=" * 78)
    print("2. POSITIVE CONTROLS — can this panel detect effects that DO exist?")
    print("=" * 78)
    show("12-month momentum", fm_slope(p.dropna(subset=["mom12"]), "mom12"),
         "expect > 0 ")
    show("1-week reversal", fm_slope(p.dropna(subset=["week_ret"]), "week_ret"),
         "expect < 0 ")
    print("\n  If BOTH of these are flat, the panel cannot detect anything and the null")
    print("  on hedger flow tells you nothing. Fix the plumbing before concluding.")

    print("\n" + "=" * 78)
    print("3. THE MECHANISM, BOTH SIDES")
    print("=" * 78)
    show("producer net trading  (KRT +8.83)", fm_slope(p, "Q"), "expect > 0 ")
    show("money manager net trading (-6.91)", fm_slope(p.dropna(subset=["Q_mm"]), "Q_mm"),
         "expect < 0 ")
    show("smoothed hedging pressure level", fm_slope(p, "HPbar"), "expect > 0 ")
    print("\n  Producers flat but money managers strongly negative would mean the")
    print("  mechanism is present and we are simply measuring the wrong side of it —")
    print("  the trade becomes 'fade the money manager' rather than 'follow the")
    print("  producer'. Economically the same transfer, cleaner to measure.")

    print("\n" + "=" * 78)
    print("4. HORIZON SWEEP — is the effect at a different holding period?")
    print("=" * 78)
    for h in (5, 10, 15, 20, 25, 40):
        ph = build_panel(cot, px, cut, h)
        r = fm_slope(ph, "Q")
        rm = fm_slope(ph.dropna(subset=["Q_mm"]), "Q_mm")
        s = (f"  {h:>3d} trading days   producer {r.get('slope', float('nan')):>8.3f} "
             f"(t={r.get('t', float('nan')):>5.2f})")
        s += (f"   money mgr {rm.get('slope', float('nan')):>8.3f} "
              f"(t={rm.get('t', float('nan')):>5.2f})")
        print(s)

    print("\n" + "=" * 78)
    print("WHAT TO DO WITH THIS")
    print("=" * 78)
    print("  controls significant, hedger flow flat at every horizon")
    print("      The null is real. The premium is not in 13 producer-only contracts")
    print("      over 2010-2022. Say so; it is a finding, not a failure. Then either")
    print("      pivot the pitch to the runner-up mechanism or rebuild the signal.")
    print()
    print("  money manager side significant and negative")
    print("      Keep the strategy, invert the measurement. Same economics, and it")
    print("      sidesteps the producer-classification objection entirely.")
    print()
    print("  controls also flat")
    print("      Do not touch the pitch. The forward returns or the alignment are")
    print("      wrong and every number so far is uninformative.")


if __name__ == "__main__":
    main()