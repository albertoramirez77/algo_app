"""
check_data.py — run this after fetching and BEFORE the backtest.

Six checks. Numbers 4 and 5 are the ones that matter: check 4 tells you whether the data is
usable, and check 5 measures the single quantity that decides whether this strategy is
viable at $450,000.

    python check_data.py --cot cot.parquet --prices px.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import immediacy as m


def hdr(n: int, title: str) -> None:
    print(f"\n{'='*74}\n{n}. {title}\n{'='*74}")


def check_coverage(cot: pd.DataFrame, px: pd.DataFrame) -> None:
    hdr(1, "COVERAGE — do you have all thirteen contracts, over the full period?")
    a = cot.groupby("symbol").agg(cot_weeks=("report_date", "size"),
                                  cot_from=("report_date", "min"),
                                  cot_to=("report_date", "max"))
    b = px.groupby("symbol").agg(px_days=("date", "size"),
                                 contracts=("contract", "nunique"),
                                 px_from=("date", "min"), px_to=("date", "max"))
    j = a.join(b, how="outer")
    print(j.to_string())
    miss = [c.symbol for c in m.UNIVERSE if c.symbol not in j.index or j.loc[c.symbol].isna().any()]
    print(f"\n  {'OK' if not miss else 'PROBLEM — missing or incomplete: ' + str(miss)}")


def check_cot_gaps(cot: pd.DataFrame) -> None:
    hdr(2, "COT CONTINUITY — the 2025 shutdown should be visible here")
    s = cot[cot["symbol"] == cot["symbol"].iloc[0]].sort_values("report_date")
    gaps = s["report_date"].diff().dt.days
    big = s.loc[gaps > 10, "report_date"]
    print(f"  weekly observations: {len(s):,}")
    print(f"  median spacing: {gaps.median():.0f} days   max: {gaps.max():.0f} days")
    if len(big):
        print("  gaps longer than 10 days (report date AFTER the gap):")
        for d, g in zip(big, gaps[gaps > 10]):
            print(f"     {d:%Y-%m-%d}   {g:.0f} day gap")
        print("\n  A ~45 day gap around Oct-Nov 2025 is EXPECTED (federal shutdown).")
        print("  If you do not see it, your data may be back-filled rather than")
        print("  point-in-time, which would be a look-ahead problem.")
    else:
        print("  No large gaps found.")
        print("  This is EXPECTED for archives downloaded after December 2025: the CFTC")
        print("  back-filled the shutdown weeks at their original measurement dates, so")
        print("  the gap is invisible in the file even though the reports did not exist")
        print("  at the time. The engine compensates — cot_release_date() holds every")
        print("  report measured between 2025-10-01 and 2025-12-29 until the backlog")
        print("  actually cleared, so the backtest cannot trade on them early.")


def check_rolls(px: pd.DataFrame) -> None:
    hdr(3, "ROLL SANITY — is the front-contract series continuous and jump-free?")
    front = m.build_front_series(px)
    print(f"  front series: {len(front):,} rows, {front['symbol'].nunique()} symbols")
    r = front.dropna(subset=["ret"])
    bad = r[r["ret"].abs() > 0.25]
    print(f"  daily returns beyond +/-25%: {len(bad)}")
    if len(bad):
        print(bad[["date", "symbol", "contract", "settle", "ret"]].head(12).to_string(index=False))
        print("\n  A handful is normal (2020 crude, 2022 nickel-style events).")
        print("  Dozens clustered on roll dates means the roll logic is stitching across")
        print("  contracts — returns must NEVER be computed across a roll boundary.")
    roll_ret = front.loc[front["is_roll"], "ret"]
    print(f"  returns on roll days that are non-null: {roll_ret.notna().sum()} "
          f"(should be 0 — a roll day starts a new contract's return series)")


def check_signal(cot: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    hdr(4, "SIGNAL SANITY — does Q look like a hedger flow measure?")
    front = m.build_front_series(px)
    wk = m.weekly_returns_from_front(front, pd.DatetimeIndex(sorted(cot["report_date"].unique())))
    sig = m.compute_signals(cot, wk)
    q = sig["Q"].dropna()
    print(f"  Q observations: {len(q):,}")
    print(f"  mean |Q|: {q.abs().mean():.4f}   (KRT report 0.035 across 26 commodities)")
    if q.abs().mean() < 0.025:
        print("       Lower than published, and expected: KRT's headline uses the legacy")
        print("       COMMERCIAL category, which bundles swap dealers in with producers.")
        print("       We read Producer/Merchant only, a narrower and more liquid group,")
        print("       so position changes are a smaller fraction of open interest. KRT's")
        print("       own disaggregated test gives producers a coefficient of 8.83 versus")
        print("       ~4.77 for all hedgers, so a smaller spread meets a larger slope.")
        print("       Roughly a wash on paper; the backtest settles it directly.")
    print(f"  std  Q  : {q.std():.4f}")
    print(f"  Q beyond +/-25% of open interest: {(q.abs() > 0.25).sum()} "
          f"({(q.abs() > 0.25).mean():.2%})")
    hp = sig["HPbar"].dropna()
    print(f"  mean HP-bar: {hp.mean():+.4f}   (should be POSITIVE — hedgers net short "
          f"on average; KRT report +0.14)")
    print(f"  share of weeks with HP-bar > 0: {(hp > 0).mean():.1%}   (KRT: ~71%)")
    if hp.mean() < 0:
        print("\n  PROBLEM: hedgers appear net LONG on average. You have almost certainly")
        print("  swapped prod_long and prod_short, or picked the wrong trader category.")
    return sig


def check_turnover(sig: pd.DataFrame) -> None:
    hdr(5, "THE NUMBER THAT DECIDES THIS — autocorrelation of hedger net trading")
    rows = []
    for sym, g in sig.sort_values("report_date").groupby("symbol"):
        q = g["Q"].dropna()
        if len(q) > 60:
            rows.append(dict(symbol=sym, n=len(q), rho1=q.autocorr(1), rho2=q.autocorr(2)))
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    rho = t["rho1"].mean()
    print(f"\n  mean lag-1 autocorrelation of Q: {rho:+.3f}")
    print(f"  KRT report +0.17 for hedgers.")
    print()
    if rho > 0.05:
        print("  GOOD. Positive persistence means the signal does not whipsaw week to")
        print("  week, turnover stays near the ~1,100 contract-sides/yr the pitch assumes,")
        print("  and the ~2% cost estimate holds.")
    elif rho > -0.10:
        print("  MARGINAL. Turnover will run above the pitch's assumption. Expect costs")
        print("  nearer 3% of capital. Re-run the backtest with buffer_frac at 0.30 and")
        print("  report the cost figure you actually measure, not the one in the draft.")
    else:
        print("  BAD. Strongly negative autocorrelation means Q oscillates and the book")
        print("  churns. Costs head toward 4% of capital and the net Sharpe goes to zero.")
        print("  This is the failure mode flagged in the last paragraph of Analytical")
        print("  Evidence. If you see this, say so in the pitch. Do not submit the 2%")
        print("  number.")


def check_lookahead(cot: pd.DataFrame) -> None:
    hdr(6, "LOOK-AHEAD — confirm the publication lag is what you think it is")
    d = pd.DatetimeIndex(sorted(cot["report_date"].unique()))
    dow = pd.Series(d.dayofweek).value_counts().sort_index()
    names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    print("  report dates by weekday:")
    for k, v in dow.items():
        print(f"    {names.get(k, k):4s} {v:5,}")
    print("\n  Almost all should be Tuesday. Wednesdays are holiday weeks, which is")
    print("  correct behaviour — the CFTC shifts measurement when Monday is a holiday.")
    print(f"\n  Engine's registered lag: report date + 3 days = Friday release,")
    print(f"  first executable price = the following Monday's settlement.")
    print(f"  That is {6} calendar days from measurement to execution. If you ever see")
    print(f"  a smaller number in the audit output, stop and find the bug.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", default="cot.parquet")
    ap.add_argument("--prices", default="px.parquet")
    a = ap.parse_args()

    cot = pd.read_parquet(a.cot)
    px = pd.read_parquet(a.prices)
    cot["report_date"] = pd.to_datetime(cot["report_date"])
    px["date"] = pd.to_datetime(px["date"])
    px["expiry"] = pd.to_datetime(px["expiry"])

    check_coverage(cot, px)
    check_cot_gaps(cot)
    check_rolls(px)
    sig = check_signal(cot, px)
    check_turnover(sig)
    check_lookahead(cot)

    print(f"\n{'='*74}")
    print("If all six look right, run the backtest. If check 4 or 5 looks wrong, fix it")
    print("first — a backtest on bad data produces a number you will have to defend.")
    print("=" * 74)


if __name__ == "__main__":
    main()