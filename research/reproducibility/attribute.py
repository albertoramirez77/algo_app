"""
attribute.py — where does the P&L actually come from?

    python attribute.py --cot cot.parquet --prices px.parquet

THE PROBLEM THIS SOLVES

The cross-sectional regression says hedger net trading has no predictive power
(t = 0.08 on producers, t = 0.24 on money managers, nothing at any horizon). The
portfolio built on that signal nonetheless earns a positive return. Those two facts
cannot both describe the same source of return.

The portfolio does more than rank names on Q. It also:

    filters to names with above-median smoothed hedging pressure  (a persistent tilt
        toward structurally backwardated contracts — that is a carry exposure, not a
        liquidity-provision one)
    doubles weight after hedger capital losses  (a volatility-state timing rule)
    demeans within sector, winsorises, buffers, and rounds to integers

Any of those can produce return on its own, and none of them is the mechanism the pitch
claims. If the money is coming from the hedging-pressure tilt, the strategy is a carry
strategy wearing a liquidity-provision costume, and it must either be re-pitched honestly
as carry or redesigned.

THE DECISIVE TEST IS THE PLACEBO

Shuffle Q across names within each week, changing nothing else. The eligibility filter,
the stress gate, the sizing, the costs and the roll all stay exactly as they are; only
the cross-sectional ranking is destroyed. If the shuffled version earns what the real one
earns, the signal contributes nothing and every dollar comes from the machinery around it.

Run with several seeds — one placebo draw is an anecdote.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import immediacy as m


def rescale_forecast(sig: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Rebuild S and F from s, using the same expanding-window scalar as the engine."""
    sig = sig.sort_values(["symbol", "report_date"]).copy()
    sig["S"] = sig.groupby("symbol")["s"].transform(
        lambda x: x.rolling(p["hold_weeks"], min_periods=1).mean())
    active = sig["S"].abs() > 1e-12
    absS = (sig.assign(_a=sig["S"].abs().where(active))
               .groupby("report_date")["_a"].transform("mean"))
    exp = (absS.groupby(sig["report_date"]).first().expanding().mean()
               .reindex(sig["report_date"]).to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        sig["F"] = np.clip(sig["S"] * m.FORECAST_SCALAR / exp,
                           -m.FORECAST_CAP, m.FORECAST_CAP)
    sig["F"] = sig["F"].fillna(0.0)
    return sig


def variant(sig: pd.DataFrame, p: dict, *, permute_seed: int | None = None,
            drop_filter: bool = False, drop_stress: bool = False) -> pd.DataFrame:
    s = sig.copy()
    if permute_seed is not None:
        rng = np.random.default_rng(permute_seed)
        s["z"] = s.groupby("report_date")["z"].transform(
            lambda x: rng.permutation(x.to_numpy()))
    e = 1.0 if drop_filter else s["e"].fillna(0.0)
    g = 1.0 if drop_stress else s["g"]
    s["s"] = s["z"].fillna(0.0) * e * g
    return rescale_forecast(s, p)


def stats_of(front, sig, p, cut) -> dict:
    res = m.run_backtest(front, sig, p, end=cut)
    st = res.stats(net=True)
    sides = (res.trades["delta"].abs().sum() if not res.trades.empty else 0)
    yrs = max((res.daily.index[-1] - res.daily.index[0]).days / 365.25, 1e-9)
    return dict(sharpe=st.get("sharpe", np.nan), ann_ret=st.get("ann_return", np.nan),
                vol=st.get("ann_vol", np.nan), dd=st.get("max_drawdown", np.nan),
                sides_yr=sides / yrs, daily=res.daily)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", default="cot.parquet")
    ap.add_argument("--prices", default="px.parquet")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    px = pd.read_parquet(a.prices); cot = pd.read_parquet(a.cot)
    px["date"] = pd.to_datetime(px["date"]); px["expiry"] = pd.to_datetime(px["expiry"])
    cot["report_date"] = pd.to_datetime(cot["report_date"])

    front = m.build_front_series(px)
    dates = pd.DatetimeIndex(sorted(front["date"].unique()))
    cut = m.split_oos(dates)
    wk = m.weekly_returns_from_front(
        front, pd.DatetimeIndex(sorted(cot["report_date"].unique())))
    base_sig = m.compute_signals(cot, wk)
    p = dict(m.PARAMS)
    print(f"in-sample only, through {cut:%Y-%m-%d}\n")

    print("=" * 76)
    print("1. THE PLACEBO — shuffle Q within each week, change nothing else")
    print("=" * 76)
    real = stats_of(front, base_sig, p, cut)
    print(f"  {'real signal':28s} sharpe {real['sharpe']:>7.3f}   "
          f"ret {real['ann_ret']*100:>6.2f}%   vol {real['vol']*100:>5.1f}%")
    ph = []
    for sd in range(a.seeds):
        r = stats_of(front, variant(base_sig, p, permute_seed=sd), p, cut)
        ph.append(r["sharpe"])
        print(f"  {'placebo seed ' + str(sd):28s} sharpe {r['sharpe']:>7.3f}   "
              f"ret {r['ann_ret']*100:>6.2f}%   vol {r['vol']*100:>5.1f}%")
    ph = np.array(ph)
    print(f"\n  placebo mean {ph.mean():+.3f}, sd {ph.std(ddof=1):.3f}   "
          f"real {real['sharpe']:+.3f}")
    z = (real["sharpe"] - ph.mean()) / max(ph.std(ddof=1), 1e-9)
    print(f"  real signal sits {z:+.1f} placebo standard deviations from the placebo mean")
    if abs(z) < 2:
        print("\n  THE SIGNAL IS NOT THE SOURCE. A random ranking earns what the real one")
        print("  earns. Whatever the strategy is making comes from the filter, the stress")
        print("  gate, or the sizing — not from hedger net trading. The pitch cannot")
        print("  claim the liquidity-provision mechanism on this evidence.")
    else:
        print("\n  The real signal is distinguishable from noise. That is consistent with")
        print("  the portfolio capturing something the linear cross-sectional regression")
        print("  misses — non-linearity, or concentration in a subset of weeks.")

    print("\n" + "=" * 76)
    print("2. ABLATIONS — remove one component at a time")
    print("=" * 76)
    for lab, kw in (("full strategy", {}),
                    ("no hedging-pressure filter", dict(drop_filter=True)),
                    ("no stress gate", dict(drop_stress=True)),
                    ("neither", dict(drop_filter=True, drop_stress=True))):
        r = stats_of(front, variant(base_sig, p, **kw), p, cut)
        print(f"  {lab:28s} sharpe {r['sharpe']:>7.3f}   ret {r['ann_ret']*100:>6.2f}%"
              f"   vol {r['vol']*100:>5.1f}%   sides/yr {r['sides_yr']:>6,.0f}")
    print("\n  If dropping the hedging-pressure filter destroys the return, the return")
    print("  was the filter: a persistent tilt toward backwardated contracts, which is")
    print("  carry, not immediacy.")

    print("\n" + "=" * 76)
    print("3. IS THE RETURN A HANDFUL OF DAYS?")
    print("=" * 76)
    full = m.run_backtest(front, base_sig, p)          # whole sample, IS + OOS
    d = full.daily["pnl_net"]
    tot = d.sum()
    for k in (1, 5, 10, 20):
        top = d.nlargest(k).sum()
        print(f"  best {k:>2d} days = {top/tot*100:>6.1f}% of total P&L"
              if tot != 0 else f"  best {k} days: total P&L is zero")
    print()
    yr = d.groupby(d.index.year).sum() / m.CAPITAL * 100
    print("  return by calendar year (% of capital, net):")
    print("   ", "  ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))
    print("\n  A strategy whose return is concentrated in a few days is a lottery ticket,")
    print("  not a premium. The January 2026 silver dislocation sits in the OOS window;")
    print("  check whether it is carrying the 0.97.")


if __name__ == "__main__":
    main()