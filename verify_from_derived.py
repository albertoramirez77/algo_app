"""
verify_from_derived.py — reproduce the headline numbers from the committed CSVs.

    python verify_from_derived.py

No Databento account, no API key, no raw price file. This reads only
data/derived/monthly_pnl.csv and recomputes the performance statistics the pitch
reports, so a reader can check the arithmetic independently of the backtest.

What this does NOT verify: that the monthly P&L series itself is correct. That requires
the price data, and the route to it is in data/README.md. What it does verify is that
every statistic quoted downstream of that series follows from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DERIVED = Path("data/derived")

# What the pitch claims. Edit only when the run changes.
CLAIMS = dict(sharpe=0.94, t=3.66, ann_return=0.183, ann_vol=0.195,
              max_dd=-0.280, longest_dd_months=42, win_rate=0.61, profit_factor=2.04)
TOL = dict(sharpe=0.01, t=0.05, ann_return=0.005, ann_vol=0.005,
           max_dd=0.005, longest_dd_months=0, win_rate=0.01, profit_factor=0.05)


def longest_drawdown(equity: pd.Series) -> int:
    """Longest run of months spent below a previous peak."""
    peak = equity.cummax()
    under = equity < peak
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def main() -> int:
    path = DERIVED / "monthly_pnl.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `make derived` first.")

    m = pd.read_csv(path, index_col=0)
    r = m["net"].astype(float)
    yrs = len(r) / 12

    equity = (1 + r).cumprod()
    wins, losses = r[r > 0], r[r < 0]

    got = dict(
        sharpe=(r.mean() * 12) / (r.std(ddof=1) * np.sqrt(12)),
        ann_return=r.mean() * 12,
        ann_vol=r.std(ddof=1) * np.sqrt(12),
        max_dd=float((equity / equity.cummax() - 1).min()),
        longest_dd_months=longest_drawdown(equity),
        win_rate=len(wins) / len(r),
        profit_factor=wins.sum() / abs(losses.sum()),
    )
    got["t"] = got["sharpe"] * np.sqrt(yrs)

    print(f"\n  {len(r)} months from {path}\n")
    print(f"  {'statistic':<22}{'claimed':>10}{'recomputed':>14}   result")
    print("  " + "-" * 60)

    failures = 0
    for k in ("sharpe", "t", "ann_return", "ann_vol", "max_dd",
              "longest_dd_months", "win_rate", "profit_factor"):
        ok = abs(got[k] - CLAIMS[k]) <= TOL[k]
        failures += not ok
        print(f"  {k:<22}{CLAIMS[k]:>10.3f}{got[k]:>14.3f}   {'ok' if ok else 'MISMATCH'}")

    print()
    if failures:
        print(f"  {failures} statistic(s) do not follow from the committed series.\n")
        return 1
    print("  Every headline statistic follows from the committed monthly series.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
