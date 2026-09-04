#!/usr/bin/env python3
"""
check_pitch_numbers.py — assert that every number printed in the pitch still
matches the run that produced it.

    python check_pitch_numbers.py docs/FINAL_NUMBERS.txt

Exits 0 if the document and the run agree, 1 otherwise. Run it before every
submission and after every re-run; wire it into CI and the "one specification,
one file, one run" claim stops being something a reader has to take on trust.

To add a claim: put one line in CLAIMS. `label` is matched against the start of
a line in the numbers file, so it must be the literal label the script prints.
"""

from __future__ import annotations

import re
import sys

# (label in FINAL_NUMBERS.txt, value printed in the pitch, tolerance, where it appears
#  [, which number on the line to read, when the label is followed by more than one])
CLAIMS = [
    ("instruments",                    16,     0,     "Core Concept, Table 1"),
    ("months",                         183,    0,     "Analytical Evidence"),
    ("Sharpe ratio",                   0.94,   0.006, "Core Concept, Rationale, Table 1"),
    ("t-statistic",                    3.66,   0.005, "Table 1, Bonferroni paragraph"),
    ("annualised return",              18.3,   0.05,  "Core Concept, Table 1"),
    ("annualised volatility",          19.5,   0.05,  "Table 1, rejected-control paragraph"),
    ("maximum drawdown",              -28.0,   0.05,  "Table 1, Losses"),
    ("gross exposure",                 2.5,    0.01,  "Sizing, Table 1"),
    ("positions rounding to zero",     17.2,   0.05,  "Table 1, Capital and Liquidity (x3)"),
    ("net dollar exposure, mean",      13.2,   0.05,  "Sizing, Table 1"),
    ("net dollar exposure, worst",     51.0,   0.05,  "Sizing, Table 1"),
    ("front momentum alone",           0.14,   0.005, "Economic Rationale"),
    ("carry alone",                    0.37,   0.005, "Economic Rationale"),
    ("correlation to trend-following", 0.007,  0.0005,"Rationale, Portfolio fit"),
    ("market beta",                   -0.06,   0.006, "Core Concept, Risk Assessment"),
    ("average pairwise correlation",   0.21,   0.005, "Economic Rationale"),
    ("correlation of the two legs",    94.9,   0.05,  "Economic Rationale"),
    ("variance of BM / front momentum", 7.6,   0.05,  "Economic Rationale"),
    ("turnover-weighted cost",         3.78,   0.005, "Capital and Liquidity"),
    ("cost as share of gross profit",  3.4,    0.05,  "Capital and Liquidity"),
    ("annual cost",                    0.64,   0.005, "Capital and Liquidity"),
    ("Sharpe at flat 10bp per side",   0.89,   0.005, "Capital and Liquidity"),
    ("Sharpe at flat 20bp per side",   0.82,   0.005, "Capital and Liquidity"),
    ("Sharpe at flat 40bp per side",   0.67,   0.005, "Capital and Liquidity"),
    ("contracts traded per month",     21,     0,     "Execution and data"),
    ("jackknife",                      0.37,   0.006, "Robustness (worst case)"),
    ("best 6 of 183 months",           31.3,   0.05,  "Robustness"),
    ("minimum detectable Sharpe diff", 0.51,   0.005, "Limits"),
    ("worst calendar year",           -12.9,   0.05,  "Losses", 1),
]

# Figures the pitch prints that the numbers file does not yet emit. Each must be
# added to the run before submission, or removed from the document.
NOT_YET_EMITTED = [
    ("longest drawdown, months",        42,    "Losses, Table 1"),
    ("win rate",                        61,    "Table 1"),
    ("profit factor",                   2.04,  "Table 1"),
    ("single-grid mean Sharpe",         0.83,  "Rebalance timing"),
    ("predicted tranched Sharpe",       0.96,  "Rebalance timing"),
    ("calm-market Sharpe",              1.01,  "The control the economics implied"),
    ("volatile-market Sharpe",          0.40,  "The control the economics implied"),
    ("half-risk Sharpe / return",       0.81,  "The control the economics implied"),
    ("parameter grid, cell count",      None,  "Robustness — 15 in the pitch, 12 in the run"),
    ("cross-asset Sharpe, commodities", None,  "Economic Rationale — currently absent"),
]

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse(path: str) -> dict[str, float]:
    """Pull `label   value` pairs out of the numbers file."""
    found: dict[str, float] = {}
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip()
        if not line.startswith("  ") or line.strip().startswith("="):
            continue
        body = line.strip()
        for claim in CLAIMS:
            label, idx = claim[0], (claim[4] if len(claim) > 4 else 0)
            if body.startswith(label) and label not in found:
                nums = NUM.findall(body[len(label):])
                if len(nums) > idx:
                    found[label] = float(nums[idx])
    return found


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/FINAL_NUMBERS.txt"
    actual = parse(path)

    width = max(len(c[0]) for c in CLAIMS) + 2
    print(f"\n  checking the pitch against {path}\n")
    print(f"  {'quantity':<{width}}{'pitch':>10}{'run':>12}   {'':<6}where")
    print("  " + "-" * (width + 40))

    failures, missing = [], []
    for claim in CLAIMS:
        label, claimed, tol, where = claim[0], claim[1], claim[2], claim[3]
        if label not in actual:
            missing.append((label, where))
            print(f"  {label:<{width}}{claimed:>10}{'absent':>12}   {'MISS':<6}{where}")
            continue
        got = actual[label]
        ok = abs(got - claimed) <= tol
        if not ok:
            failures.append((label, claimed, got, where))
        print(f"  {label:<{width}}{claimed:>10}{got:>12}   {'ok' if ok else 'FAIL':<6}{where}")

    if NOT_YET_EMITTED:
        print(f"\n  in the pitch but not emitted by the run — add to the script or cut from the document:")
        for label, val, where in NOT_YET_EMITTED:
            shown = "?" if val is None else val
            print(f"    {label:<34}{str(shown):>8}   {where}")

    print()
    if failures:
        print(f"  {len(failures)} MISMATCH(ES). The document and the run disagree:")
        for label, claimed, got, where in failures:
            print(f"    {label}: pitch says {claimed}, run says {got}  ({where})")
    if missing:
        print(f"  {len(missing)} label(s) not found in the numbers file — check the label text.")
    if not failures and not missing:
        print("  All checked figures agree.")
    print()
    return 1 if (failures or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())