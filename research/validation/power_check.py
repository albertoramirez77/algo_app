"""
power_check.py — was the test capable of finding what it was looking for?

    python power_check.py

A null result means nothing until you know the minimum effect the test could have
detected. Two of the tests in this project returned "no effect" and they carry completely
different evidentiary weight:

    hedger net trading   expected t under the published effect: 122   -> decisive null
    basis / carry        expected t under the published effect: 0.79  -> no information

The first is evidence of absence. The second is absence of evidence. Reporting them the
same way would be the single most misleading thing this project could do.

This should have been run before the first backtest, not after the fourth. It is
result-independent: every number below comes from the standard error of the test and the
published effect size, not from what the test happened to return.
"""

from __future__ import annotations

import numpy as np

# Each entry: what we measured, and what the literature says the effect is, in the SAME
# units as the measured slope. The published figures are converted to per-unit-of-regressor
# terms so that they are directly comparable.
TESTS = [
    dict(
        name="Hedger net trading (Q)",
        slope=-0.011, t=-0.28, n=631,
        published=4.77,
        source="Kang, Rouwenhorst & Tang, JF 2020, Fama-MacBeth slope",
        note="Marechal (JFM 2023) replicates at 3.80 risk-adjusted; 3.20 extended to 2020.",
    ),
    dict(
        name="Hedger quintile spread",
        slope=-0.00158, t=-0.67, n=631,
        published=0.0045,
        source="KRT days 5-20 high-minus-low quintile, +0.45%",
        note="Cruder statistic: discards the middle of the cross-section.",
    ),
    dict(
        name="Basis / carry (standardised)",
        slope=-0.0050, t=-1.19, n=2480,
        published=0.0033,
        source="Szymanowska et al., JF 2014: spot premia 5-14%/yr from basis sorts",
        note="5-14%/yr across ~2.5 sd of basis is ~0.33%/month per sd. Their sample is "
             "21 commodities over 300 months; ours is 13 over 194.",
    ),
    dict(
        name="Delivery cycle, near minus far",
        slope=-0.000224, t=-2.16, n=50184,
        published=None,
        source="No published benchmark — this hypothesis is ours",
        note="Placebo mean +0.86 sd 0.43, so the measurement is ~7 placebo-sd from noise. "
             "But the sign is OPPOSITE to the forced-exit hypothesis.",
    ),
]


def audit(t: dict) -> dict:
    se = abs(t["slope"] / t["t"]) if t["t"] else np.nan
    mde = 2.0 * se                                    # smallest slope reaching t = 2
    out = dict(t, se=se, mde=mde)
    if t["published"] is not None:
        out["exp_t"] = t["published"] / se
        out["power_ratio"] = t["published"] / mde     # >1 means detectable
        # SE scales as 1/sqrt(N). How many times more observations to reach exp_t = 3?
        out["obs_multiple"] = (3.0 / out["exp_t"]) ** 2 if out["exp_t"] > 0 else np.inf
    return out


def verdict(a: dict) -> str:
    if a["published"] is None:
        return "no benchmark — judge against the placebo, not against power"
    if a["exp_t"] >= 3:
        return "DECISIVE: the effect would have been unmissable. The null is evidence."
    if a["exp_t"] >= 2:
        return "ADEQUATE: borderline. Treat the null as suggestive, not conclusive."
    return "UNDERPOWERED: this test could not have found the effect. The null is silence."


def main() -> None:
    print("=" * 78)
    print("STATISTICAL POWER AUDIT")
    print("=" * 78)
    print("  exp_t  = t-statistic we would expect IF the published effect were present")
    print("  mde    = smallest slope this test could have detected at t = 2")
    print()

    for t in TESTS:
        a = audit(t)
        print("-" * 78)
        print(f"{a['name']}")
        print(f"  measured   slope {a['slope']:+.5g}   t {a['t']:+.2f}   n {a['n']:,}")
        print(f"  std error  {a['se']:.5g}")
        print(f"  detectable {a['mde']:+.5g} at t=2")
        if a["published"] is not None:
            print(f"  published  {a['published']:+.5g}   ({a['source']})")
            print(f"  EXPECTED t IF PUBLISHED EFFECT HELD:  {a['exp_t']:.2f}")
            print(f"  power ratio (published / detectable): {a['power_ratio']:.2f}x")
            if a["exp_t"] < 3:
                print(f"  observations needed for exp_t = 3:    "
                      f"{a['obs_multiple']:.1f}x current")
        else:
            print(f"  benchmark  none  ({a['source']})")
        print(f"  -> {verdict(a)}")
        print(f"     {a['note']}")

    print("\n" + "=" * 78)
    print("WHAT FOLLOWS")
    print("=" * 78)
    print("""
  The hedger null is real and quantified. The test could detect an effect one sixtieth
  the published size and found nothing, on both sides of the transfer, at every horizon,
  in every subperiod, with a placebo confirming the signal added nothing. That belongs in
  the pitch as a finding.

  The basis null is not a finding. Expected t under the published effect is below one.
  It says nothing about whether carry works — only that 13 instruments over 194 months
  cannot answer the question with a within-instrument monthly regression.

  THE FIX IS NOT MORE INSTRUMENTS ALONE. Reaching exp_t = 3 on that regression needs
  roughly 14x the observations, which 13 -> 35 names does not deliver.

  The fix is to test carry the way the literature tests it: at the PORTFOLIO level. A
  diversified carry portfolio aggregates the cross-section into one return series, and
  Koijen, Moskowitz, Pedersen & Vrugt report a Sharpe near 0.9 for global carry. A Sharpe
  of 0.9 over 16 years carries t = 0.9 * sqrt(16) = 3.6 — detectable, where the
  per-instrument regression is not.

  That is the same data, the same hypothesis, and a test with four times the power.
""")


if __name__ == "__main__":
    main()