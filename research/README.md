# research/ — Supporting Research

Every subfolder here validates, extends, or challenges some aspect of the core strategy.
The strategy claimed nothing that isn't backed by at least one script here.

---

## mechanism/
**Question**: Why does basis-momentum work?

| Script | Tests |
|--------|-------|
| `mechanism_bm.py` | Interaction regressions: does the basis-momentum effect strengthen under high basis, high volatility, or high illiquidity (Amihud)? Validates Boons & Prado's liquidity-provision mechanism in modern data. |
| `pathbm.py` | Variance decomposition (BM = front momentum − deferred momentum, cancels spot beta). Path consistency test: monotone curve moves vs. shocks (Kaufman efficiency ratio). |
| `decompose_bm.py` | P&L attribution showing where each Sharpe improvement came from: raw ranks (0.602) → vol scaling (0.792) → integer contracts (0.848) → costs → volatility timing → entry price. |

---

## signal_development/
**Question**: Is there a better formulation of the signal?

| Script | Tests |
|--------|-------|
| `normbm.py` | Level normalization: divide basis by maturity gap before differencing. **Fails** — jumps at every roll date as the denominator changes mechanically with the listing calendar. SR −0.055. Documents the failure cleanly. |
| `tvbm.py` | Time-varying correction validation. Decomposes gap weighting into static (per-instrument tilt) and time-varying components adversarially to show the correction is not just a sector tilt. |
| `break_flowbm.py` | Eight adversarial attacks on the units correction: random constants, reverse scaling, sector neutralization, within-sector gains, shuffled gaps, static gaps, equal weights, quintile ranks. All eight fail to replicate the gain. |

The units-corrected signal (`flowbm.py` in `engine/`) survived all eight attacks.

---

## validation/
**Question**: Does the signal survive systematic stress testing?

| Script | Tests |
|--------|-------|
| `validate_bm.py` | Initial validation: Sharpe 0.602, t=2.34, placebo +2.3 σ away, correct turnover measurement with symbol indexing. |
| `backtest_bm.py` | Full backtest with integer contracts and real costs. |
| `stress_bm.py` | Jackknife (drop each instrument one at a time), parameter grid (52 formation/vol window/target combinations), Boons & Prado decay check (published 2015), weighting scheme comparison (rank vs. quintile vs. equal), costs swept to 20 bp. |
| `audit.py` | Seven adversarial attacks: listing date sensitivity, multiplier sanity, parameter grid bias, persistent sector tilts, convexity under stress, single-month P&L concentration, recent subperiod honesty. |
| `power_check.py` | Power audit: confirms each test had a detectable expected t-statistic under the published effect size. Hedger flow test had expected t=0.08 — it could never have worked. |
| `controls_check.py` | Panel integrity: momentum should flip sign pre/post-2011 (crowding effect). Validates that the alignment machinery is correct before testing anything else. |
| `test_basismomentum.py` | Benchmark comparison: Boons & Prado (2019) SR 0.9, Fan (2025) currency basis-momentum SR 0.52. |

---

## portfolio_construction/
**Question**: How is the book sized, tranched, and cost-adjusted?

| Script | Answers |
|--------|---------|
| `sizing.py` | IDM (instrument diversification multiplier) = 1/√(w′Rw) analytically; measured value 3.96 (vs. 2.5 assumed). Strategy running at 63% of intended risk. |
| `tranche.py` | 21 overlapping daily grids averaged before rounding. Reduces daily position volatility 74%, same mean return. Standard industry practice (Carver). |
| `tranche2.py` | Corrected construction: marks daily on averaged position, not on averaged signals. Fixes smoothed-artifact error in `tranche.py`. |
| `neutrality.py` | Clarifies "dollar-neutral": the strategy is actually risk-neutral (inverse-volatility weighted), not dollar-neutral. Net notional can reach ±50% of NAV — measured and disclosed. |
| `netcap.py` | Demonstrates that a single constant applied to all weights achieves exact net-notional zero without reordering. Tested on single-grid book (improved drawdown); rejected on tranched book (hurt Sharpe by 0.11). |
| `micros.py` | Optimistic/realistic/conservative versions of the micro-contract assumption. Strategy survives all three. |
| `speedlimit.py` | Excludes instruments where per-side cost > ex-ante threshold (QG at 27.92 bp). Rule uses only contract specs, never realized performance. |
| `regime_cost.py` | Bottom-up cost model: half-spread (one tick) + slippage (one tick) + commission, per instrument. Characterizes sample by regime (momentum, stress, recovery). |

---

## factor_analysis/
**Question**: What explains the returns, and are they spanned by known factors?

| Script | Tests |
|--------|-------|
| `channels.py` | Three residualization channels — curve (front vs. deferred), cross-section (rank vs. peer), sector (energy/grain/metal) — with variance removed per regressor. Curve hedging is uniquely efficient. |
| `proximity.py` | Hedge quality by economic proximity: deferred contract vs. crush/crack spread vs. cherry-picked peer vs. principal components. |
| `exposures.py` | Eight factor exposures: commodity market, USD (sign-flipped), rates level, curve slope, equity market, equity volatility, commodity carry, sector tilts. Both univariate and multivariate, rolling and conditional. |
| `controls.py` | Net sector cap, dollar hedge overlay (trailing 36-month beta regression), strategy decay since 2019, volatility-state conditioning. |
| `whence.py` | Tranching gain audit: where does the vol reduction come from? Also tests an economically-derived control: reduce sizing in high cross-sectional volatility (inventory shock regime). |
| `legs_bm.py` | Asymmetry test: long leg (backwardation) vs. short leg (contango into delivery wall). Tests whether storage bounds create one-sided predictability. |
| `asym_bm.py` | 2×2 decomposition: long/short × backwardated/contangoed. Detects storage-bound asymmetry vs. generic carry. |

---

## cross_asset/
**Question**: Does basis-momentum generalize beyond commodities?

| Script | Tests |
|--------|-------|
| `crossasset.py` | Three-tier universe expansion: Tier 1 (17 commodities), Tier 2 (+8 FX), Tier 3 (+6 rates, 4 equity). Tests two competing mechanisms: curve efficiency (general) vs. inventory scarcity (commodity-specific). |
| `novel_spanning_2leg_v2.py` | Valid 2-leg diagnostic on `px_clean.parquet`. Narrow question: does maturity differential survive common momentum and current basis as controls? |

---

## reproducibility/
**Question**: Does every pitch number trace back to a single, reproducible run?

| Script | Purpose |
|--------|---------|
| `final_numbers.py` | Moved to `engine/` — the canonical pitch number generator. |
| `regenerate.py` | One specification, one file, one run. Resolves the 0.756-vs-0.586 inconsistency (data repair) and 13.07%-vs-15.54% IDM source discrepancy. |
| `reconcile.py` | Quantifies the impact of the data repair: compares frozen spec on `px_wide.parquet` vs. `px_clean.parquet`. |
| `check_pitch_numbers.py` | CI assertion: every number in the pitch document matches the run that produced it. Exits 0 on agreement, 1 on divergence. Wire into CI. |
| `attribute.py` | P&L attribution: does the signal contribute, or does the portfolio machinery earn everything? Placebo test shuffles signal across names within week. |
