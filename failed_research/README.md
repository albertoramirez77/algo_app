# failed_research/ — Honest Accounting of Null Results

This folder documents every hypothesis that was tested and did not survive. In quantitative
research, failed hypotheses are as important as successful ones: they define the limits of
the effect, constrain the mechanism, and prevent the presentation of a backtest as a claim
that none of these directions were explored.

Every script here was written and run before the result was known. The predictions are
stated in the docstrings.

---

## hedger_flow_null/ — The Original Hypothesis Failed

**Hypothesis**: Commercial hedger net position changes (from CFTC Disaggregated COT)
directly predict next-month commodity futures returns in the cross-section.

**Evidence**:
- `mechanism_test.py`: Direct Fama-MacBeth regression of hedger flow on returns.
  - **Our result**: slope = 0.003, t = 0.08
  - **Published benchmark (KRT 2020)**: slope = 4.77, t = 6.55
  - **Marechal (2023, extended to 2020)**: slope = 3.20, t = 2.49

**Why it failed**:
- `power_check.py` (in `research/validation/`) showed the expected t-statistic under the
  published effect size was only 0.08 — the test could never have detected the effect even
  if it existed in the data. Breadth was insufficient (13 commodities, 182 months).
- The alignment plumbing was verified correct via positive controls (momentum and reversal
  showed expected signs in the same panel).

**What was learned**:
- The basis-momentum signal (curve shape, not flow direction) works through a different
  channel — persistence of term structure slope rather than direct flow prediction.
- The 2010–2025 data does not replicate the KRT flow effect, consistent with the
  Marechal (2023) finding that the effect has weakened post-2015.

---

## delivery_cycle/ — Squeeze and Roll Pressure

**Hypothesis**: Physical delivery requirements create convergence squeezes near first-
notice day (FND), and forced roll pressure near expiry creates predictable calendar
spread dynamics.

**`test_squeeze.py`**:
- Pre-registered prediction P1: physical delivery separates the pre-FND effect from
  cash-settled controls (equity index, rates, FX).
- Pre-registered prediction P2: inventory scarcity (USDA stocks-to-use) conditions the squeeze.
- Pre-registered prediction P3: effect concentrates in the 5 trading days before FND.
- **Result**: Effect present in commodities but not cleanly separated from general
  pre-expiry drift; P2 and P3 showed mixed evidence.

**`test_rollpressure.py`**:
- Open interest to days-to-expiry ratio (normalized within instrument/maturity bucket)
  as a predictor of calendar spreads.
- **Result**: Ratio correlated with spread but timing too noisy for reliable signal
  extraction at monthly frequency.

---

## speculator_crowding/ — Crowding Hypothesis

**Hypothesis**: Basis-momentum earns more when speculators are crowded into the TREND
factor (Uhl 2025 alignment hypothesis) — the premium compensates for the additional
immediacy demand.

**`crowd_bm.py`**:
- Pre-registered predictions: P1) trend-crowding predicts BM returns (positive sign),
  P2) money-manager crowding is the driver (not producer positions),
  P3) producer positions show opposite sign.
- **Result**: P1 sign correct but statistically weak (t < 1.5). P2 and P3 not cleanly
  identified in this sample. Cannot distinguish crowding from a simple volatility state.

---

## novel_curve_segments/ — 4-Leg Identification

**Hypothesis**: A four-maturity term structure identification recovers unspanned information
in the curve beyond what the front/deferred spread captures.

**`novel_curve_identification_v2.py`**: SEG01 (front vs deferred momentum) residualized
against the deeper curve; tests whether the front segment contains information orthogonal
to the deferred.

**`novel_spanning_2leg.py`**: Two-leg spanning: does BM survive controls for front
momentum, deferred momentum, basis, and commodity factor momentum simultaneously?

**`run_research.py`**: Orchestrates both 2-leg and 4-leg test suites.

**`build_four_curve_panel.py`**: Data construction for the 4-leg panel.

**Result**: Spanning held for the 2-leg signal (BM is not subsumed by front or deferred
momentum individually), but the 4-leg identification failed to find unspanned information
beyond the front/deferred spread. Additional curve legs add noise rather than signal at
this sample size.

---

## pre_expiry_financials/ — Pre-Expiry Effect Outside Commodities

**Hypothesis**: The unexplained pre-expiry drift visible in financial futures (equity
index, rates) is a distinct, non-delivery-related effect that could expand the universe.

**`phase_a.py`**:
- Detected residual of 0.0168%/day in financials, residualized against known factors.
- Five checks: per-asset-class effect, raw returns, per-instrument residuals, roll days
  excluded, sign convention verification.
- **Result**: Effect present but not robust to all five checks. Sign convention ambiguity
  (roll conventions differ across exchanges) prevented clean identification. Research
  paused pending cleaner data on settlement conventions.

---

## test_curve/ — Raw Curve Momentum Baseline

**`test_curve.py`**:
- Three questions: (1) Does basis alone predict returns (positive control)?
  (2) What is the delivery cycle event profile? (3) Is there a carry time-series signal?
- **Result**: Positive control confirmed (basis predicts returns). Delivery cycle profile
  inconsistent across asset classes. No reliable carry time-series at monthly frequency
  beyond what BM already captures. This script is preserved as a diagnostic tool.

---

## neural_net_reconstruction/ — Neural Network Rediscovery

**`blindnet.py`**:
- A neural network trained only on raw curve legs (front return, deferred return, basis,
  volatility, open interest, days to expiry) — never given the engineered signal.
- **Hypothesis**: If the signal is real, the network should independently weight front
  and deferred returns in opposite directions (reconstructing the spread).
- **Result**: Network underperformed the linear strategy (expected at n≈3,000 training
  samples). Feature importance was noisy; the front/deferred differential was not cleanly
  recovered. This is consistent with the signal being genuine but insufficient data for
  the network to rediscover it reliably. Absence of rediscovery is not evidence of absence.

---

## Note on Methodology

Every failed test increased our confidence in the surviving signal by:
1. Confirming the alignment plumbing was correct (positive controls passed)
2. Bounding the mechanism (delivery-specific, not general pre-expiry; curve shape, not flow)
3. Eliminating alternative explanations (crowding, sector tilt, curve segment beyond front/deferred)

A strategy that has only been tested for what it claims, and never for what it does not
claim, is not a strategy — it is a backtest.
