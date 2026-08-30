# Selling Immediacy in Commodity Futures
### A Cross-Sectional Basis-Momentum Liquidity-Provision Strategy

---

## Strategy Summary

This repository implements and validates a **commodity futures basis-momentum strategy** that earns a liquidity premium by taking the side of commercial hedgers' net position changes. The economic mechanism — intermediaries compensated for absorbing impatient speculator flow — is documented in Kang, Rouwenhorst & Tang (JF 2020) and Boons & Prado (2019).

**Headline Performance** (16-instrument universe, 2010–2025, $450k capital, real costs):

| Metric | Value |
|--------|-------|
| Annualized Sharpe | 0.94 |
| Annualized Return | 18.3% |
| Annualized Volatility | 19.5% |
| Maximum Drawdown | -28% |
| Market Beta (commodity) | -0.06 |
| Turnover | ~3× / year |
| Instruments | 16 (commodity futures, micros where needed) |

Numbers are produced by a single run of `engine/final_numbers.py` on `px_clean.parquet`. Every pitch number is traceable to its generating script.

---

## Repository Structure

```
├── engine/                    # The core strategy — start here
│   ├── universe.py            # 35-instrument universe with all contract specs
│   ├── immediacy.py           # Main backtest engine (clock rules, integer sizing, OOS wall)
│   ├── flowbm.py              # Production signal: units-corrected basis-momentum
│   └── final_numbers.py       # Generates all headline numbers in one run
│
├── research/                  # Supporting research that validates the engine
│   ├── mechanism/             # Economic mechanism investigation
│   ├── signal_development/    # Signal variants and units-correction derivation
│   ├── validation/            # Robustness tests, stress tests, adversarial audits
│   ├── portfolio_construction/ # Sizing, tranching, neutrality, cost model
│   ├── factor_analysis/       # Factor exposures, residualization channels, controls
│   ├── cross_asset/           # Extension to FX, rates, equity
│   └── reproducibility/       # Every pitch number verified and traced
│
├── failed_research/           # Honest accounting of null results (see README inside)
│   ├── hedger_flow_null/      # The original signal: hedger positioning had zero effect
│   ├── delivery_cycle/        # Squeeze and roll-pressure tests (inconclusive)
│   ├── speculator_crowding/   # Crowding hypothesis (did not condition the signal)
│   ├── novel_curve_segments/  # 4-leg identification beyond front/deferred
│   ├── pre_expiry_financials/ # Pre-expiry effect outside commodities
│   ├── test_curve/            # Raw curve momentum baseline
│   └── neural_net_reconstruction/ # Neural net rediscovery test
│
├── data/                      # Data pipeline (Databento + CFTC)
│   ├── fetch/                 # Price and COT fetchers
│   ├── clean/                 # Data repair and verification
│   └── diagnostics/           # API health checks, schema validation
│
├── tests/                     # Pytest unit test suite (all passing)
├── exhibits/                  # Publication-quality figure generators
├── diagnostics/               # Operational health scripts
└── docs/                      # Pitch numbers and final outputs
```

---

## Replication Guide

### Prerequisites

```bash
pip install -r requirements.txt
```

You will need:
- **Databento** subscription (CME Globex, dataset `GLBX.MDP3`) — for price data
- **CFTC** Disaggregated COT files — freely available at cftc.gov (fetched automatically)

### Step 1 — Fetch Data

```bash
# Prices (Databento batch job, ~15 min download)
python data/fetch/fetch_prices_batch.py

# COT data (CFTC disaggregated, 2006–present)
python data/fetch/fetch_cot.py

# Curve (front + deferred settlements, for basis construction)
python data/fetch/fetch_curve.py
```

### Step 2 — Clean & Verify

```bash
# Fix Cartesian-product bug in roll-date merges, verify non-positive settlements
python data/clean/repair.py

# Independent re-implementation verification (catches look-ahead, unit errors)
python data/clean/verify.py

# Rebuild from correct timestamp basis (ts_ref not ts_recv)
python data/clean/curve_to_px.py
```

### Step 3 — Check Data Integrity

```bash
python data/diagnostics/status.py    # "What do I run next?" guidance
python data/diagnostics/check_data.py  # Six pre-backtest integrity checks
```

### Step 4 — Run the Engine

```bash
# Smoke test (synthetic data, no API needed)
python engine/immediacy.py --smoke

# Full backtest (requires real data)
python engine/immediacy.py --run

# Generate all pitch numbers
python engine/final_numbers.py --prices px_clean.parquet > docs/FINAL_NUMBERS.txt

# Verify pitch document matches run
python research/reproducibility/check_pitch_numbers.py
```

### Step 5 — Run Unit Tests

```bash
pytest tests/ -v
```

### Step 6 — Generate Exhibits

```bash
python exhibits/make_exhibits.py --prices px_clean.parquet
python exhibits/riskfigs.py --prices px_clean.parquet
```

---

## Design Principles

This codebase is built around four non-negotiable rules enforced in code:

1. **Point-in-time timestamps** — every input carries the timestamp it became knowable. COT is measured Tuesday close, published Friday 15:30 ET, executable Monday settlement. Signals assert against this.
2. **Integer positions** — all positions are integers times real contract multipliers. No fractional contracts anywhere.
3. **No back-adjusted prices** — every return is computed from consecutive settlement pairs. No spliced series.
4. **Sealed OOS window** — the most recent 25% of the sample is sealed behind an explicit flag (`--oos-unlock`). It may be unlocked once.

---

## Key Results

| Test | Result | Published Benchmark |
|------|--------|---------------------|
| Raw basis-momentum Sharpe | 0.760 (t=2.96) | Boons & Prado (2019): 0.90 |
| Units-corrected signal Sharpe | 0.968 (t=3.77) | — |
| Full strategy (costs, integers, tranched) | 0.94 | — |
| Placebo separation | +2.3 σ above shuffled | — |
| Alpha over momentum factor | 5.59%/yr (t=2.42) | — |
| Market beta (commodity index) | -0.06 | — |
| Hedger flow direct test | slope 0.003, t=0.08 | KRT (2020): slope 4.77, t=6.55 |

The hedger flow null (last row) is a key result. The original hypothesis — that hedger positioning changes would directly predict returns in the 2010–2025 data — showed zero effect. The basis-momentum signal works through a different channel (curve shape persistence), and this is documented honestly in `failed_research/hedger_flow_null/`.

---

## Statistical Methodology

- **Clustered standard errors** (by date) on all panel regressions
- **Fama-MacBeth** for cross-sectional inference
- **Block bootstrap** (12-month blocks) for performance distributions — preserves autocorrelation and loss clustering
- **Pre-registered hypotheses** — `failed_research/` documents predictions made before seeing results
- **Power audit** — every test quantifies expected t-statistic under published effect sizes to confirm tests had power to detect the claimed effects

---

## Six Free Parameters

The strategy has exactly six tunable parameters, all economically motivated:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Formation window | 52 weeks | Published production cycle length |
| Holding period | 3 weeks | Observed half-life of imbalance |
| Stress quantile | 0.75 | KRT capital-loss dummy |
| Vol span | 60 days | Hedger margin recalibration horizon |
| Vol target | 20% | Set by integer granularity, not by return |
| Buffer fraction | 10% | Carver no-trade band |

---

## References

- Kang, Rouwenhorst & Tang (2020). "A Tale of Two Premiums." *Journal of Finance*.
- Boons & Prado (2019). "Basis-Momentum." *Journal of Finance*.
- Carver (2015). *Systematic Trading*. Harriman House.
- Fan (2025). "Currency Basis-Momentum." Working paper.
