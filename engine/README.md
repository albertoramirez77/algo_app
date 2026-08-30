# engine/ — The Core Strategy

This folder contains the production implementation. Everything a replicator needs is here.

## Files

| File | Purpose |
|------|---------|
| `universe.py` | 35-instrument universe across 6 asset classes. Every contract spec: symbol, multiplier, tick, commission, CFTC code, price scale correction. The 13-commodity tradeable subset is flagged `tradeable_450k=True`. |
| `immediacy.py` | Main backtest engine. Enforces the four non-negotiable rules: point-in-time timestamps, integer contracts, no back-adjusted prices, sealed OOS window. Run `--smoke` for a synthetic-data sanity check; `--run --oos-unlock` for the real backtest (unlock once). |
| `flowbm.py` | Production signal. Units-corrected basis-momentum: accumulates within-contract monthly spread returns scaled by 365.25/gap, never crossing a roll date. Sharpe 0.968 vs 0.760 for the raw signal. Run standalone to see the full validation suite. |
| `final_numbers.py` | Every pitch number in one script. Run this last; it asserts against a frozen spec and prints a labeled table. Pipe to `docs/FINAL_NUMBERS.txt`. |

## Quick Start

```bash
# Synthetic smoke test (no data needed, verifies machinery)
python engine/immediacy.py --smoke

# Validate the units-corrected signal
python engine/flowbm.py --prices ../px_clean.parquet

# Generate all headline numbers
python engine/final_numbers.py --prices ../px_clean.parquet > ../docs/FINAL_NUMBERS.txt
```

## Design Rules (Enforced in Code)

### 1. Point-in-Time Clock
COT is measured Tuesday close → published Friday 15:30 ET → executable Monday settlement.
The engine asserts that no signal can be computed before the report publication timestamp.
See `tests/test_engine_clock.py` for the full clock specification.

### 2. Integer Positions
`pos = round(raw_size / multiplier)` and positions are stored as `int64`. Any fractional
contract arising from the optimizer is rounded; the buffer prevents excessive trading.
See `tests/test_engine_roll_blocks.py`.

### 3. No Back-Adjusted Prices
Returns are computed as `log(settle_t / settle_{t-1})` within a continuous contract.
Cross-contract returns are blocked at expiry. The roll is NOT adjusted into the prior series.
This matches what a live trader would see.

### 4. Sealed OOS Window
`OOS_FRACTION = 0.25` (most recent 25% of the sample) is sealed. The engine raises if
you attempt to view OOS without `--oos-unlock`. This flag should only be passed once,
after all specification decisions are final.

## Performance (Full Run)

```
Sharpe (annualized)     0.94
Return (annualized)    18.3%
Volatility             19.5%
Max Drawdown          -28.0%
Market Beta            -0.06
Alpha / Momentum      5.59%/yr  (t = 2.42)
Placebo separation    +2.3 σ
```
