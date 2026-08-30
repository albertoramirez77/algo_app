# tests/ — Unit Test Suite

All tests pass. Run with:

```bash
pytest tests/ -v
```

## Test Files

| File | Covers |
|------|--------|
| `test_engine_clock.py` | Point-in-time clock rules. COT measured Tuesday close, published Friday 15:30 ET, executable Monday settlement. Guards against look-ahead with strict enforcement. |
| `test_engine_degenerate.py` | Edge cases: empty universes, mismatched schemas, single-instrument books, missing columns. |
| `test_engine_roll_blocks.py` | Roll logic: contract changes, within-contract vs. cross-contract returns, expiry boundaries. |
| `test_data_semantics.py` | Price alignment, COT alignment, signal construction semantics — verifies that every timestamp join is exact and non-anticipating. |
| `test_inference_cluster_ols.py` | Clustered standard errors on synthetic data with known cluster structure and ground-truth coefficients. |
| `test_inference_forward_returns.py` | Forward return computation: overlapping windows (with correct de-overlapping), non-overlapping windows, edge months. |
| `test_inference_monthly_panel.py` | Monthly Fama-MacBeth panel construction, alignment to COT report dates, handling of partial months. |
| `test_inference_placebo.py` | Placebo generation: within-month shuffles, across-month permutations, correct preservation of distribution shape. |

## Test Philosophy

Every test builds a **known-answer synthetic input** and asserts exact output. There are no
mocks of the strategy engine itself — the tests drive real code with constructed data.

`conftest.py` provides shared builders for: price panels, COT data, clustered regression
samples, and roll date sequences.

`reference.py` contains independent implementations of key operations for comparison.
