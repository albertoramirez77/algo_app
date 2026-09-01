# data/ — Data Pipeline

All data originates from two sources:
- **Databento** (CME Globex, dataset `GLBX.MDP3`) — price settlements, open interest, definitions
- **CFTC** — Disaggregated Commitments of Traders (Futures Only)

---

## Pipeline Order

```
fetch/ → clean/ → [px_clean.parquet, cot.parquet] → engine/
```

Run in it the exact order, each step validates its output before passing it downstream.

---

## fetch/

| Script | What it does |
|--------|-------------|
| `fetch_prices_batch.py` | Submits 13 batch jobs to Databento (safe restart via `jobs.json`). Downloads continuous front and deferred settlements for all commodity instruments. ~15 min wall time, ~$2–5 Databento credit. |
| `fetch_prices.py` | Streaming fallback fetcher. Year-by-year chunking with bisection on failure. Use this if batch jobs fail. |
| `fetch_curve.py` | Fetches front and second contract settlements plus front expiry definitions via continuous symbology. Outputs `px_curve.parquet`. |
| `fetch_cot.py` | Downloads CFTC Disaggregated COT zip files (2006–present), handles .xls/.xlsx/.txt/.csv format changes across years, caches as `cot.parquet`. Free to run at any time. |

```bash
# Recommended order
python data/fetch/fetch_cot.py                    # Free, run first
python data/fetch/fetch_curve.py                  # Requires Databento key
python data/fetch/fetch_prices_batch.py           # Requires Databento key
```

Set your Databento API key:
```bash
export DATABENTO_API_KEY="your-key-here"
```

---

## clean/

| Script | What it does |
|--------|-------------|
| `repair.py` | Fixes a Cartesian product bug in roll-date merges that produced 6,007 duplicate rows. Applies continuity rules to select the correct row when duplicates exist. Verifies the repair does not change strategy headline numbers. |
| `verify.py` | Independent re-implementation of the entire data pipeline from scratch. Checks: no look-ahead (future prices), signal construction, portfolio mechanics, headline reproduction, economic claims, edge cases. Exits non-zero on any divergence. |
| `diagnose.py` | Pinpoints specific verification failures from `verify.py` and determines whether they touch the commodity strategy or only the wider research universe. |
| `verify_listing.py` | Queries Databento to find actual micro contract listing dates (MCL 2021-07, MGC 2010-10, SIL 2013-03, MHG 2022-05) vs. assumed dates. Run when adding new instruments. |
| `curve_to_px.py` | Rebuilds the price file using `ts_ref` (reference timestamp = the settlement date) rather than `ts_recv` (the API delivery timestamp). Eliminates Sunday rows and the 305-vs-252 session mismatch that inflated apparent coverage. |

```bash
# After fetching
python data/clean/repair.py
python data/clean/curve_to_px.py
python data/clean/verify.py          # Should exit 0
```

---

## diagnostics/

| Script | What it does |
|--------|-------------|
| `status.py` | Checks every artifact the backtest needs, reports what is present/missing, and prints the single next command to run. Safe to run at any time; costs nothing. |
| `check_data.py` | Six pre-backtest integrity checks: coverage (all products over full period), COT continuity (gap detection), forward return validity, roll date integrity, basis reasonableness, micro contract cost sensitivity. |
| `dbn_diagnose.py` | Databento API health check. Tests `get_record_count` on all 13 products before spending download credit. |
| `dbn_speedtest.py` | Two small requests measuring connection speed (~110 rec/sec) and payload size (identifies calendar spread bloat). |
| `dbn_strategy_test.py` | Tests three Databento request shapes (parent, OI-rank, calendar) to determine which symbology resolves correctly for this strategy. |

```bash
python data/diagnostics/status.py    # Always run this first
```

---

## Output Files

| File | Description |
|------|-------------|
| `cot.parquet` | COT data: weekly hedger and speculator positions, 2006–present |
| `px_clean.parquet` | Clean price panel: daily settlements, ts_ref basis, no duplicates |
| `px_curve.parquet` | Curve panel: front and second contract settlements with expiry dates |
| `px_wide.parquet` | Wide-format price panel (pre-repair, kept for reconciliation) |

These files are not committed to git (too large), so they are generated locally by running the pipeline above.
