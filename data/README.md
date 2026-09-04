# data/ — Data Pipeline

All data originates from two sources:
- **Databento** (CME Globex, dataset `GLBX.MDP3`) — price settlements, open interest, definitions
- **CFTC** — Disaggregated Commitments of Traders (Futures Only)

---

## The actual pipeline

```
fetch_curve.py  ──Databento──▶  px_curve.parquet  ──curve_to_px.py──▶  px_clean.parquet  ──▶  everything else
fetch_cot.py    ──CFTC──────▶  cot.parquet                                                  ──▶  scripts that take --cot
```

`px_clean.parquet` is the file every `--prices` script in `research/`, `exhibits/`, `engine/`,
`failed_research/` and `data/` defaults to. It is committed to the repo (`data/px_clean.parquet`), so
everything downstream of it — every script that only takes `--prices` — runs with **no Databento key
and no network access**. Only the two fetch steps above touch Databento or the CFTC.

```bash
# Regenerate px_clean.parquet from scratch (needs a Databento key)
export DATABENTO_API_KEY="your-key-here"
python data/fetch/fetch_curve.py --submit
python data/fetch/fetch_curve.py --status     # poll until done
python data/fetch/fetch_curve.py --download --out data/px_curve.parquet
python data/clean/curve_to_px.py --in data/px_curve.parquet --out data/px_clean.parquet

# cot.parquet (needed only by scripts that take --cot) — free, no key
python data/fetch/fetch_cot.py

# Everything else runs off the committed data/px_clean.parquet, no key needed
make test
make verify
```

### Legacy path (superseded, kept for reconciliation)

An earlier fetcher (`fetch_prices.py` / `fetch_prices_batch.py`) built `px.parquet` dated from
`ts_recv` (wall-clock receipt) rather than the settlement session, which put 16.9% of rows on a
Sunday. Its wide-format output, `px_wide.parquet`, also had a roll-date merge bug that produced
6,007 duplicate rows. `data/clean/repair.py` fixes that bug and produces its own `px_clean.parquet`
from `px_wide.parquet` — this is a separate, older lineage from the `fetch_curve.py` /
`curve_to_px.py` path above and is kept only so the repair can be independently verified; it is not
part of the pipeline new data should go through.

---

## fetch/ — needs `DATABENTO_API_KEY` (all four scripts)

| Script | What it does |
|--------|-------------|
| `fetch_curve.py` | **Current fetcher.** Submits/polls/downloads Databento batch jobs for front and second contract settlements plus front-contract expiry definitions via continuous symbology. Outputs `px_curve.parquet`. |
| `fetch_cot.py` | Downloads CFTC Disaggregated COT zip files (2006–present), handles .xls/.xlsx/.txt/.csv format changes across years, caches as `cot.parquet`. **No Databento key — CFTC data is free and public.** |
| `fetch_prices.py` | Legacy streaming fetcher, superseded by `fetch_curve.py`. Produces the deprecated `px.parquet` (ts_recv-dated). |
| `fetch_prices_batch.py` | Legacy batch fetcher, superseded by `fetch_curve.py`. Produces the deprecated `px_wide.parquet`. |

```bash
export DATABENTO_API_KEY="your-key-here"
python data/fetch/fetch_cot.py                      # free, no key needed, run any time
python data/fetch/fetch_curve.py --submit            # needs key
```

---

## clean/ — no Databento key needed (reads local files only)

| Script | What it does |
|--------|-------------|
| `curve_to_px.py` | **Current.** Converts `px_curve.parquet` (ts_ref-dated, zero weekend rows, 252 sessions/year) into the two-leg schema the rest of the codebase expects. Produces `px_clean.parquet`. |
| `repair.py` | Legacy repair for the `px_wide.parquet` roll-date bug (see "Legacy path" above). Produces its own `px_clean.parquet` from `px_wide.parquet`. |
| `verify.py` | Independent re-implementation of the pipeline from scratch: no look-ahead, signal construction, portfolio mechanics, headline reproduction, economic claims, edge cases. Exits non-zero on any divergence. |
| `diagnose.py` | Pinpoints specific `verify.py` failures and determines whether they touch the commodity strategy or only the wider research universe. |
| `verify_listing.py` | Queries Databento to find actual micro contract listing dates vs. assumed dates. Needs a Databento key. Run when adding new instruments. |

```bash
python data/clean/curve_to_px.py --in data/px_curve.parquet --out data/px_clean.parquet
python data/clean/verify.py --prices data/px_clean.parquet   # should exit 0
```

---

## diagnostics/

| Script | Needs a key? | What it does |
|--------|:---:|-------------|
| `status.py` | no | Checks every artifact the backtest needs, reports what is present/missing, and prints the single next command to run. Safe to run at any time; costs nothing. |
| `check_data.py` | no | Six pre-backtest integrity checks against `--cot`/`--prices`: coverage, COT continuity, forward return validity, roll date integrity, basis reasonableness, micro contract cost sensitivity. |
| `dbn_diagnose.py` | **yes** | Databento API health check. Tests `get_record_count` on all 13 products before spending download credit. |
| `dbn_speedtest.py` | **yes** | Two small requests measuring connection speed (~110 rec/sec) and payload size. |
| `dbn_strategy_test.py` | **yes** | Tests three Databento request shapes (parent, OI-rank, calendar) to determine which symbology resolves correctly for this strategy. |

```bash
python data/diagnostics/status.py    # Always run this first
```

---

## Output Files

| File | Description | Committed? |
|------|-------------|:---:|
| `px_curve.parquet` | Curve panel: front and second contract settlements with expiry dates, ts_ref-dated | no |
| `px_clean.parquet` | Clean two-leg price panel that every `--prices` script defaults to | **yes** (`data/px_clean.parquet`) |
| `cot.parquet` | COT data: weekly hedger and speculator positions, 2006–present | no |
| `px.parquet`, `px_wide.parquet` | Deprecated intermediates from the legacy fetch/repair path | no |

Everything except `data/px_clean.parquet` is regenerated locally and is not committed to git (too
large, or superseded).

---

## Makefile targets

`make mechanism`, `make channels`, `make crossasset`, `make validate` and `make check` each run the
matching research script(s) with `PRICES ?= data/px_clean.parquet`, so they work against the
committed file with no arguments and no Databento key:

```bash
make mechanism    # research/mechanism/{pathbm,decompose_bm}.py
make channels     # research/factor_analysis/channels.py
make crossasset    # research/cross_asset/crossasset.py
make validate      # research/validation/validate_bm.py
make check         # data/diagnostics/check_data.py — needs cot.parquet (make ... after fetch_cot.py)
```

Override the price file with `make <target> PRICES=path/to/other.parquet`.
