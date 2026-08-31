# The Same Barrel

**Curve-residual momentum in commodity futures.** Sixteen CME contracts, 183 months (June 2011 – August 2026), a $450,000 account, whole contracts, costs built from contract specifications. Sharpe **0.94** net.

Alberto Ramirez-Aguiar · aramirezaguiar@ufl.edu

[![tests](https://github.com/albertoramirez77/algo_app/actions/workflows/tests.yml/badge.svg)](https://github.com/albertoramirez77/algo_app/actions/workflows/tests.yml)

---

## The idea in four sentences

A barrel of oil for next month and a barrel for three months out are the same barrel; the gap between their prices measures how physically tight the commodity is right now. Inventories move slowly — you cannot refill a silo out of season — so a gap that has been tightening tends to keep tightening. Each month the strategy ranks sixteen commodities on the twelve-month change in that gap, buys the tightening curves and sells the loosening ones, sized so each contributes equal risk. It takes no view on the direction of commodity prices: realised beta to the commodity complex is −0.06 and R² is 0.01.

## Headline

| | | | |
|---|---:|---|---:|
| Sharpe, net of costs | **0.94** | Longest drawdown | 42 months |
| *t*-statistic | 3.66 | Winning months / profit factor | 61% / 2.04 |
| Return / volatility | 18.3% / 19.5% | Positions rounding to zero | 17.2% |
| Maximum drawdown | −28.0% | Beta / R² vs. complex | −0.06 / 0.01 |
| Correlation to trend-following | +0.007 | Front momentum / carry, same universe | 0.14 / 0.37 |

Every figure is produced by one run of `engine/final_numbers.py` and committed to `docs/FINAL_NUMBERS.txt`. Run `make verify` to recompute the table above from the committed monthly return series — no data subscription required.

---

## Quick start

**Without a Databento subscription** — verifies the machinery and reproduces the reported performance statistics from committed derived data:

```bash
git clone https://github.com/albertoramirez77/algo_app
cd algo_application
make install
make test        # 179 tests
make smoke       # synthetic-data run of the engine, no API key needed
```

**With a Databento subscription** — rebuilds everything from raw settlements:

```bash
export DATABENTO_API_KEY='db-...'
python data/fetch/fetch_curve.py      # batch job, ~15 minutes
python data/clean/curve_to_px.py      # rebuild on ts_ref, repair roll-date duplicates
make numbers                          # writes docs/FINAL_NUMBERS.txt
make derived                          # regenerates the committed CSVs
make exhibits
```

`make numbers` should reproduce the committed `docs/FINAL_NUMBERS.txt` byte for byte. If it does not, the run is not deterministic and that is a bug.

---

## What is committed, and what is not

Databento's licence does not permit redistributing historical CME data, so no raw price file is in this repository. What *is* here is enough to check every reported number:

| Path | Contents |
|---|---|
| `docs/FINAL_NUMBERS.txt` | the canonical run — every figure quoted in the pitch |
| `docs/figures/` | the exhibits |
| `data/derived/monthly_pnl.csv` | strategy return by month, net of costs |
| `data/derived/pnl_by_instrument.csv` | monthly P&L contribution per instrument |
| `data/derived/signal_ranks.csv` | monthly cross-sectional signal and rank |
| `data/derived/cost_table.csv` | per-instrument notional, tick value, cost in bp |
| `data/derived/benchmarks.csv` | equal-weighted complex, front momentum, carry, trend |

From those five CSVs a reader reproduces the Sharpe, *t*, drawdown, the 42-month underwater period, the regime table, the bootstrap, the jackknife and the portfolio combination — with no vendor data at all. See `data/README.md` for the full fetch specification.

---

## Repository map

```
engine/                     the traded strategy
    universe.py             35 contracts with full specifications; the cost rule
    final_numbers.py        the backtest and every pitch number, one run
    flowbm.py               the production signal, with its validation suite
    immediacy.py            engine for the REJECTED hedger-flow hypothesis (see below)

research/                   the supporting work
    mechanism/              why the deferred contract is the right hedge
    signal_development/     signal variants and the units correction
    validation/             robustness, stress, adversarial audits
    portfolio_construction/ sizing, tranching, neutrality, cost model
    factor_analysis/        factor exposures, residualisation channels, controls
    cross_asset/            the boundary test across FX, rates and equity
    reproducibility/        every pitch number traced to its generating script

failed_research/            hypotheses tested and rejected — read the README
data/                       fetch, clean, diagnose; plus derived/ and the data spec
tests/                      179 tests
exhibits/                   figure generators
docs/                       FINAL_NUMBERS.txt, figures, replication notes
```

---

## What to read first

- **`failed_research/README.md`** — six hypotheses that did not survive, with predictions stated in the docstrings before the results were known. The original thesis of this project was that commercial hedger positioning predicts returns; it produced a slope of 0.003 (*t* = 0.08) against a published benchmark of 4.77 (*t* = 6.55). That null is reported here rather than buried, and it is why the strategy that shipped works through a different channel.
- **`engine/final_numbers.py`, the header** — a net-exposure control that helped one construction and hurt another, recorded as rejected, with an earlier hypothesis of mine explicitly marked wrong.
- **`docs/REPLICATION_NOTES.md`** — three real data defects and their fixes: a Cartesian-product join on roll dates, `ts_recv` vs `ts_ref` timestamps, and the cents-versus-dollars scale error that inflated seven of seventeen notionals by 100×.
- **`research/mechanism/`** — the horse race establishing that the deferred contract of the *same* commodity removes 93.6% of the shared price movement, against 47.5% for eight principal components built from other commodities.

## Design rules enforced in code

1. **Point-in-time inputs.** Every input carries the timestamp at which it became knowable, and signals assert against it.
2. **Whole contracts.** Positions are integers times real multipliers. 17.2% of intended positions round to zero and are not held; that cost is inside every number reported.
3. **No back-adjusted prices.** Returns are chained strictly within a single contract's life. The price gap at a roll is a bookkeeping artefact, not a return.
4. **Ex-ante universe rule.** One instrument is excluded on contract specifications alone, before any return was computed.

## A note on `immediacy.py`

The file is named after this project's original hypothesis — selling immediacy to commercial hedgers, following Kang, Rouwenhorst & Tang (JF 2020). **That hypothesis was rejected** (`failed_research/hedger_flow_null/`). The engine survives because its machinery — the point-in-time clock, integer sizing, the sealed out-of-sample window — is sound and well tested, and because `make smoke` exercises it without needing data. The strategy reported above is produced by `engine/final_numbers.py`, which does not import it.

## References

- Boons & Prado (2019). "Basis-Momentum." *Journal of Finance*. — the documented factor this strategy builds on.
- Kang, Rouwenhorst & Tang (2020). "A Tale of Two Premiums." *Journal of Finance*. — the rejected hypothesis.
- Carver (2015). *Systematic Trading*. Harriman House.

## What this is not

The underlying signal is published. This repository does not claim to have discovered it. What is mine: the evidence for **why** the deferred contract is the right hedge (won 16 of 16 against a hindsight-chosen peer), a test of where the mechanism predicts the effect should be absent (35 instruments, four asset classes), an implementation that survives whole contracts and real costs at $450,000, and a risk control derived from the economics, pre-specified, tested, and rejected.

## Licence

MIT for the code — see `LICENSE`. Market data is subject to Databento's terms and is not redistributed here.
