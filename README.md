# The Same Barrel

**Curve-residual momentum in commodity futures.** Sixteen CME contracts, 183 months (June 2011 – August 2026), a $450,000 account, whole contracts, costs built from contract specifications. Sharpe **0.94** net.

Alberto Ramirez-Aguiar · aramirezaguiar@ufl.edu

[![tests](https://github.com/albertoramirez77/algo_app/actions/workflows/tests.yml/badge.svg)](https://github.com/albertoramirez77/algo_app/actions/workflows/tests.yml)

---

## The idea in four sentences

A barrel of oil for next month and a barrel for three months out are the same barrel, and the gap between their prices measures how physically tight the commodity is right now. Inventories move slowly, you cannot refill a silo out of season, so a gap that has been tightening tends to keep tightening. Each month the strategy ranks sixteen commodities on the twelve-month change in that gap, buys the tightening curves and sells the loosening ones, sized so each contributes equal risk. It takes no view on the direction of commodity prices as realized beta to the commodity complex is −0.06 and R^2 is 0.01.

## Headline

| | | | |
|---|---:|---|---:|
| Sharpe, net of costs | **0.94** | Longest drawdown | 42 months |
| *t*-statistic | 3.66 | Winning months / profit factor | 61% / 2.04 |
| Return / volatility | 18.3% / 19.5% | Positions rounding to zero | 17.2% |
| Maximum drawdown | −28.0% | Beta / R² vs. complex | −0.06 / 0.01 |
| Correlation to trend-following | +0.007 | Front momentum / carry, same universe | 0.14 / 0.37 |

Every figure is produced by one run of `engine/final_numbers.py` and committed to `docs/FINAL_NUMBERS.txt`. Run `make verify` to recompute the table above from the committed monthly return series — no data subscription required.

<img width="2200" height="1320" alt="image" src="https://github.com/user-attachments/assets/6435b1bf-fa32-4763-a03a-31e97b077e9b" />

---

## Quick start

**Without a Databento subscription** — verifies the machinery and reproduces the reported performance statistics from committed derived data:
```bash
git clone https://github.com/albertoramirez77/algo_app
cd algo_app
make install
make test        # 179 tests
make verify      # recomputes headline table
make smoke       # synthetic-data run of the engine, no API key needed
```

**With a Databento subscription** — rebuilds everything from raw settlements:
** (I used $125 free credits from Databento, this took ~$10 total) **

```bash
export DATABENTO_API_KEY='db-...'
python data/fetch/fetch_curve.py      # batch job, ~15 minutes
python data/clean/curve_to_px.py      # rebuild on ts_ref, repair roll-date duplicates
make numbers                          # writes docs/FINAL_NUMBERS.txt
make derived                          # regenerates the committed CSVs
make exhibits
```

`make numbers` should reproduce the committed `docs/FINAL_NUMBERS.txt` byte for byte. If it does not, the run is not deterministic and that is a bug.

**run the command make all to regenerate every number and figure in the pitch from raw data** 

---
## What is committed, and what is not

Databento's license realistically doesn't permit redistributing historical CME data, so no raw price file is in this repo. What is here is enough to check every reported number:

| Path | Contents |
|---|---|
| `docs/FINAL_NUMBERS.txt` | the canonical run — every figure quoted in the pitch |
| `docs/figures/` | the exhibits |
| `data/derived/monthly_pnl.csv` | strategy return by month, net of costs |
| `data/derived/pnl_by_instrument.csv` | monthly P&L contribution per instrument |
| `data/derived/signal_ranks.csv` | monthly cross-sectional signal and rank |
| `data/derived/cost_table.csv` | per-instrument notional, tick value, cost in bp |
| `data/derived/benchmarks.csv` | front momentum, carry, trend |

From those five CSVs one can reproduce the Sharpe, t-stat, drawdown, the 42-month underwater period, the regime table, the bootstrap, the jackknife, with no vendor data at all. But see `data/README.md` for the full fetch specification.

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
- **`docs/REPLICATION_NOTES.md`** — three real data defects and their fixes: a Cartesian-product join on roll dates, `ts_recv` vs `ts_ref` timestamps, and the cents-versus-dollars scale error that inflated seven of seventeen notionals by 100×.
- **`research/mechanism/`** — the horse race establishing that the deferred contract of the *same* commodity removes 93.6% of the shared price movement, against 47.5% for eight principal components built from other commodities.

<img width="2200" height="1000" alt="image" src="https://github.com/user-attachments/assets/07273066-6fc8-4381-8814-c693f55e0eed" />


## Design rules enforced in code

1. **Point-in-time inputs.** Every input carries the timestamp at which it became knowable, and signals assert against it.
2. **Whole contracts.** Positions are integers times real multipliers. 17.2% of intended positions round to zero and are not held, and that cost is inside every number reported.
3. **No back-adjusted prices.** Returns are chained strictly within a single contract's life. The price gap at a roll is a bookkeeping artifact, not a return.
4. **Ex-ante universe rule.** One instrument is excluded on contract specifications alone, before any return was computed.

## References

- Boons & Prado (2019). "Basis-Momentum." *Journal of Finance*.

## What this is not

The underlying signal is published, this repo does not claim to have discovered it. What is novel is the evidence for **why** the deferred contract is the right hedge (won 16 of 16 against a hindsight-chosen peer), a test of where the mechanism predicts the effect should be absent (35 instruments, four asset classes), an implementation that survives whole contracts and real costs at $450,000, and a risk control derived from the economics, pre-specified, tested, and rejected.

## License

MIT for the code — see `LICENSE`. Market data is fully subject to Databento's terms and is not redistributed here.
