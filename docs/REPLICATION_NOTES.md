# Replication Notes

## Numbers Reference

The files `FINAL_NUMBERS.txt` and `PITCH_NUMBERS.txt` are generated outputs, not committed
to this repository. To regenerate:

```bash
python engine/final_numbers.py --prices px_clean.parquet > FINAL_NUMBERS.txt
python research/reproducibility/regenerate.py --prices px_clean.parquet > PITCH_NUMBERS.txt
python research/reproducibility/check_pitch_numbers.py  # exits 0 if they agree
```

## Known Data Issues (Now Fixed)

### Issue 1: Cartesian Product in Roll-Date Merges
When merging price data on roll dates, a non-unique join key produced 6,007 duplicate rows.
Fixed in `data/clean/repair.py` using price continuity rules to select the correct row.

### Issue 2: Timestamp Basis (ts_recv vs ts_ref)
Original data used `ts_recv` (API delivery time), which introduced Sunday timestamps and a
305-day-vs-252-day session mismatch. Fixed in `data/clean/curve_to_px.py` by rebuilding
from `ts_ref` (the settlement reference date).

### Issue 3: Price Scale (Cents vs Dollars)
Grains, soy complex, and livestock are quoted in CENTS per bushel/pound, not dollars.
Using face-value quotes inflated notional by 100× for 7 of 17 instruments. Fixed in
`engine/universe.py` via `price_scale=0.01` for affected instruments.

## OOS Window Policy

The most recent 25% of the sample (`OOS_FRACTION = 0.25`) is sealed. To touch it:

```bash
python engine/immediacy.py --run --oos-unlock
```

This flag has been used once, after all specification decisions were finalized. The OOS
Sharpe and the IS Sharpe are both reported in `engine/final_numbers.py`.

## Reproducibility Commitment

Every number in any presentation of this strategy was produced by a single run of a
single script on a single data file. The file, the script, and the run timestamp are
recorded in `FINAL_NUMBERS.txt`. `check_pitch_numbers.py` asserts this contract is kept.
