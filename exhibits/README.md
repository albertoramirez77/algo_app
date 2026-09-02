# exhibits/ — Publication-Quality Figures

All exhibit generators write both PDF and PNG output.

## Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `analysis.py` | `ANALYSIS.pdf/png` | Six-panel bootstrap analysis: cumulative returns, rolling 36-month Sharpe, Sharpe distribution, Sortino distribution, maximum drawdown distribution, R-squared distribution. Uses 12-month block bootstrap (preserves loss clustering). |
| `exhibits.py` | `EXHIBITS.pdf/png` | Six-panel exhibit page: hedge quality by proximity, equity curve with drawdown, placebo distribution, parameter surface, trade statistics, bootstrap drawdown distribution. |


## Usage

```bash
python exhibits/analysis.py   --prices ../px_clean.parquet
```
