# ESG Alpha Generation Project (Modular OOP)

This project implements a production-style ESG quant pipeline with interchangeable methods for each stage:

1. Data ingestion and PIT-safe processing
2. Signal construction
3. Alpha forecasting
4. Portfolio construction
5. Backtesting and evaluation

Each stage is a module with strategy classes, selected from one config file (`configs/default.yaml`).

## Project Structure

```text
configs/
  constrained.yaml
  default.yaml
src/esg_alpha/
  config.py
  factory.py
  pipeline.py
  data/
  signal/
  forecast/
  portfolio/
  backtest/
run_alpha_pipeline.py
environment.yml
tests/test_pipeline_smoke.py
requirements.txt
requirements-nlp.txt
```

## Environment Setup (Conda)

### Windows (PowerShell)

```powershell
cd d:\NTU\Intern\temasek
# First-time setup
conda env create -f environment.yml
conda activate temasek

# Existing environment (run this instead of create)
# conda env update -f environment.yml --prune
# conda activate temasek
```

Optional NLP stack (FinBERT signal):

```powershell
pip install -r requirements-nlp.txt
```

### macOS/Linux

```bash
cd /path/to/temasek
# First-time setup
conda env create -f environment.yml
conda activate temasek

# Existing environment (run this instead of create)
# conda env update -f environment.yml --prune
# conda activate temasek
```

Optional NLP stack:

```bash
pip install -r requirements-nlp.txt
```

## Run Pipeline

```powershell
python run_alpha_pipeline.py --config configs/default.yaml

# Optional: constrained mean-variance optimizer profile
python run_alpha_pipeline.py --config configs/constrained.yaml
```

## Swap Methods (Single Config)

Edit `configs/default.yaml` only.

Available options:

- `data.ingestor`: `mock_pit`
- `data.imputer`: `forward_fill_median`, `knn`
- `data.normalizer`: `sector_zscore`, `none`
- `signal.builder`: `materiality_momentum`, `lexicon_nlp`, `finbert_nlp`
- `signal.orthogonalizer`: `ols`, `none`
- `forecast.method`: `rank_to_return`, `ic_scaled_rank`
- `portfolio.constructor`: `constrained_mean_variance`, `quantile_long_short`

Example:

```yaml
signal:
  builder: lexicon_nlp
  orthogonalizer: none
  builder_params:
    smoothing_window: 3
```

## Why This Matches Your 5-Step Refinement

- Look-ahead bias control: PIT panel built from `asof_date` snapshots.
- Missing data: pluggable imputation methods.
- Financial materiality: sector-specific metric weights in `MaterialityMomentumSignalBuilder`.
- Orthogonalization: cross-sectional OLS neutralization against style factors + sector dummies.
- Constraints-aware portfolio construction: gross/net, max weight, turnover, sector neutrality.
- Backtest realism: execution lag, transaction costs, slippage, turnover tracking.
- Evaluation depth: IC, IR, drawdown, and factor attribution.

## Run Tests

```powershell
pytest -q
```
