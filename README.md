# Macro-Driven Forex Bias System

Bachelor's-thesis implementation of a directional bias model for EUR/USD,
built on macroeconomic event data. Ingests indicators from FRED and FX
prices via `yfinance`, harmonises them into an event matrix, fits a
logistic regression with explicit standard errors and p-values per
coefficient, and emits long/short signals over four forecast horizons
(1 / 5 / 20 / 250 trading days).

The codebase is structured as three independent modules — Input,
Evaluation, Output — that can be run in isolation or as a complete
workflow via `main.py`.

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/pandas-2.x-150458?logo=pandas" alt="pandas" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License" />
</p>

> Academic research code. Models use simplified assumptions on a single
> currency pair (EUR/USD) over a fixed sample period. Nothing in this
> repository is investment advice.

---

## What this project does

The thesis question: can publicly available macroeconomic indicator
releases be combined into a meaningful directional bias for EUR/USD on
short, medium, and long horizons?

The implementation answers it as a small system:

- **Input** — pulls 12+ FRED indicators (interest-rate spreads, inflation,
  unemployment, GDP, leading indicators) and EUR/USD daily closes,
  harmonises heterogeneous release frequencies into a single event matrix
  indexed by release date.
- **Evaluation** — engineers standardised features, runs logistic
  regression per horizon, reports coefficient standard errors and p-values
  alongside AUC, accuracy, precision, recall, and F1.
- **Output** — generates time-series plots, ROC curves, confusion
  matrices, feature-importance plots split between training and test,
  and an interactive HTML dashboard summarising every horizon.
- **Workflow** — `main.py` orchestrates the modules with subcommands
  (`complete`, `training`, `prediction`, `status`).

---

## Modules

### `input_module.py`
- Pulls FRED indicators with the official client; pulls EUR/USD via
  `yfinance`
- Outlier handling, missing-value interpolation, deduplication
- Frequency harmonisation (daily / monthly / quarterly → release-date
  event matrix)
- Persists an event matrix CSV that downstream modules consume

### `evaluation_module.py`
- Standardisation (z-transform) of every indicator
- Lag and rolling-window feature toggles (configurable in `config.py`)
- Logistic regression with explicit standard-error and p-value output
- Classification into LONG / SHORT signals with intensity buckets

### `output_module.py`
- Time-series plots of indicators vs. realised returns
- ROC curves, confusion matrices per horizon
- Feature importance plots, separately for training and test windows
- Interactive HTML dashboards (Plotly)
- Text reports per run

### `main.py`
- One CLI entry point with four modes
- Loads / saves trained models with deterministic file naming

---

## Methodology

| Aspect | Choice |
| --- | --- |
| Target | Binary classification of next-window directional move (LONG / SHORT) |
| Threshold | 0.25% close-to-close move |
| Decision boundary | 0.5 logistic probability |
| Indicators | FRED: interest-rate spreads, CPI, unemployment, GDP growth, PMI, leading indicators |
| FX data | EUR/USD daily close, `yfinance` |
| Training window | 2003-01-01 → 2019-12-31 |
| Test window | 2020-01-01 → 2025-12-31 |
| Horizons | 1, 5, 20, 250 trading days |
| Validation | Stratified split, out-of-sample test, structural-break analysis |
| Metrics | AUC, accuracy, precision, recall, F1, statistical tests across horizons |

The full methodology — sample selection, feature engineering rationale,
robustness tests — is documented in the thesis itself.

---

## Local setup

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
echo "FRED_API_KEY=your_key_here" > .env
```

## Usage

```bash
# Full workflow (ingest → train → evaluate → report)
python main.py --mode complete --start_date 2003-01-01 --end_date 2025-12-31

# Training only
python main.py --mode training --start_date 2003-01-01 --end_date 2019-12-31

# Prediction from a saved model
python main.py --mode prediction \
  --load_data event_matrix_20241201_120000.csv \
  --model_prefix models_20241201_120000

# System status (data freshness, last model trained)
python main.py --mode status
```

### Python API

```python
from main import ForexPredictionSystem

system = ForexPredictionSystem()

results = system.run_complete_workflow(
    start_date='2003-01-01',
    end_date='2025-12-31',
)

print(results['metrics']['horizon_5d']['auc'])
```

---

## Output

```
bachelorarbeit/
├── data/      # event matrix CSVs
├── models/    # one .pkl per horizon
├── results/   # CSV reports with metrics + coefficients
├── plots/     # PNG visualisations
└── logs/      # run logs
```

Per run, the system writes:

- **Event matrix** (CSV) — harmonised feature panel
- **Model artefacts** (pickle) — one per horizon
- **Result reports** (CSV) — metrics + coefficient table with p-values
- **Plots** (PNG) — time series, ROC, confusion, feature importance
- **HTML dashboard** — interactive per-horizon summary
- **Text report** — narrative summary of the run

Additional documentation:

- [`MODULE_DOCUMENTATION.md`](MODULE_DOCUMENTATION.md) — function-level reference
- [`TRAINING_TEST_COMPARISON_IMPLEMENTATION.md`](TRAINING_TEST_COMPARISON_IMPLEMENTATION.md) — out-of-sample comparison details
- [`VARIABLE_SUMMARY_IMPLEMENTATION.md`](VARIABLE_SUMMARY_IMPLEMENTATION.md) — variable-summary methodology

---

## Configuration

All system parameters live in `config.py`:

| Parameter | Default | Description |
| --- | --- | --- |
| `FRED_API_KEY` | from env | FRED Data API key (free) |
| `TRAINING_START` / `END` | 2003-01-01 / 2019-12-31 | Training window |
| `TEST_START` / `END` | 2020-01-01 / 2025-12-31 | Out-of-sample window |
| `FORECAST_HORIZONS` | `[1, 5, 20, 250]` | Trading-day horizons |
| `MOVEMENT_THRESHOLD` | `0.0025` | Min directional move for label |
| `CLASSIFICATION_THRESHOLD` | `0.5` | Logistic decision boundary |
| `FRED_INDICATORS` | (see file) | 12+ macro series IDs |

---

## Limitations

- **Single pair (EUR/USD).** The harmonisation pipeline is currency-pair-agnostic; the calibration is not.
- **Logistic regression only.** The thesis intentionally uses a transparent linear model; non-linear models (RF, XGBoost, neural nets) are a natural extension.
- **No live-trading wiring.** The system emits signals; it does not place orders.
- **FRED rate limits apply.** Heavy parameter sweeps may need a cache layer.

---

## Future work

- Additional FX pairs and cross-currency tests
- Tree-based and ensemble models alongside logistic regression
- News-sentiment indicators as additional features
- Real-time signal generation hooked into a paper-trading ledger
- Bayesian shrinkage / regularisation of coefficients

---

## License

MIT — see [`LICENSE`](LICENSE). Bachelor's thesis project, 2025.
