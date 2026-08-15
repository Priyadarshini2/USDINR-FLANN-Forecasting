# USD/INR Exchange Rate Forecasting using FLANN

### FLANN • MLP • ARIMA(1,1,1)

A comparative time-series forecasting project for 1-day-ahead
USD/INR exchange-rate prediction using real historical exchange-rate data.

The project evaluates three approaches:

- Functional Link Artificial Neural Network (FLANN)
- Multilayer Perceptron (MLP)
- ARIMA(1,1,1)

The original research work was developed in MATLAB. A reproducible Python
implementation was subsequently developed using real USD/INR data, with
automated training, evaluation and visualization.

## Data source (real, not synthetic)

Historical USD/INR rates are pulled from the [`datasets/exchange-rates`](https://github.com/datasets/exchange-rates)
GitHub data package, which republishes the U.S. Federal Reserve's
[H.10 Foreign Exchange Rates](https://www.federalreserve.gov/releases/h10/)
release (Indian Rupees per U.S. Dollar). This is a standard reference
series with data back to 1973, updated on U.S. business days.

The project uses the most recent **383 valid daily observations** at the
time of download, split **335 train / 48 test** — a period that, at the
time this was built, ran from late January 2025 to early August 2026 and
covered a fairly strong dollar-appreciation trend (~84 → ~96 INR/USD).

## Project structure

```
USDINR_FLANN_REALDATA/
├── data/
│   ├── usdinr_raw.csv       # full downloaded history (real, all dates)
│   └── usdinr_383.csv       # 387 most recent raw points -> 383 usable samples
├── src/
│   ├── download_data.py     # pulls real data from GitHub/FRED
│   ├── prepare_data.py      # cleans + extracts the 383-record window
│   ├── flann_model.py       # FLANN implemented from scratch (NumPy)
│   ├── model_comparison.py  # MLP, ARIMA, metrics, shared utilities
│   └── run_project.py       # orchestrates the full pipeline end-to-end
├── matlab/
│   └── run_all.m            # equivalent MATLAB implementation
├── results/
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   └── forecast_comparison.png
├── requirements.txt
├── run_project.bat          # Windows one-click runner
├── README.md
└── INTERVIEW_NOTES.md       # talking points / anticipated interview Q&A
```

## How to run

```bash
pip install -r requirements.txt

python src/download_data.py     # fetch real USD/INR history
python src/prepare_data.py      # extract the 383-record modeling window
python src/run_project.py       # train FLANN, MLP, ARIMA; evaluate; save results
```

Or, on Windows, just double-click `run_project.bat`.

`run_project.py` will auto-download/prepare data if `data/usdinr_383.csv`
is missing, so `python src/run_project.py` alone is also enough for a
fresh checkout. Pass `--download` to force a fresh pull of the latest data.

### MATLAB

```matlab
cd matlab
run_all
```

(Requires the data/ files to already exist — run the Python data steps
first if starting from a clean checkout.)

## Methodology

- **Features**: each model predicts day *t* from the exchange rate on
  days *t-1* through *t-4* (a 4-day lag window).
- **Scaling**: inputs/targets are min-max scaled to [-1, 1], fit on the
  training set only (no leakage).
- **Evaluation protocol**: one-step-ahead walk-forward. Every model
  predicts each test day using the *true* values of the preceding 4 days
  (ARIMA is re-fit at each of the 48 steps as new true observations
  arrive — the standard way ARIMA is evaluated for short-horizon
  day-ahead forecasting).
- **Hyperparameters** (FLANN: 3 trigonometric expansions, lr=0.12, 4000
  epochs; MLP: hidden layers (8,4), tanh, lr=0.01) were selected via a
  grid search on a validation split carved out of the *training* data
  only — the test set was never used for model selection.

## Results

The final real-data experiment was evaluated on 48 unseen test observations.

| Model | MAPE (%) | RMSE | MAE |
|---|---:|---:|---:|
| **ARIMA(1,1,1)** | **0.2693** | 0.3399 | 0.2570 |
| FLANN | 0.3046 | 0.3366 | 0.2908 |
| MLP | 0.3302 | 0.3799 | 0.3155 |

### Key Finding

ARIMA(1,1,1) achieved the lowest MAPE on this particular real-data
test window.

FLANN outperformed MLP, indicating that functional nonlinear expansion
can provide competitive performance without using a deeper neural
network.

The result should not be interpreted as evidence that ARIMA or FLANN
will always outperform the other models. Performance depends on the
dataset, forecasting horizon, market regime and evaluation protocol.

## Reproducing with fresh/different data

Re-run `python src/download_data.py --download` (or just re-run
`download_data.py`) at any time to pull the latest published rates, then
`prepare_data.py` to re-slice the most recent 383-record window, then
`run_project.py` to regenerate all results.
