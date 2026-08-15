from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "usdinr_383.csv"


# ============================================================
# SETTINGS
# ============================================================

TRAIN_SAMPLES = 335
TEST_SAMPLES = 48


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("ARIMA(1,1,1) INDEPENDENT TEST")
print("=" * 60)

print(f"Total records: {len(df)}")


# ============================================================
# TARGET TIME SERIES
# ============================================================

series = df["target"].values.astype(float)

train = series[:TRAIN_SAMPLES]

test = series[TRAIN_SAMPLES:]


print()
print(f"Training samples: {len(train)}")
print(f"Testing samples : {len(test)}")


# ============================================================
# CREATE ARIMA MODEL
# ============================================================

print()
print("Training ARIMA(1,1,1)...")

model = ARIMA(
    train,
    order=(1, 1, 1)
)

fitted_model = model.fit()

print("ARIMA training completed.")


# ============================================================
# FORECAST 48 TEST OBSERVATIONS
# ============================================================

predictions = fitted_model.forecast(
    steps=TEST_SAMPLES
)

predictions = np.asarray(
    predictions,
    dtype=float
)


# ============================================================
# FIRST TEST PREDICTION
# ============================================================

print()
print("=" * 60)
print("FIRST TEST PREDICTION")
print("=" * 60)

print(
    f"Actual USD/INR    : {test[0]:.4f}"
)

print(
    f"Predicted USD/INR : {predictions[0]:.4f}"
)


# ============================================================
# MAPE
# ============================================================

ape = (
    np.abs(
        (test - predictions)
        / test
    )
    * 100
)

mape = np.mean(ape)


# ============================================================
# MAE
# ============================================================

mae = np.mean(
    np.abs(
        test - predictions
    )
)


# ============================================================
# RMSE
# ============================================================

rmse = np.sqrt(
    np.mean(
        (test - predictions) ** 2
    )
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("ARIMA(1,1,1) TEST RESULT")
print("=" * 60)

print(
    f"MAPE : {mape:.4f}%"
)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)