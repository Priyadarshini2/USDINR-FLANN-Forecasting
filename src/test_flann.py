from pathlib import Path
import pandas as pd
import numpy as np

from flann_model import FLANN


# Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "usdinr_383.csv"


# Experiment settings
TRAIN_SAMPLES = 335
TEST_SAMPLES = 48
LAGS = 4


# ------------------------------------------------------------
# Load prepared dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("FLANN INDEPENDENT TEST")
print("=" * 60)

print(f"Total records: {len(df)}")


# ------------------------------------------------------------
# Separate inputs and target
# ------------------------------------------------------------

feature_columns = [
    "lag_4",
    "lag_3",
    "lag_2",
    "lag_1"
]

X = df[feature_columns].values

y = df["target"].values


# ------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------

X_train = X[:TRAIN_SAMPLES]

y_train = y[:TRAIN_SAMPLES]


X_test = X[TRAIN_SAMPLES:]

y_test = y[TRAIN_SAMPLES:]


print()
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ------------------------------------------------------------
# Create FLANN
# ------------------------------------------------------------

model = FLANN(
    alpha=1.0
)


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print()
print("Training FLANN...")

model.fit(
    X_train,
    y_train
)


print("FLANN training completed.")


# ------------------------------------------------------------
# Predict test data
# ------------------------------------------------------------

predictions = model.predict(
    X_test
)


# ------------------------------------------------------------
# Show first prediction
# ------------------------------------------------------------

print()
print("=" * 60)
print("FIRST TEST PREDICTION")
print("=" * 60)

print("Input:")
print(X_test[0])

print()

print(
    f"Actual USD/INR    : {y_test[0]:.4f}"
)

print(
    f"Predicted USD/INR : {predictions[0]:.4f}"
)


# ------------------------------------------------------------
# Calculate APE
# ------------------------------------------------------------

ape = (
    abs(
        (y_test - predictions)
        / y_test
    )
    * 100
)


mape = np.mean(ape)


print()
print("=" * 60)
print("FLANN TEST RESULT")
print("=" * 60)

print(
    f"MAPE: {mape:.4f}%"
)