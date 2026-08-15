from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "usdinr_383.csv"


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TRAIN_SAMPLES = 335
TEST_SAMPLES = 48

FEATURES = [
    "lag_4",
    "lag_3",
    "lag_2",
    "lag_1"
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("MLP INDEPENDENT TEST")
print("=" * 60)

print(f"Total records: {len(df)}")


# ============================================================
# INPUTS AND TARGET
# ============================================================

X = df[FEATURES].values

y = df["target"].values


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train = X[:TRAIN_SAMPLES]
y_train = y[:TRAIN_SAMPLES]

X_test = X[TRAIN_SAMPLES:]
y_test = y[TRAIN_SAMPLES:]


print()
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# SCALE INPUT FEATURES
# ============================================================

x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)

X_test_scaled = x_scaler.transform(X_test)


# ============================================================
# SCALE TARGET
# ============================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
).ravel()


# ============================================================
# CREATE MLP
# ============================================================

model = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation="tanh",
    solver="lbfgs",
    alpha=0.001,
    max_iter=5000,
    random_state=42
)


# ============================================================
# TRAIN
# ============================================================

print()
print("Training MLP...")

model.fit(
    X_train_scaled,
    y_train_scaled
)

print("MLP training completed.")


# ============================================================
# PREDICT
# ============================================================

predicted_scaled = model.predict(
    X_test_scaled
)


# Convert predictions back to USD/INR
predictions = y_scaler.inverse_transform(
    predicted_scaled.reshape(-1, 1)
).ravel()


# ============================================================
# FIRST TEST PREDICTION
# ============================================================

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


# ============================================================
# MAPE
# ============================================================

ape = (
    np.abs(
        (y_test - predictions)
        / y_test
    )
    * 100
)

mape = np.mean(ape)


# ============================================================
# MAE
# ============================================================

mae = np.mean(
    np.abs(
        y_test - predictions
    )
)


# ============================================================
# RMSE
# ============================================================

rmse = np.sqrt(
    np.mean(
        (y_test - predictions) ** 2
    )
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("MLP TEST RESULT")
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