from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

from statsmodels.tsa.arima.model import ARIMA
N_LAGS = 4
N_TRAIN = 335
N_TEST = 48


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "usdinr_383.csv"

RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    exist_ok=True
)


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
# METRIC FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    ape = (
        np.abs(
            (actual - predicted)
            / actual
        )
        * 100
    )

    mape = np.mean(ape)

    mae = np.mean(
        np.abs(
            actual - predicted
        )
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    return mape, mae, rmse


# ============================================================
# FLANN
# ============================================================

class FLANN:

    def __init__(self, alpha=1.0):

        self.input_scaler = StandardScaler()

        self.feature_scaler = StandardScaler()

        self.model = Ridge(
            alpha=alpha,
            fit_intercept=True
        )

    @staticmethod
    def functional_expansion(X):

        return np.hstack([

            X,

            X ** 2,

            np.sin(np.pi * X),
            np.cos(np.pi * X),

            np.sin(2 * np.pi * X),
            np.cos(2 * np.pi * X),

            np.sin(3 * np.pi * X),
            np.cos(3 * np.pi * X)

        ])

    def fit(self, X, y):

        X_scaled = (
            self.input_scaler
            .fit_transform(X)
        )

        Z = self.functional_expansion(
            X_scaled
        )

        Z_scaled = (
            self.feature_scaler
            .fit_transform(Z)
        )

        self.model.fit(
            Z_scaled,
            y
        )

        return self

    def predict(self, X):

        X_scaled = (
            self.input_scaler
            .transform(X)
        )

        Z = self.functional_expansion(
            X_scaled
        )

        Z_scaled = (
            self.feature_scaler
            .transform(Z)
        )

        return self.model.predict(
            Z_scaled
        )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("USD/INR MODEL COMPARISON")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Total records: {len(df)}"
)


# ============================================================
# PREPARE INPUTS
# ============================================================

X = df[FEATURES].values

y = df["target"].values.astype(float)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

X_train = X[:TRAIN_SAMPLES]

X_test = X[TRAIN_SAMPLES:]

y_train = y[:TRAIN_SAMPLES]

y_test = y[TRAIN_SAMPLES:]


print()
print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# 1. FLANN
# ============================================================

print()
print("-" * 70)
print("1. TRAINING FLANN")
print("-" * 70)

flann = FLANN(
    alpha=1.0
)

flann.fit(
    X_train,
    y_train
)

flann_predictions = flann.predict(
    X_test
)

flann_mape, flann_mae, flann_rmse = (
    calculate_metrics(
        y_test,
        flann_predictions
    )
)

print("FLANN completed.")


# ============================================================
# 2. MLP
# ============================================================

print()
print("-" * 70)
print("2. TRAINING MLP")
print("-" * 70)


# Scale input
x_scaler = StandardScaler()

X_train_scaled = (
    x_scaler.fit_transform(
        X_train
    )
)

X_test_scaled = (
    x_scaler.transform(
        X_test
    )
)


# Scale target
y_scaler = StandardScaler()

y_train_scaled = (
    y_scaler
    .fit_transform(
        y_train.reshape(-1, 1)
    )
    .ravel()
)


mlp = MLPRegressor(

    hidden_layer_sizes=(16, 8),

    activation="tanh",

    solver="lbfgs",

    alpha=0.001,

    max_iter=5000,

    random_state=42
)


mlp.fit(
    X_train_scaled,
    y_train_scaled
)


mlp_predictions_scaled = (
    mlp.predict(
        X_test_scaled
    )
)


mlp_predictions = (
    y_scaler
    .inverse_transform(
        mlp_predictions_scaled.reshape(-1, 1)
    )
    .ravel()
)


mlp_mape, mlp_mae, mlp_rmse = (
    calculate_metrics(
        y_test,
        mlp_predictions
    )
)

print("MLP completed.")


# ============================================================
# 3. ARIMA(1,1,1)
# ============================================================

print()
print("-" * 70)
print("3. TRAINING ARIMA(1,1,1)")
print("-" * 70)


arima = ARIMA(
    y_train,
    order=(1, 1, 1)
)

arima_fitted = arima.fit()


arima_predictions = (
    arima_fitted
    .forecast(
        steps=TEST_SAMPLES
    )
)

arima_predictions = np.asarray(
    arima_predictions,
    dtype=float
)


arima_mape, arima_mae, arima_rmse = (
    calculate_metrics(
        y_test,
        arima_predictions
    )
)

print("ARIMA completed.")


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [
        "FLANN",
        "MLP",
        "ARIMA(1,1,1)"
    ],

    "MAPE": [
        flann_mape,
        mlp_mape,
        arima_mape
    ],

    "MAE": [
        flann_mae,
        mlp_mae,
        arima_mae
    ],

    "RMSE": [
        flann_rmse,
        mlp_rmse,
        arima_rmse
    ]

})


# Sort by MAPE
results = results.sort_values(
    "MAPE"
).reset_index(
    drop=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = (
    RESULTS_DIR
    / "model_metrics.csv"
)

results.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "date":
        df["date"]
        .iloc[TRAIN_SAMPLES:]
        .values,

    "actual":
        y_test,

    "FLANN":
        flann_predictions,

    "MLP":
        mlp_predictions,

    "ARIMA_1_1_1":
        arima_predictions

})


prediction_path = (
    RESULTS_DIR
    / "predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    prediction_df["actual"],
    label="Actual"
)

plt.plot(
    prediction_df["FLANN"],
    label="FLANN"
)

plt.plot(
    prediction_df["MLP"],
    label="MLP"
)

plt.plot(
    prediction_df["ARIMA_1_1_1"],
    label="ARIMA(1,1,1)"
)

plt.title(
    "USD/INR Actual vs Predicted"
)

plt.xlabel(
    "Test Observation"
)

plt.ylabel(
    "USD/INR Exchange Rate"
)

plt.legend()

plt.tight_layout()

actual_plot = (
    RESULTS_DIR
    / "actual_vs_predicted.png"
)

plt.savefig(
    actual_plot,
    dpi=300
)

plt.close()


# ============================================================
# MODEL ERROR GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    results["Model"],
    results["MAPE"]
)

plt.title(
    "Model Comparison - MAPE"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "MAPE (%)"
)

plt.tight_layout()

comparison_plot = (
    RESULTS_DIR
    / "model_comparison.png"
)

plt.savefig(
    comparison_plot,
    dpi=300
)

plt.close()


# ============================================================
# BEST MODEL
# ============================================================

best_model = results.iloc[0]["Model"]

best_mape = results.iloc[0]["MAPE"]


print()
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Best model : {best_model}"
)

print(
    f"Lowest MAPE: {best_mape:.4f}%"
)


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print("=" * 70)
print("RESULT FILES CREATED")
print("=" * 70)

print(
    f"Metrics     : {metrics_path}"
)

print(
    f"Predictions : {prediction_path}"
)

print(
    f"Graph 1     : {actual_plot}"
)

print(
    f"Graph 2     : {comparison_plot}"
)

print()
print("MODEL COMPARISON COMPLETE.")