from pathlib import Path

import numpy as np
import pandas as pd

from flask import Flask, render_template, request

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "usdinr_383.csv"

RESULTS_PATH = ROOT / "results" / "model_metrics.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "lag_4",
    "lag_3",
    "lag_2",
    "lag_1"
]

X = df[FEATURES].values

y = df["target"].values.astype(float)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

TRAIN_SAMPLES = 335

X_train = X[:TRAIN_SAMPLES]

y_train = y[:TRAIN_SAMPLES]


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
# TRAIN FLANN
# ============================================================

flann = FLANN(
    alpha=1.0
)

flann.fit(
    X_train,
    y_train
)


# ============================================================
# TRAIN MLP
# ============================================================

x_scaler = StandardScaler()

X_train_scaled = (
    x_scaler.fit_transform(X_train)
)


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


# ============================================================
# TRAIN ARIMA
# ============================================================

arima_model = ARIMA(
    y_train,
    order=(1, 1, 1)
)

arima_fitted = arima_model.fit()


# ============================================================
# LOAD MODEL METRICS
# ============================================================

if RESULTS_PATH.exists():

    metrics_df = pd.read_csv(
        RESULTS_PATH
    )

else:

    metrics_df = pd.DataFrame()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    mlp_prediction = None

    arima_prediction = None

    error = None

    input_values = [
        "",
        "",
        "",
        ""
    ]

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # Read four inputs
            # ------------------------------------------------

            input_values = [

                request.form["lag_4"],
                request.form["lag_3"],
                request.form["lag_2"],
                request.form["lag_1"]

            ]

            values = np.array([

                float(input_values[0]),
                float(input_values[1]),
                float(input_values[2]),
                float(input_values[3])

            ]).reshape(
                1,
                -1
            )


            # ------------------------------------------------
            # FLANN prediction
            # ------------------------------------------------

            prediction = float(
                flann.predict(values)[0]
            )


            # ------------------------------------------------
            # MLP prediction
            # ------------------------------------------------

            values_scaled = (
                x_scaler.transform(values)
            )

            mlp_scaled_prediction = (
                mlp.predict(
                    values_scaled
                )
            )

            mlp_prediction = float(
                y_scaler
                .inverse_transform(
                    mlp_scaled_prediction.reshape(
                        -1,
                        1
                    )
                )[0][0]
            )


            # ------------------------------------------------
            # ARIMA next forecast
            # ------------------------------------------------

            arima_forecast = (
                arima_fitted
                .forecast(
                    steps=1
                )
            )

            arima_prediction = float(
                arima_forecast[0]
            )


        except (ValueError, KeyError):

            error = (
                "Please enter four valid "
                "numeric USD/INR values."
            )


    # --------------------------------------------------------
    # Convert metrics to records for HTML
    # --------------------------------------------------------

    if not metrics_df.empty:

        metrics = (
            metrics_df
            .round(4)
            .to_dict(
                orient="records"
            )
        )

    else:

        metrics = []


    return render_template(

        "index.html",

        prediction=prediction,

        mlp_prediction=mlp_prediction,

        arima_prediction=arima_prediction,

        error=error,

        input_values=input_values,

        metrics=metrics

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("USD/INR FLANN FORECASTING WEB APPLICATION")
    print("=" * 60)

    print(
        "Open your browser at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    app.run(
        debug=True
    )