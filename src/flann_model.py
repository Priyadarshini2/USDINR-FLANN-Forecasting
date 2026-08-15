import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class FLANN:

    def __init__(self, alpha=1.0):

        self.alpha = alpha

        # Scale original inputs
        self.input_scaler = StandardScaler()

        # Scale expanded features
        self.feature_scaler = StandardScaler()

        # Ridge output layer
        self.model = Ridge(
            alpha=self.alpha,
            fit_intercept=True
        )

    # ========================================================
    # FUNCTIONAL EXPANSION
    # ========================================================

    @staticmethod
    def functional_expansion(X):

        return np.hstack([

            # Original inputs
            X,

            # Polynomial term
            X ** 2,

            # Harmonic terms
            np.sin(np.pi * X),
            np.cos(np.pi * X),

            np.sin(2 * np.pi * X),
            np.cos(2 * np.pi * X),

            np.sin(3 * np.pi * X),
            np.cos(3 * np.pi * X)

        ])

    # ========================================================
    # TRAIN
    # ========================================================

    def fit(self, X, y):

        # Step 1: scale original inputs
        X_scaled = self.input_scaler.fit_transform(X)

        # Step 2: functional expansion
        Z = self.functional_expansion(X_scaled)

        # Step 3: scale expanded features
        Z_scaled = self.feature_scaler.fit_transform(Z)

        # Step 4: learn output weights
        self.model.fit(Z_scaled, y)

        return self

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, X):

        # Use training scaler
        X_scaled = self.input_scaler.transform(X)

        # Functional expansion
        Z = self.functional_expansion(X_scaled)

        # Use training feature scaler
        Z_scaled = self.feature_scaler.transform(Z)

        # Prediction
        return self.model.predict(Z_scaled)