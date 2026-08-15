# Interview Notes — Dollar Exchange Rate Prediction using FLANN

Talking points and anticipated questions for discussing this project in an
interview. Read this alongside `README.md`.

---

## 1. "Walk me through the project."

"I built a system to predict the next day's USD/INR exchange rate,
comparing three models: FLANN (Functional Link ANN), a standard MLP, and
a classical ARIMA(1,1,1) statistical model. I used 383 real daily USD/INR
observations — sourced from the Federal Reserve's H.10 release — split
335 for training and 48 for out-of-sample testing. Each model predicts a
given day from the previous 4 days' rates. I evaluated all three with the
same walk-forward, one-step-ahead protocol so the comparison is fair, and
scored them with MAPE, RMSE, and MAE."

## 2. "What is FLANN and why use it instead of a normal neural network?"

A standard MLP captures nonlinearity by stacking hidden layers with
nonlinear activations, learned jointly via backpropagation — more
parameters, more compute, more risk of overfitting on small datasets.

FLANN instead **expands the input space** using a *fixed* set of nonlinear
basis functions (here: trigonometric — sin/cos harmonics of each lag),
then fits a **single linear layer** of weights on top of that expanded
representation. All the nonlinearity comes from the fixed expansion, not
from learned hidden layers, so:

- Far fewer trainable parameters (here: `4 inputs × (1 + 2×3 harmonics) =
  28` weights, vs. an MLP with two hidden layers with many more).
- Training reduces to a simple (regularized) linear regression / gradient
  descent problem — fast, stable, less prone to getting stuck in local
  minima.
- Works well when the true relationship has smooth periodic/oscillatory
  structure the basis functions can represent, and is a good choice on
  small datasets where a full MLP might overfit.

## 3. "Your results show ARIMA winning, but the resume implies FLANN was
   best — how do you explain that?"

This is the most important question to be ready for, and the honest
answer is a strength, not a weakness, if you explain it well:

- **The specific test window matters a lot.** The 48-day test period used
  here (mid-2026) happened to have a fairly strong, fairly smooth
  dollar-appreciation trend. A linear ARIMA model, especially one
  re-estimated at every step with the true recent history, is very good
  at tracking smooth trends/momentum — that's exactly what it's designed
  for. There's a well-known result in FX forecasting (going back to
  Meese & Rogoff, 1983) that short-horizon FX moves are extremely close
  to a random walk, which is close to ARIMA's home turf.
- **FLANN and ANN-style models earn their advantage when the underlying
  process has genuine nonlinear structure** — regime shifts, volatility
  clustering, threshold effects — that a linear model structurally cannot
  capture. In a smoother, more trend-dominated window, that advantage
  shrinks or disappears.
- **FLANN still beat the deeper MLP** (0.305% vs 0.330% MAPE) in this run,
  which supports the narrower, more defensible claim: *for a fixed
  "neural network" budget, the simpler functional-expansion architecture
  generalized better than the deeper network* — probably because with
  only 335 training samples, the MLP's larger parameter count is more
  prone to overfitting.
- **Model choice should depend on the regime.** In practice you'd want to
  either (a) test across multiple non-overlapping historical windows and
  report the distribution of outcomes rather than one run, or (b) build
  an ensemble / regime-detector that leans on ARIMA when the series is
  trending smoothly and leans on the nonlinear models when volatility or
  curvature is high.

Be upfront that this is real data and a real result, not a cherry-picked
outcome — that honesty plays well in interviews.

## 4. "How did you avoid overfitting / data leakage?"

- All scalers (`MinMaxScaler`) are fit **only on the training set**, then
  applied to test data.
- Hyperparameters for FLANN and MLP were chosen via a grid search on a
  **validation split carved out of the training data only** (the last 48
  training days), never touching the true test set, before evaluating
  the final chosen configuration once on test.
- ARIMA is refit at each walk-forward step using only data available up
  to that point in time (no future leakage).

## 5. "Why ARIMA(1,1,1) specifically?"

`(p,d,q) = (1,1,1)`: one autoregressive term, first-order differencing
(appropriate since exchange rate levels are non-stationary / roughly
integrated of order 1 — a standard unit-root finding for FX series), and
one moving-average term. It's a minimal, standard baseline specification
rather than a fully tuned/optimized ARIMA (which would typically involve
an AIC/BIC-based order search, e.g., via `auto_arima`) — worth mentioning
if asked whether it could be improved.

## 6. "What would you do differently / how would you extend this?"

- Try `auto_arima`-style order search instead of a fixed (1,1,1).
- Add exogenous features (interest-rate differentials, oil prices, DXY
  index, FII/FDI flow data) — FLANN and MLP can take these as extra
  inputs trivially; this would move toward ARIMAX/SARIMAX for the
  statistical side.
- Add a GARCH-family volatility model, since FX returns are well known to
  show volatility clustering that none of these three models directly
  captures.
- Backtest across multiple non-overlapping historical windows (rolling
  origin) instead of a single 48-day test period, and report a
  distribution of MAPE rather than a single number, since forecasting
  accuracy for FX is highly regime-dependent (see Q3).
- Try a wider FLANN functional-expansion basis (Chebyshev polynomials,
  or a mix of polynomial + trigonometric terms) and formally compare via
  cross-validation.

## 7. "How is 'accuracy' defined here — is 0.3% MAPE good?"

For daily FX prediction, MAPE in the 0.2–0.5% range is broadly
comparable to a naive "tomorrow = today" persistence forecast, because
day-to-day FX moves are typically well under 1%. This is worth saying
explicitly if asked — it's honest framing, and a good interviewer will
respect knowing the naive baseline. Comparing against a simple
persistence/random-walk model is a natural next addition if extending
the project.

## 8. Quick technical facts to have ready

- Dataset: 383 usable records after 4-day lagging (335 train / 48 test),
  real Federal Reserve H.10 series via the `datasets/exchange-rates`
  GitHub mirror.
- FLANN basis: sin/cos trigonometric expansion, 3 harmonics per lag → 28
  expanded features from 4 raw inputs.
- FLANN training: gradient descent, learning rate 0.12, 4000 epochs, L2
  regularization 5e-5.
- MLP: 2 hidden layers (8, 4 units), tanh activation, Adam optimizer.
- Evaluation: one-step-ahead walk-forward (all models see true recent
  history at each test step; ARIMA is refit every step).
- Metrics: MAPE, RMSE, MAE, max single-day APE.
