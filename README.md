# USD/INR Exchange Rate Forecasting using FLANN

### FLANN • MLP • ARIMA(1,1,1) • MATLAB • Python • Flask

A comparative time-series forecasting project for **1-day-ahead USD/INR exchange-rate prediction** using real historical exchange-rate data.

The project investigates whether a **Functional Link Artificial Neural Network (FLANN)** can effectively model nonlinear exchange-rate relationships and compares its performance with a **Multilayer Perceptron (MLP)** and the classical **ARIMA(1,1,1)** time-series model.

The original research implementation was developed in **MATLAB**. A reproducible **Python implementation** was subsequently developed with automated data preparation, model training, evaluation, visualization, and a **Flask-based web interface** for prediction.

## Project Objective

The objective is to predict the **next USD/INR exchange-rate observation** using recent historical exchange-rate values.

The study compares:
- **FLANN** — Functional Link Artificial Neural Network
- **MLP** — Multilayer Perceptron
- **ARIMA(1,1,1)** — Classical statistical time-series model

The models are evaluated using MAPE, MAE and RMSE on an unseen test dataset.

## Problem Statement

Exchange rates are influenced by complex economic and market factors and may exhibit nonlinear patterns that are difficult to capture using traditional linear forecasting approaches.

> **Can a Functional Link Artificial Neural Network effectively predict the next USD/INR exchange-rate observation using recent historical observations, and how does its performance compare with MLP and ARIMA(1,1,1)?**

## Dataset

The project uses **real historical USD/INR exchange-rate observations**, not synthetic data. The historical series is based on the U.S. Federal Reserve's **H.10 Foreign Exchange Rates** data.

| Description | Value |
|---|---:|
| Raw observations selected | 387 |
| Supervised samples | **383** |
| Training samples | **335** |
| Testing samples | **48** |
| Input lag window | **4 observations** |
| Forecast horizon | **1 day ahead** |

The four previous observations are used to predict the next observation:

```text
t-4   t-3   t-2   t-1
 │     │     │     │
 └─────┴─────┴─────┴──────► Model
                              │
                              ▼
                         Prediction t
```

# Methodology

## 1. Data Collection

Historical USD/INR data is downloaded programmatically.

The raw dataset is preserved in:

```text
data/usdinr_raw.csv
```

## 2. Data Preparation

The time series is transformed into a supervised-learning format using a **4-day lag window**.

```text
lag_4   lag_3   lag_2   lag_1   target
86.42   86.20   86.34   86.52   86.61
86.20   86.34   86.52   86.61   86.55
86.34   86.52   86.61   86.55   86.54
```

Therefore:

```text
Input  = [t-4, t-3, t-2, t-1]
Target = t
```

This produces **383 supervised samples**.

## 3. Train/Test Split

The observations are divided chronologically:

```text
335 observations → Training
48 observations  → Testing
```

No random shuffling is used because this is a **time-series forecasting problem**.

# FLANN Model

## Functional Link Artificial Neural Network

FLANN is a single-layer neural-network approach that expands the original input features using nonlinear functional transformations.

The project uses **trigonometric functional expansion** to create additional nonlinear representations.

```text
Original inputs
      │
      ▼
Functional expansion
      │
      ├── Original terms
      ├── Sinusoidal terms
      └── Cosine terms
      │
      ▼
Expanded feature vector
      │
      ▼
FLANN output
      │
      ▼
Predicted USD/INR
```

This allows the network to model nonlinear relationships without requiring a deep neural-network architecture.

# MLP Model

A Multilayer Perceptron is used as a neural-network benchmark.

The project uses a hidden-layer architecture based on:

```text
Input layer
     │
     ▼
8 neurons
     │
     ▼
4 neurons
     │
     ▼
Output
```

# ARIMA Model

The classical statistical model used for comparison is:

```text
ARIMA(1,1,1)
```

where:

```text
p = 1
d = 1
q = 1
```

ARIMA provides a traditional time-series forecasting baseline.

# Data Scaling

For the neural-network models, input and target values are scaled using **min-max scaling to [-1, 1]**. The scaling parameters are learned from the training data to avoid using information from the test set during preprocessing.

# Evaluation

The models are evaluated on **48 unseen test observations**.

### MAPE

```text
MAPE = mean(|Actual - Predicted| / |Actual|) × 100
```

### MAE

```text
MAE = mean(|Actual - Predicted|)
```

### RMSE

```text
RMSE = sqrt(mean((Actual - Predicted)²))
```

Lower values indicate lower forecasting error.

# Final Results

| Model | MAPE (%) | MAE | RMSE |
|---|---:|---:|---:|
| **FLANN** | **0.2899** | **0.2765** | **0.3362** |
| MLP | 0.3015 | 0.2878 | 0.3688 |
| ARIMA(1,1,1) | 0.6501 | 0.6226 | 0.7626 |

## Best Model

**FLANN**

FLANN achieved the lowest MAPE:

```text
0.2899%
```

It also achieved the lowest MAE and RMSE among the three models in this experiment.

### Key Finding

FLANN outperformed the MLP and ARIMA(1,1,1) models on this particular 48-observation test window.

This result is specific to the dataset, time period and experimental configuration used in the study. It should not be interpreted as evidence that FLANN will always outperform other forecasting approaches.

# Example Prediction

For the four recent observations:

```text
95.68
95.69
95.69
95.00
```

the models produced:

| Model | Predicted USD/INR |
|---|---:|
| **FLANN** | **95.2490** |
| MLP | 94.8686 |
| ARIMA(1,1,1) | 94.9902 |

The corresponding actual observation used during the test experiment was:

```text
94.9900
```

# Visualization

The project generates:

- Actual vs Predicted USD/INR graph
- Model MAPE comparison graph

Generated result files include:

```text
results/
├── actual_vs_predicted.png
├── forecast_comparison.png
├── model_comparison.png
├── model_metrics.csv
├── model_comparison.csv
├── model_comparisons.csv
├── predictions.csv
└── test_predictions.csv
```

# Flask Web Application

A Flask-based web interface was developed to make the forecasting system interactive.

The application allows the user to enter:

```text
t-4
t-3
t-2
t-1
```

and generates predictions from:

```text
FLANN
MLP
ARIMA(1,1,1)
```

The interface displays model predictions, model performance, MAPE, MAE, RMSE, graphs and the best-performing model.

```text
flask_app/
├── app.py
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── graphs/
        ├── actual_vs_predicted.png
        └── model_comparison.png
```

# MATLAB Implementation

The original research implementation was developed in MATLAB.

```text
matlab/
└── run_all.m
```

The MATLAB implementation represents the original research workflow, while the Python implementation provides a reproducible software pipeline.

# Project Structure

```text
USDINR-FLANN-Forecasting/
│
├── data/
│   ├── usdinr_raw.csv
│   └── usdinr_383.csv
│
├── flask_app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── graphs/
│
├── matlab/
│   └── run_all.m
│
├── results/
│   ├── actual_vs_predicted.png
│   ├── forecast_comparison.png
│   ├── model_comparison.png
│   ├── model_metrics.csv
│   ├── model_comparison.csv
│   ├── model_comparisons.csv
│   ├── predictions.csv
│   └── test_predictions.csv
│
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── flann_model.py
│   ├── model_comparison.py
│   ├── test_flann.py
│   ├── test_mlp.py
│   ├── test_arima.py
│   └── run_project.py
│
├── .gitignore
├── README.md
├── INTERVIEW_NOTES.md
├── requirements.txt
└── run_project.bat
```

# How to Run

## 1. Clone the repository

```bash
git clone https://github.com/Priyadarshini2/USDINR-FLANN-Forecasting.git
cd USDINR-FLANN-Forecasting
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```cmd
venv\Scriptsctivate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the complete forecasting pipeline

```bash
python src/run_project.py
```

The pipeline performs:

```text
Data
  ↓
Preparation
  ↓
FLANN
  ↓
MLP
  ↓
ARIMA
  ↓
Evaluation
  ↓
Results + Graphs
```

A Windows batch runner is also available:

```text
run_project.bat
```

# Running the Flask Application

From the project root:

```bash
python flask_app/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

# Technologies Used

### Programming
- Python
- MATLAB

### Machine Learning
- NumPy
- Scikit-learn
- FLANN
- MLP

### Time-Series Forecasting
- Statsmodels
- ARIMA

### Data Processing
- Pandas

### Visualization
- Matplotlib

### Web Development
- Flask
- HTML
- CSS

### Version Control
- Git
- GitHub

# Key Contributions

This project demonstrates:

- Real-world time-series data acquisition
- Time-series preprocessing
- Lag-feature engineering
- Functional Link Artificial Neural Network implementation
- Neural-network forecasting
- Statistical time-series forecasting
- Model comparison
- Quantitative evaluation using MAPE, MAE and RMSE
- MATLAB-to-Python implementation
- Flask-based interactive interface
- Reproducible ML pipeline
- Git/GitHub project management

# Limitations

The project uses a relatively small experimental window of **383 supervised observations** and a **48-observation test set**.

USD/INR exchange rates are influenced by many external variables that are not included in the current feature set, such as:

- Interest rates
- Inflation
- Monetary policy
- Crude oil prices
- Foreign capital flows
- Global economic conditions
- Geopolitical events

The current model uses only historical USD/INR observations.

Therefore, the system should be considered a **research and educational forecasting system**, not a financial trading or investment recommendation system.

# Future Improvements

Possible extensions include:

- Longer historical training periods
- Additional economic indicators
- Volatility features
- Technical indicators
- LSTM/GRU comparison
- XGBoost comparison
- Transformer-based forecasting
- Hyperparameter optimization
- Multi-step forecasting
- Automated daily data updates
- Model retraining pipeline
- Docker deployment
- Cloud deployment
- Real-time exchange-rate API integration

# Research Conclusion

The experimental results show that **FLANN achieved the best performance among the three models in the final 48-observation test set**, with a MAPE of **0.2899%**.

The comparison demonstrates that a functional-expansion-based neural architecture can achieve competitive forecasting performance without requiring a deep neural-network structure.

The results are specific to the dataset, time period and experimental configuration used in this study.

---

## Author

**Priyadarshini Behera**

**Project:** Dollar Exchange Rate Prediction Using Functional Link Artificial Neural Network (FLANN)

**Models:** FLANN • MLP • ARIMA(1,1,1)

**Implementation:** MATLAB • Python • Flask
