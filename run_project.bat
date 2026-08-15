@echo off
REM run_project.bat
REM One-click runner for the USD/INR FLANN prediction project on Windows.

echo ============================================
echo  Dollar Exchange Rate Prediction using FLANN
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found on PATH. Install Python 3.9+ and try again.
    pause
    exit /b 1
)

echo [1/4] Installing dependencies from requirements.txt ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [2/4] Downloading real USD/INR historical data ...
python src\download_data.py
if errorlevel 1 (
    echo ERROR: Data download failed.
    pause
    exit /b 1
)

echo.
echo [3/4] Preparing the 383-record modeling dataset ...
python src\prepare_data.py
if errorlevel 1 (
    echo ERROR: Data preparation failed.
    pause
    exit /b 1
)

echo.
echo [4/4] Training FLANN, MLP, ARIMA(1,1,1) and evaluating ...
python src\run_project.py
if errorlevel 1 (
    echo ERROR: Pipeline run failed.
    pause
    exit /b 1
)

echo.
echo Done. See the results\ folder for:
echo   - model_comparison.csv
echo   - test_predictions.csv
echo   - forecast_comparison.png
echo.
pause
