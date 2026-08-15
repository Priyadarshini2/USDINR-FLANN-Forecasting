from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "usdinr_raw.csv"
OUTPUT_PATH = ROOT / "data" / "usdinr_383.csv"


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

RAW_OBSERVATIONS = 387
LAGS = 4

SUPERVISED_SAMPLES = 383

TRAIN_SAMPLES = 335
TEST_SAMPLES = 48


# ============================================================
# PREPARE DATA
# ============================================================

def prepare():

    print("=" * 60)
    print("PREPARING USD/INR DATA")
    print("=" * 60)

    # --------------------------------------------------------
    # Check raw file
    # --------------------------------------------------------

    if not RAW_PATH.exists():

        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_PATH}"
        )

    # --------------------------------------------------------
    # Read the raw FRED data
    # --------------------------------------------------------

    df = pd.read_csv(
        RAW_PATH
    )

    print("Columns found in raw dataset:")
    print(list(df.columns))
    print()

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "date",
        "usd_inr"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column '{column}' not found."
            )

    # --------------------------------------------------------
    # Convert data types
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["usd_inr"] = pd.to_numeric(
        df["usd_inr"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "date",
            "usd_inr"
        ]
    )

    # Sort by date
    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Total valid raw observations: {len(df)}"
    )

    # ========================================================
    # SELECT 387 OBSERVATIONS
    # ========================================================

    if len(df) < RAW_OBSERVATIONS:

        raise ValueError(
            f"Need {RAW_OBSERVATIONS} observations, "
            f"but only {len(df)} are available."
        )

    # Use the most recent 387 observations
    df = df.tail(
        RAW_OBSERVATIONS
    ).reset_index(drop=True)

    print(
        f"Observations selected for experiment: {len(df)}"
    )

    # ========================================================
    # CREATE LAGGED SUPERVISED DATA
    # ========================================================

    rows = []

    for i in range(
        LAGS,
        len(df)
    ):

        row = {

            "date":
                df.loc[i, "date"],

            "lag_4":
                df.loc[i - 4, "usd_inr"],

            "lag_3":
                df.loc[i - 3, "usd_inr"],

            "lag_2":
                df.loc[i - 2, "usd_inr"],

            "lag_1":
                df.loc[i - 1, "usd_inr"],

            "target":
                df.loc[i, "usd_inr"]
        }

        rows.append(row)

    supervised = pd.DataFrame(
        rows
    )

    # ========================================================
    # VALIDATE EXPERIMENT SIZE
    # ========================================================

    print()
    print("Experiment validation:")
    print(
        f"Raw observations       : {len(df)}"
    )

    print(
        f"Lag observations       : {LAGS}"
    )

    print(
        f"Supervised samples     : {len(supervised)}"
    )

    print(
        f"Training samples       : {TRAIN_SAMPLES}"
    )

    print(
        f"Testing samples        : {TEST_SAMPLES}"
    )

    # Check 387 - 4 = 383
    assert (
        len(supervised)
        == SUPERVISED_SAMPLES
    ), (
        f"Expected {SUPERVISED_SAMPLES} "
        f"samples, got {len(supervised)}"
    )

    # Check 335 + 48 = 383
    assert (
        TRAIN_SAMPLES + TEST_SAMPLES
        == SUPERVISED_SAMPLES
    )

    # ========================================================
    # SAVE
    # ========================================================

    supervised.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )

    # ========================================================
    # DISPLAY FIRST 5 ROWS
    # ========================================================

    print()
    print("First 5 supervised records:")
    print()

    print(
        supervised
        .head()
        .to_string(index=False)
    )

    # ========================================================
    # DISPLAY LAST 5 ROWS
    # ========================================================

    print()
    print("Last 5 supervised records:")
    print()

    print(
        supervised
        .tail()
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    prepare()