from pathlib import Path
from io import StringIO

import pandas as pd
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "data" / "usdinr_raw.csv"


# ============================================================
# FRED DATA SOURCE
# ============================================================

FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=DEXINUS"
)


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_data():

    print("=" * 60)
    print("DOWNLOADING REAL USD/INR DATA")
    print("=" * 60)

    print("Source:")
    print(FRED_URL)
    print()

    # --------------------------------------------------------
    # Download CSV
    # --------------------------------------------------------

    response = requests.get(
        FRED_URL,
        timeout=30
    )

    response.raise_for_status()

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        StringIO(response.text)
    )

    print("Columns received from FRED:")
    print(list(df.columns))
    print()

    # --------------------------------------------------------
    # Find date column
    # --------------------------------------------------------

    possible_date_columns = [
        "DATE",
        "date",
        "observation_date"
    ]

    date_column = None

    for column in possible_date_columns:

        if column in df.columns:
            date_column = column
            break

    if date_column is None:

        raise ValueError(
            "Could not find the date column. "
            f"Columns received: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Find USD/INR column
    # --------------------------------------------------------

    possible_rate_columns = [
        "DEXINUS",
        "dexinus",
        "USDINR",
        "usd_inr"
    ]

    rate_column = None

    for column in possible_rate_columns:

        if column in df.columns:
            rate_column = column
            break

    if rate_column is None:

        raise ValueError(
            "Could not find the USD/INR column. "
            f"Columns received: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    df = df.rename(
        columns={
            date_column: "date",
            rate_column: "usd_inr"
        }
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

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "date",
            "usd_inr"
        ]
    )

    after = len(df)

    removed = before - after

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUT,
        index=False
    )

    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print("=" * 60)
    print("USD/INR DATA DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"Date column used       : {date_column}")
    print(f"Exchange-rate column   : {rate_column}")
    print(f"Valid observations     : {len(df)}")
    print(f"Missing rows removed   : {removed}")
    print(
        f"Start date             : "
        f"{df['date'].min().date()}"
    )
    print(
        f"End date               : "
        f"{df['date'].max().date()}"
    )

    print()
    print("First 5 observations:")
    print(
        df.head().to_string(index=False)
    )

    print()
    print("Last 5 observations:")
    print(
        df.tail().to_string(index=False)
    )

    print()
    print("Saved to:")
    print(OUT)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    download_data()