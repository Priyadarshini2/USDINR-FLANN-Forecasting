"""
run_project.py
--------------
End-to-end driver for the USD/INR FLANN forecasting project.

Usage:
    python src/run_project.py
    python src/run_project.py --download
"""

import argparse
import os
import sys
import runpy

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "usdinr_383.csv")

sys.path.insert(0, SRC_DIR)

import download_data
import prepare_data


def ensure_data(force_download=False):
    """Ensure the raw and prepared USD/INR datasets exist."""

    raw_path = os.path.join(DATA_DIR, "usdinr_raw.csv")

    if force_download or not os.path.exists(raw_path):
        print("=" * 70)
        print("DOWNLOADING REAL USD/INR DATA")
        print("=" * 70)
        download_data.download()

    if force_download or not os.path.exists(DATASET_PATH):
        print("=" * 70)
        print("PREPARING USD/INR DATA")
        print("=" * 70)
        prepare_data.prepare()


def main(force_download=False):

    print("=" * 70)
    print("USD/INR FLANN FORECASTING PROJECT")
    print("=" * 70)

    ensure_data(force_download=force_download)

    print()
    print("=" * 70)
    print("RUNNING MODEL COMPARISON")
    print("=" * 70)
    print("Models: FLANN, MLP, ARIMA(1,1,1)")
    print("Dataset: 383 supervised samples")
    print("Training: 335 samples")
    print("Testing : 48 samples")
    print("=" * 70)
    print()

    # model_comparison.py is already a complete executable pipeline.
    # Run it directly instead of importing it and calling nonexistent
    # helper functions.
    model_comparison_path = os.path.join(
        SRC_DIR,
        "model_comparison.py"
    )

    runpy.run_path(
        model_comparison_path,
        run_name="__main__"
    )

    print()
    print("=" * 70)
    print("PROJECT PIPELINE COMPLETE")
    print("=" * 70)

    print("Results are available in:")
    print(
        os.path.join(
            ROOT_DIR,
            "results"
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the complete USD/INR FLANN forecasting project."
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Force re-download and re-prepare the latest USD/INR data."
    )

    args = parser.parse_args()

    main(
        force_download=args.download
    )