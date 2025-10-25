from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import load_raw
from src.data.preprocess import preprocess_all_customers


def main(args: argparse.Namespace) -> None:
    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    print(f"Loading raw data from {raw_path}")
    raw_df = load_raw(raw_path)

    print("Processing data with preprocess_all_customers...")
    cleaned_df, summary = preprocess_all_customers(
        raw_df,
        long_gap_days=30,
        min_nonzero_run=5,
        min_nonzero_value=1.0,
        gap_limit=7,
        causal=True,
        verbose=True,)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    summary_path = output_path.with_name(output_path.stem + '_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Preprocessing complete. Cleaned data saved to {output_path}, summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw Alki demand data.")
    parser.add_argument(
        "--raw-path",
        default="data/raw/train set.csv",
        help="Path to the raw CSV file (must exist).",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/cleaned_data.csv",
        help="Destination for the processed CSV.",
    )
    main(parser.parse_args())