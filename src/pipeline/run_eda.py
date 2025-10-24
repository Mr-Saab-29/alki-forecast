from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Build a simple EDA summary DataFrame.
    
    Args:
        df: Input cleaned DataFrame with at least 'CUSTOMER', 'DATE', and 'QUANTITY' columns.

    Returns:
        DataFrame with EDA summary statistics per customer.
        
    """
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    grouped = (
        df.groupby("CUSTOMER")
        .agg(
            start_date=("DATE", "min"),
            end_date=("DATE", "max"),
            n_rows=("QUANTITY", "size"),
            mean_qty=("QUANTITY", "mean"),
            median_qty=("QUANTITY", "median"),
            std_qty=("QUANTITY", "std"),
            total_qty=("QUANTITY", "sum"),
        )
        .reset_index()
    )
    grouped["start_date"] = grouped["start_date"].dt.strftime("%Y-%m-%d")
    grouped["end_date"] = grouped["end_date"].dt.strftime("%Y-%m-%d")
    return grouped


def main(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data at {data_path}")

    df = pd.read_csv(data_path)
    eda_df = build_eda(df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "eda_summary.csv"
    summary_json = output_dir / "eda_summary.json"

    eda_df.to_csv(summary_csv, index=False)
    eda_df.to_json(summary_json, orient="records", indent=2)
    print(f"EDA complete: outputs at {summary_csv} and {summary_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simple EDA summary.")
    parser.add_argument(
        "--data-path",
        default="data/processed/cleaned_data.csv",
        help="Path to cleaned data CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Directory to store EDA outputs.",
    )
    main(parser.parse_args())
