from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_customer(
    df_clean: pd.DataFrame,
    forecast_df: pd.DataFrame,
    customer: str,
    output_dir: Path,
    history_days: int,
) -> None:
    """Plot actuals and forecast quantiles for a single customer.
    Args:
        df_clean: Cleaned historical data.
        forecast_df: Forecast quantiles data.
        customer: Customer name to plot.
        output_dir: Directory to save the plot.
        history_days: Number of days of history to show before forecast horizon.
    
    Returns:
        None
    """
    hist = (
        df_clean[df_clean["CUSTOMER"] == customer]
        .sort_values("DATE")
        .tail(history_days)
    )
    preds = forecast_df[forecast_df["CUSTOMER"] == customer].sort_values("DATE")
    if hist.empty or preds.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist["DATE"], hist["QUANTITY"], label="Actual", color="black")
    ax.plot(preds["DATE"], preds["P50"], label="Forecast P50", color="tab:blue")
    ax.fill_between(
        preds["DATE"],
        preds["P10"],
        preds["P90"],
        color="tab:blue",
        alpha=0.25,
        label="P10-P90",
    )
    ax.set_title(f"{customer} – Actuals and {len(preds)}-day Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{customer.replace(' ', '_')}.png", dpi=200)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    df_clean = pd.read_csv(args.data_path)
    df_clean["DATE"] = pd.to_datetime(df_clean["DATE"])
    forecast_df = pd.read_csv(args.forecast_path)
    forecast_df["DATE"] = pd.to_datetime(forecast_df["DATE"])

    customers = forecast_df["CUSTOMER"].unique()
    output_dir = Path(args.output_dir)

    for cust in customers:
        plot_customer(df_clean, forecast_df, cust, output_dir, args.history_days)

    print(f"Plotting complete: files in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot forecast quantiles with recent history.")
    parser.add_argument(
        "--data-path",
        default="data/processed/cleaned_data.csv",
        help="Path to cleaned data CSV.",
    )
    parser.add_argument(
        "--forecast-path",
        default="outputs/forecasts/forecast_quantiles.csv",
        help="CSV produced by run_forecast.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="Directory to save PNG plots.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=90,
        help="Number of most recent days to show before the forecast horizon.",
    )
    main(parser.parse_args())
