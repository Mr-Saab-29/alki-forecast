from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline.save_models_and_forecast import save_models_and_forecast
import warnings


def main(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data at {data_path}")

    df_clean = pd.read_csv(data_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast_df = save_models_and_forecast(
            df_clean=df_clean,
            best_yaml_path=args.best_yaml,
            horizon=args.horizon,
            models_dir=args.models_dir,
            forecasts_path=args.forecast_path,
        )

    print(f"Forecasting complete: models -> {args.models_dir}, forecasts -> {args.forecast_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit best models and produce forecasts.")
    parser.add_argument(
        "--data-path",
        default="data/processed/cleaned_data.csv",
        help="Path to cleaned data CSV.",
    )
    parser.add_argument(
        "--best-yaml",
        default="outputs/cv/candidates/best_models_composite.yaml",
        help="Path to best-model YAML produced in training stage.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=21,
        help="Forecast horizon (days).",
    )
    parser.add_argument(
        "--models-dir",
        default="outputs/models",
        help="Directory to store fitted model artifacts.",
    )
    parser.add_argument(
        "--forecast-path",
        default="outputs/forecasts/forecast_quantiles.csv",
        help="Destination CSV for forecast quantiles.",
    )
    main(parser.parse_args())
