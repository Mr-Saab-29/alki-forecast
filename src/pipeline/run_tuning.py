from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from src.eval.tuner import tune_per_customer


def main(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data at {data_path}")

    df_clean = pd.read_csv(data_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_fold, summary, tuned_cfg = tune_per_customer(
            df_clean=df_clean,
            base_yaml_path=args.base_config,
            search=args.search,
            n_trials=args.n_trials,
            metric=args.metric,
            out_yaml_path=args.output_config,
            cv_defaults=None,
            features_defaults=None,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_fold.to_csv(output_dir / "tuning_per_fold.csv", index=False)
    summary.to_csv(output_dir / "tuning_summary.csv", index=False)

    print(f"Tuning complete: configuration -> {args.output_config}")
    print(f"Tuning metrics stored in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameter tuning per customer.")
    parser.add_argument(
        "--data-path",
        default="data/processed/cleaned_data.csv",
        help="Path to cleaned data CSV.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/model_matrix.yaml",
        help="Base model matrix YAML to tune from.",
    )
    parser.add_argument(
        "--output-config",
        default="configs/model_matrix_tuned.yaml",
        help="Destination YAML for tuned configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tuning",
        help="Directory to store tuning metrics.",
    )
    parser.add_argument(
        "--search",
        default="grid",
        choices=["grid", "random"],
        help="Search strategy per model.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of trials for random search (ignored for grid).",
    )
    parser.add_argument(
        "--metric",
        default="sMAPE",
        choices=["MAE", "RMSE", "sMAPE"],
        help="Metric used for model selection.",
    )
    main(parser.parse_args())
