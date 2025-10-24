from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.run_candidates import run_candidates_per_customer
import warnings


def main(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data at {data_path}")
    df_clean = pd.read_csv(data_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_fold, summary, peak_metrics, best_models = run_candidates_per_customer(
            df_clean=df_clean,
            model_matrix_path=args.config,
            n_folds=args.n_folds,
            window_type=args.window_type,
            step_days=args.step_days,
            horizon_days=args.horizon_days,
            initial_train_days=args.initial_train_days,
            out_dir=out_dir,
            save_csv=True,
            best_yaml_path=args.best_yaml,
        )

    print(f"Training complete: candidate metrics in {out_dir}")
    if not peak_metrics.empty:
        print("Composite best models:")
        print(best_models[["CUSTOMER", "model", "CompositeScore"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training / candidate evaluation.")
    parser.add_argument(
        "--data-path",
        default="data/processed/cleaned_data.csv",
        help="Path to cleaned data CSV.",
    )
    parser.add_argument(
        "--config",
        default="configs/model_matrix.yaml",
        help="Model matrix YAML describing candidate models.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of CV folds."
    )
    parser.add_argument(
        "--window-type",
        default="expanding",
        choices=["expanding", "sliding"],
        help="CV window type.",
    )
    parser.add_argument(
        "--step-days", type=int, default=7, help="CV step between folds."
    )
    parser.add_argument(
        "--horizon-days", type=int, default=25, help="Validation horizon length."
    )
    parser.add_argument(
        "--initial-train-days",
        type=int,
        default=90,
        help="Initial warm-up window in days.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cv/candidates",
        help="Directory to store CV outputs.",
    )
    parser.add_argument(
        "--best-yaml",
        default="outputs/cv/candidates/best_models_composite.yaml",
        help="Path to write the best-model YAML.",
    )
    main(parser.parse_args())
