# src/eval/run_candidates.py
from __future__ import annotations
import copy
import numpy as np, pandas as pd, yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from src.utils.timeseries_split import compute_min_hist, rolling_time_series_cv, select_by_index
from src.features.build_features import build_features
from src.features.future_features import build_future_features

from src.models import arima_like as arima
from src.models.prophet_model import fit_forecast_prophet
from src.models.gbm import fit_predict_gbm_recursive
from src.eval.run_baselines import mae, rmse, smape
from src.utils.peak_metrics import compute_peak_metrics

def _build_features(
    df_slice,
    *,
    max_lag,
    roll_windows,
    holiday_country,
    holiday_subdiv_map,
    holiday_window,
    trim_by_history,
    dropna_mode,
    feature_set: str = "full",
):
    """
    Wrapper to build features with specified parameters.
    
    Args:
        df_slice (pd.DataFrame): DataFrame slice for a specific customer.
        max_lag (int): Maximum lag to create.
        roll_windows (list[int]): List of window sizes for rolling means.
        holiday_country (str): Country code for holiday features.
        holiday_subdiv_map (Optional[Dict[str, str]]): Mapping of CUSTOMER to holiday
        holiday_window (int): Window size for holiday effect smoothing.
        trim_by_history (bool): Whether to trim by history.
        dropna_mode (str): Mode for dropping NA values.
    
    Returns:
        pd.DataFrame: DataFrame with built features.
    """
    return build_features(
        df_slice,
        max_lag=max_lag,
        roll_windows=roll_windows,
        holiday_country=holiday_country,
        holiday_subdiv_map=holiday_subdiv_map,
        holiday_window=holiday_window,
        trim_by_history=trim_by_history,
        dropna_mode=dropna_mode,
        feature_set=feature_set,
    )

def _to_series(df: pd.DataFrame) -> pd.Series:
    s = df[["DATE","QUANTITY"]].set_index("DATE").sort_index()["QUANTITY"].asfreq("D")
    return s.fillna(0.0)

def _apply_transform(y: np.ndarray, transform: str, inverse: bool=False) -> np.ndarray:
    if transform == "log1p":
        return (np.expm1(y) if inverse else np.log1p(y))
    return y

def run_candidates_per_customer(
    df_clean: pd.DataFrame,
    model_matrix_path: str | Path,
    *,
    # CV
    n_folds: int = 5, window_type: str = "expanding", step_days: int = 7, horizon_days: int = 25,
    gap_days: int = 0, train_window_days: int = 365, initial_train_days: int = 90,
    # features
    max_lag: int = 30, roll_windows: List[int] = [7,14,30], holiday_country: str = "FR", holiday_subdiv_map: Optional[Dict[str,str]] = None, holiday_window: int = 3,
    trim_by_history: bool = True, dropna_mode: str = "none",
    # output
    out_dir: str | Path = "outputs/cv/candidates", save_csv: bool = True,
    composite_weights: Tuple[float, float, float] = (0.3, 0.5, 0.2),
    best_yaml_path: str | Path | None = None,
    write_best_yaml: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run candidate models per customer over rolling CV folds as specified in model_matrix_path.
    Returns (per_fold_df, summary_df, peak_metrics_df, best_models_df). Optionally writes CSVs and
    a tuned YAML with the composite-best model per customer.

    Args:
        df_clean (pd.DataFrame): Cleaned input DataFrame with columns CUSTOMER, DATE, QUANTITY.
        model_matrix_path (str | Path): Path to YAML file specifying candidate models per customer.
        n_folds (int): Number of CV folds.
        window_type (str): Type of CV window ('expanding' or 'sliding').
        step_days (int): Step size in days between folds.
        horizon_days (int): Forecast horizon in days.
        gap_days (int): Gap days between training and validation.
        train_window_days (int): Training window size in days (for sliding window).
        initial_train_days (int): Minimum initial training days before first fold.
        max_lag (int): Maximum lag for feature engineering.
        roll_windows (List[int]): List of rolling window sizes for features.
        holiday_country (str): Country code for holiday features.
        holiday_subdiv_map (Optional[Dict[str,str]]): Mapping of CUSTOMER to holiday subdivision.
        holiday_window (int): Window size for holiday effect smoothing.
        trim_by_history (bool): Whether to trim features by history.
        dropna_mode (str): Mode for dropping NA values in features.
        out_dir (str | Path): Output directory to save results.
        save_csv (bool): Whether to save output DataFrames as CSV files.
        composite_weights (Tuple[float, float, float]): Weights for (MAE, RMSE, sMAPE) in composite score.
        best_yaml_path (str | Path | None): Path to save best models YAML; if None, defaults to out_dir/best_models_composite.yaml.
        write_best_yaml (bool): Whether to write the best models YAML file.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - per_fold_df: DataFrame with per-fold metrics for all candidates.
            - summary_df: DataFrame with average metrics per customer and model.
            - peak_metrics_df: DataFrame with peak metrics for predictions.
            - best_models_df: DataFrame with the best model per customer based on composite score.
    """
    def _log_error(cust, fold, model, err, rows, horizon, anchor=None):
        msg = f"[ERROR] {cust} | Fold {fold} | Model: {model} → {type(err).__name__}: {err}"
        print(msg)
        rows.append({
            "CUSTOMER": cust, "fold": fold, "anchor": anchor, "model": model + "_ERROR",
            "MAE": np.nan, "RMSE": np.nan, "sMAPE": np.nan, "n": horizon, "error": str(err)
        })

    # -----------------------------
    # Hardened param sanitizers
    # -----------------------------

    def _norm_key(k: str) -> str:
        """Normalize YAML keys like 'p:1' or ' P ' to canonical form."""
        if not isinstance(k, str):
            return k
        k = k.strip()
        if ":" in k:
            k = k.split(":", 1)[0].strip()
        return k

    def _coerce_int(x, default: int) -> int:
        """Coerce various YAML values (None, '', '1', '1.0', 1.0) to int safely."""
        try:
            if x is None:
                return default
            if isinstance(x, str):
                s = x.strip()
                if s == "":
                    return default
                if ":" in s:               # e.g., "p:1"
                    s = s.split(":", 1)[-1].strip()
                # try as int, then float→int
                return int(float(s))
            if isinstance(x, (int, np.integer)):
                return int(x)
            if isinstance(x, (float, np.floating)):
                return int(round(float(x)))
            return default
        except Exception:
            return default

    def _coerce_float(x, default: float) -> float:
        """Coerce various YAML values to float safely."""
        try:
            if x is None:
                return default
            if isinstance(x, str):
                s = x.strip()
                if s == "":
                    return default
                if ":" in s:
                    s = s.split(":", 1)[-1].strip()
                return float(s)
            if isinstance(x, (int, np.integer, float, np.floating)):
                return float(x)
            return default
        except Exception:
            return default

    def _sanitize_params(d: dict) -> dict:
        """Normalize keys (strip, split ':'), keep raw values (coercion later)."""
        out = {}
        for k, v in (d or {}).items():
            out[_norm_key(k)] = v
        return out

    def _first_from_grid(grid: dict) -> dict:
        """Choose the first candidate for each grid key; normalize keys."""
        if not isinstance(grid, dict):
            return {}
        out = {}
        for k, v in grid.items():
            key = _norm_key(k)
            if isinstance(v, (list, tuple)) and len(v):
                out[key] = v[0]
            else:
                out[key] = v
        return out

    def _sanitize_arima_params(raw: dict) -> dict:
        """Return complete ARIMA params with safe defaults (no None)."""
        d = _sanitize_params(raw)
        return {
            "p": _coerce_int(d.get("p"), 1),
            "d": _coerce_int(d.get("d"), 1),
            "q": _coerce_int(d.get("q"), 1),
        }

    def _sanitize_sarima_params(raw: dict) -> dict:
        """Return complete SARIMA params with safe defaults (no None)."""
        d = _sanitize_params(raw)
        return {
            "p":  _coerce_int(d.get("p"), 1),
            "d":  _coerce_int(d.get("d"), 1),
            "q":  _coerce_int(d.get("q"), 1),
            "P":  _coerce_int(d.get("P"), 0),
            "D":  _coerce_int(d.get("D"), 1),
            "Q":  _coerce_int(d.get("Q"), 1),
            "sp": _coerce_int(d.get("sp"), 7),
        }

    def _pick_sanitized(order_type: str, grid: Optional[dict], params: dict) -> dict:
        """
        If a grid is present, pick first values then sanitize; else sanitize params.
        order_type ∈ {'arima','sarima'}.
        """
        raw = _first_from_grid(grid) if isinstance(grid, dict) else params
        if order_type == "arima":
            return _sanitize_arima_params(raw or {})
        else:
            return _sanitize_sarima_params(raw or {})

    cfg = yaml.safe_load(Path(model_matrix_path).read_text())
    df = df_clean.copy()
    if not np.issubdtype(df["DATE"].dtype, np.datetime64):
        df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(["CUSTOMER","DATE"]).reset_index(drop=True)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    min_hist = compute_min_hist(max_lag, roll_windows)

    rows: list[dict] = []
    detail_rows: list[pd.DataFrame] = []

    for cust, spec in cfg["customers"].items():
        df_c = df[df["CUSTOMER"] == cust].reset_index(drop=True)
        if df_c.empty:
            continue

        transform = spec.get("transform", "raw")
        models = spec.get("models", [])
        cv_cfg = spec.get("cv", {}) or {}
        cust_window_type = cv_cfg.get("window_type", window_type)
        cust_train_window_days = cv_cfg.get("train_window_days", train_window_days)
        cust_step_days = cv_cfg.get("step_days", step_days)
        cust_gap_days = cv_cfg.get("gap_days", gap_days)
        cust_horizon_days = cv_cfg.get("horizon_days", horizon_days)
        cust_n_folds = cv_cfg.get("n_folds", n_folds)
        cust_initial_train_days = cv_cfg.get("initial_train_days", initial_train_days)

        cust_n_folds = _coerce_int(cust_n_folds, n_folds)
        cust_step_days = _coerce_int(cust_step_days, step_days)
        cust_gap_days = _coerce_int(cust_gap_days, gap_days)
        cust_horizon_days = _coerce_int(cust_horizon_days, horizon_days)
        cust_initial_train_days = _coerce_int(cust_initial_train_days, initial_train_days)
        if cust_window_type == "sliding":
            base_window = train_window_days if train_window_days is not None else 0
            cust_train_window_days = _coerce_int(cust_train_window_days, base_window)
            if cust_train_window_days <= 0:
                raise ValueError(f"Customer {cust} uses sliding CV but train_window_days is not set.")
        else:
            cust_train_window_days = None

        folds = rolling_time_series_cv(
            df_c,
            n_folds=cust_n_folds,
            window_type=cust_window_type,
            train_window_days=cust_train_window_days,
            step_days=cust_step_days,
            horizon_days=cust_horizon_days,
            gap_days=cust_gap_days,
            by_customer=True,
            min_hist=min_hist,
            initial_train_days=cust_initial_train_days,
        )

        if not folds:
            _log_error(cust, None, "NO_FOLDS", Exception("No valid folds"), rows, 0)
            continue

        for f in folds:
            train_df = select_by_index(df_c, f.train_idx)
            val_df = select_by_index(df_c, f.val_idx)

            s_train = _to_series(train_df)
            horizon = val_df["DATE"].nunique()

            feats_tr = _build_features(
                train_df,
                max_lag=max_lag,
                roll_windows=roll_windows,
                holiday_country=holiday_country,
                holiday_subdiv_map=holiday_subdiv_map,
                holiday_window=holiday_window,
                trim_by_history=False,
                dropna_mode="none",
                feature_set="deterministic",
            )
            exog_cols = [c for c in feats_tr.columns if c not in {"DATE", "CUSTOMER", "QUANTITY"}]
            feats_tr = feats_tr.set_index("DATE").sort_index().reindex(s_train.index)
            exog_tr = feats_tr[exog_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

            fut_feats = build_future_features(
                train_df,
                horizon=horizon,
                max_lag=max_lag,
                roll_windows=roll_windows,
                holiday_country=holiday_country,
                holiday_subdiv_map=holiday_subdiv_map,
                holiday_window=holiday_window,
                feature_set="deterministic",
            )
            fut_feats = fut_feats.set_index("DATE").sort_index()
            exog_fut = fut_feats.reindex(columns=exog_cols).fillna(0.0).replace([np.inf, -np.inf], 0.0)

            exog_tr_arr = None if exog_tr.empty else exog_tr.to_numpy()
            exog_fut_arr = None if exog_fut.empty else exog_fut.to_numpy()

            dates_sorted = val_df.sort_values("DATE")
            y_true = dates_sorted["QUANTITY"].to_numpy()
            dates_array = dates_sorted["DATE"].to_numpy()

            for m in models:
                name = m["name"]
                family = m["family"]
                params = m.get("params", {}).copy()

                try:
                    if family == "arima":
                        params = _pick_sanitized("arima", m.get("grid"), params)
                        y_tr = _apply_transform(s_train.values, transform, inverse=False)
                        s_tr = pd.Series(y_tr, index=s_train.index)
                        yhat = arima.fit_forecast_arima(
                            s_tr,
                            horizon=horizon,
                            exog_train=exog_tr_arr,
                            exog_future=exog_fut_arr,
                            **params,
                        )
                        yhat = _apply_transform(yhat, transform, inverse=True)

                    elif family == "sarima":
                        params = _pick_sanitized("sarima", None, params)
                        y_tr = _apply_transform(s_train.values, transform, inverse=False)
                        s_tr = pd.Series(y_tr, index=s_train.index)
                        try:
                            yhat = arima.fit_forecast_sarima(
                                s_tr,
                                horizon=horizon,
                                exog_train=exog_tr_arr,
                                exog_future=exog_fut_arr,
                                **params,
                            )
                        except Exception:
                            yhat = arima.fit_forecast_sarima(
                                s_tr,
                                horizon=horizon,
                                p=0,
                                d=1,
                                q=1,
                                P=0,
                                D=1,
                                Q=1,
                                sp=7,
                                exog_train=exog_tr_arr,
                                exog_future=exog_fut_arr,
                            )
                        yhat = _apply_transform(yhat, transform, inverse=True)

                    elif family == "ets":
                        y_tr = _apply_transform(s_train.values, transform, inverse=False)
                        s_tr = pd.Series(y_tr, index=s_train.index)
                        yhat = arima.fit_forecast_ets(s_tr, horizon=horizon, **params)
                        yhat = _apply_transform(yhat, transform, inverse=True)

                    elif family == "prophet":
                        try:
                            exog_tr_df = None if exog_tr.empty else exog_tr
                            exog_fut_df = None if exog_fut.empty else exog_fut
                            yhat = fit_forecast_prophet(
                                s_train,
                                horizon=horizon,
                                exog_train=exog_tr_df,
                                exog_future=exog_fut_df,
                                **params,
                            )
                        except AttributeError as e:
                            if "stan_backend" in str(e):
                                yhat = arima.fit_forecast_ets(
                                    s_train, horizon=horizon, trend="add", seasonal="add", sp=7
                                )
                            else:
                                raise

                    elif family == "gbm":
                        use_lgb = bool(m.get("use_lightgbm", False))
                        yhat = fit_predict_gbm_recursive(
                            train_df,
                            build_features_fn=lambda d: _build_features(
                                d, max_lag=max_lag, roll_windows=roll_windows,
                                holiday_country=holiday_country, holiday_subdiv_map=holiday_subdiv_map,
                                holiday_window=holiday_window, trim_by_history=trim_by_history,
                                dropna_mode=dropna_mode,
                            ),
                            build_future_features_fn=lambda d, horizon: build_future_features(d, horizon=horizon),
                            horizon=horizon,
                            params=params,
                            transform=transform,
                            use_lightgbm=use_lgb,
                            max_lag=max_lag,
                            roll_windows=roll_windows,
                        )

                    else:
                        raise ValueError(f"Unknown model family: {family}")

                    metric_mae = mae(y_true, yhat)
                    metric_rmse = rmse(y_true, yhat)
                    metric_smape = smape(y_true, yhat)

                    rows.append({
                        "CUSTOMER": cust,
                        "fold": f.fold,
                        "anchor": f.meta["anchor"].date(),
                        "model": name,
                        "MAE": metric_mae,
                        "RMSE": metric_rmse,
                        "sMAPE": metric_smape,
                        "n": horizon,
                    })

                    detail_rows.append(pd.DataFrame({
                        "CUSTOMER": cust,
                        "model": name,
                        "family": family,
                        "fold": f.fold,
                        "DATE": dates_array,
                        "y_true": y_true,
                        "y_pred": yhat,
                    }))

                except Exception as e:
                    _log_error(cust, f.fold, name, e, rows, horizon, anchor=f.meta["anchor"].date())

    # Convert to DataFrame
    per_fold = pd.DataFrame(rows).sort_values(["CUSTOMER", "model", "fold"]).reset_index(drop=True)
    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()

    # Save separate error log
    err_df = per_fold[per_fold["model"].str.contains("_ERROR", na=False)]
    if not err_df.empty:
        err_df.to_csv(out_path / "candidates_errors.csv", index=False)

    # Summary (ignore error rows)
    summary = (
        per_fold[~per_fold["model"].str.contains("_ERROR", na=False)]
        .groupby(["CUSTOMER", "model"], as_index=False)[["MAE", "RMSE", "sMAPE"]]
        .mean()
        .reset_index(drop=True)
    )

    if not summary.empty:
        for metric in ["MAE", "RMSE", "sMAPE"]:
            col_norm = f"{metric}_norm"
            summary[col_norm] = 0.0
            for cust, idx in summary.groupby("CUSTOMER").groups.items():
                values = summary.loc[idx, metric]
                best_val = values.min()
                denom = best_val if best_val > 1e-8 else 1e-8
                summary.loc[idx, col_norm] = values / denom
        w_mae, w_rmse, w_smape = composite_weights
        summary["CompositeScore"] = (
            summary["MAE_norm"] * w_mae
            + summary["RMSE_norm"] * w_rmse
            + summary["sMAPE_norm"] * w_smape
        )
        summary = summary.sort_values(["CUSTOMER", "CompositeScore"]).reset_index(drop=True)
        best_idx = summary.groupby("CUSTOMER")["CompositeScore"].idxmin()
        best_models_df = summary.loc[best_idx].reset_index(drop=True)
    else:
        best_models_df = pd.DataFrame(columns=["CUSTOMER", "model", "CompositeScore"])

    peak_metrics = compute_peak_metrics(detail_df) if not detail_df.empty else pd.DataFrame()

    if write_best_yaml and not best_models_df.empty:
        best_cfg = {"customers": {}}
        for _, row in best_models_df.iterrows():
            cust = row["CUSTOMER"]
            model_name = row["model"]
            orig_spec = copy.deepcopy(cfg["customers"].get(cust, {}))
            chosen = None
            for entry in orig_spec.get("models", []):
                if entry.get("name") == model_name:
                    chosen = copy.deepcopy(entry)
                    break
            if chosen is None:
                continue
            cust_cfg = {
                "transform": orig_spec.get("transform", "raw"),
                "cv": orig_spec.get("cv", {}),
                "models": [chosen],
            }
            best_cfg["customers"][cust] = cust_cfg

        best_path = Path(best_yaml_path) if best_yaml_path else (Path(out_dir) / "best_models_composite.yaml")
        best_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_path, "w") as f:
            yaml.safe_dump(best_cfg, f, sort_keys=False)

    if save_csv:
        per_fold.to_csv(out_path / "candidates_per_fold.csv", index=False)
        summary.to_csv(out_path / "candidates_summary.csv", index=False)
        if not detail_df.empty:
            detail_df.to_csv(out_path / "candidates_predictions.csv", index=False)
        if not peak_metrics.empty:
            peak_metrics.to_csv(out_path / "candidates_peak_metrics.csv", index=False)

    return per_fold, summary, peak_metrics, best_models_df
