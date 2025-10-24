from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.utils.timeseries_split import compute_min_hist, rolling_time_series_cv, select_by_index
from src.features.build_features import build_features

# Metrics
def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return float(100.0 * np.mean(2.0 * num / den))

def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(e**2)))

# Baseline Models
def _naive_last_value(train_series: pd.Series, horizon: int) -> np.ndarray:
    """Forecast horizon steps using the last observed level from the training series.
    
    Args:
        train_series: pd.Series of historical values (may contain NaNs).
        horizon: int, number of steps to forecast.
    Returns:
        np.ndarray of shape (horizon,) with the forecasted values.
    """
    if train_series.empty:
        return np.zeros(horizon, dtype=float)
    last = float(train_series.dropna().iloc[-1]) if not train_series.dropna().empty else 0.0
    return np.full(horizon, last, dtype=float)

def _seasonal_weekly(train_series: pd.Series, horizon: int) -> np.ndarray:
    """
    Weekly seasonal naïve: repeat the most recent 7-day pattern from the training series.
    Falls back to last-value if fewer than 7 non-NaN observations exist.

    Args:
        train_series: pd.Series of historical values (may contain NaNs).
        horizon: int, number of steps to forecast.
    
    Returns:
        np.ndarray of shape (horizon,) with the forecasted values.
    """
    values = train_series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.zeros(horizon, dtype=float)
    if values.size >= 7:
        pattern = values[-7:]
    else:
        last = values[-1]
        pattern = np.full(7, last, dtype=float)
    preds = np.array([pattern[i % len(pattern)] for i in range(horizon)], dtype=float)
    return preds

def _ets_forecast(
    train_series: pd.Series,
    horizon: int,
    seasonal: str = "add",
    seasonal_periods: int = 7,
) -> np.ndarray:
    """
    Holt–Winters ETS forecast. seasonal ∈ {"add","mul"}.
    Fallback to last value if fitting fails.

    Args:
        train_series: pd.Series of historical values (may contain NaNs).
        horizon: int, number of steps to forecast.
        seasonal: str, "add" or "mul" for seasonal component.
        seasonal_periods: int, number of periods in a seasonal cycle.
    
    Returns:
        np.ndarray of shape (horizon,) with the forecasted values.

    """
    try:
        model = ExponentialSmoothing(
            train_series.astype(float),
            trend="add",                      # light trend
            seasonal=seasonal,                # additive or multiplicative
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)
        fc = model.forecast(steps=horizon)
        return np.asarray(fc, dtype=float)
    except Exception:
        last = float(train_series.iloc[-1])
        return np.full(horizon, last, dtype=float)

# Feature Builder Wrapper
def _build_features_slice(
    df_slice: pd.DataFrame,
    *,
    max_lag: int,
    roll_windows: List[int],
    holiday_country: str,
    holiday_subdiv_map: Optional[Dict[str, str]],
    holiday_window: int,
    trim_by_history: bool,
    dropna_mode: str,
) -> pd.DataFrame:
    return build_features(
        df_slice,
        max_lag=max_lag,
        roll_windows=roll_windows,
        holiday_country=holiday_country,
        holiday_subdiv_map=holiday_subdiv_map,
        holiday_window=holiday_window,
        trim_by_history=trim_by_history,
        dropna_mode=dropna_mode,
    )

# Orchestrator
def run_baselines_per_customer(
    df_clean: pd.DataFrame,
    *,
    # CV config
    n_folds: int = 5,
    window_type: str = "expanding",   # or "sliding"
    train_window_days: int = 365,     # used only for sliding
    step_days: int = 7,
    horizon_days: int = 25,
    gap_days: int = 0,
    initial_train_days: int = 90,
    cv_overrides: Optional[Dict[str, Dict[str, object]]] = None,
    # Features config
    max_lag: int = 30,
    roll_windows: List[int] = [7, 14, 30],
    holiday_country: str = "FR",
    holiday_subdiv_map: Optional[Dict[str, str]] = None,
    holiday_window: int = 3,
    trim_by_history: bool = True,
    dropna_mode: str = "none",
    # Output
    out_dir: str | Path = "outputs/cv",
    save_csv: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Naive-1, Seasonal-7, and ETS(Add/Mul) per customer over rolling CV folds.
    Returns (per_fold_df, summary_df). Optionally writes CSVs to out_dir.

    Args:
        df_clean: pd.DataFrame with columns ["CUSTOMER","DATE","QUANTITY"].
        n_folds: int, number of CV folds.
        window_type: str, "expanding" or "sliding".
        train_window_days: int, for sliding window, size of training window in days.
        step_days: int, number of days to step forward for each fold.
        horizon_days: int, forecast horizon in days.
        gap_days: int, gap between training end and validation start in days.
        initial_train_days: int, minimum initial training period in days.
        cv_overrides: optional dict mapping CUSTOMER to dict of CV params to override.
        max_lag: int, maximum lag feature to build.
        roll_windows: list of int, rolling window sizes for features.
        holiday_country: str, country code for holiday features.
        holiday_subdiv_map: optional dict mapping CUSTOMER to holiday subdivision code.
        holiday_window: int, window size for holiday features.
        trim_by_history: bool, whether to trim features by history availability.
        dropna_mode: str, how to handle NaNs in feature building.
        out_dir: str or Path, directory to save output CSVs.
        save_csv: bool, whether to save per-fold and summary CSVs.
    
    Returns:
        Tuple of (per_fold_df, summary_df).
    """
    df = df_clean.copy()
    if not np.issubdtype(df["DATE"].dtype, np.datetime64):
        df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(["CUSTOMER", "DATE"]).reset_index(drop=True)

    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    # history requirement
    min_hist = compute_min_hist(max_lag, roll_windows)

    rows = []
    customers = df["CUSTOMER"].dropna().unique().tolist()

    for cust in customers:
        df_c = df[df["CUSTOMER"] == cust].reset_index(drop=True)
        if df_c.empty:
            continue

        # Fold generation on the customer's own timeline
        cfg_cv = (cv_overrides or {}).get(cust, {}) or {}
        cust_window_type = cfg_cv.get("window_type", window_type)
        cust_train_window_days = cfg_cv.get("train_window_days", train_window_days)
        cust_step_days = cfg_cv.get("step_days", step_days)
        cust_gap_days = cfg_cv.get("gap_days", gap_days)
        cust_horizon_days = cfg_cv.get("horizon_days", horizon_days)
        cust_n_folds = cfg_cv.get("n_folds", n_folds)
        cust_initial_train_days = cfg_cv.get("initial_train_days", initial_train_days)

        def _coerce_int_local(value, default):
            try:
                if value is None:
                    return default
                return int(value)
            except Exception:
                return default

        cust_n_folds = _coerce_int_local(cust_n_folds, n_folds)
        cust_step_days = _coerce_int_local(cust_step_days, step_days)
        cust_gap_days = _coerce_int_local(cust_gap_days, gap_days)
        cust_horizon_days = _coerce_int_local(cust_horizon_days, horizon_days)
        cust_initial_train_days = _coerce_int_local(cust_initial_train_days, initial_train_days)
        if cust_window_type == "sliding":
            base_window = train_window_days if train_window_days is not None else 0
            cust_train_window_days = _coerce_int_local(cust_train_window_days, base_window)
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
            min_hist=min_hist,  # ensure lags/rolls are defined
            initial_train_days=cust_initial_train_days,
        )
        if not folds:
            rows.append({
                "CUSTOMER": cust, "fold": None, "anchor": None, "model": "NO_FOLDS",
                "MAE": np.nan, "RMSE": np.nan, "sMAPE": np.nan, "n": 0
            })
            continue

        # ETS flavor from your EDA
        ets_seasonality = "add" if cust == "ARGALYS" else "mul"

        for f in folds:
            train_df = select_by_index(df_c, f.train_idx)
            val_df   = select_by_index(df_c, f.val_idx)

            # Build features independently (leak-safe)
            Xy_tr = _build_features_slice(
                train_df,
                max_lag=max_lag, roll_windows=roll_windows,
                holiday_country=holiday_country,
                holiday_subdiv_map=holiday_subdiv_map,
                holiday_window=holiday_window,
                trim_by_history=trim_by_history,
                dropna_mode=dropna_mode,
            )
            Xy_va = _build_features_slice(
                val_df,
                max_lag=max_lag, roll_windows=roll_windows,
                holiday_country=holiday_country,
                holiday_subdiv_map=holiday_subdiv_map,
                holiday_window=holiday_window,
                trim_by_history=trim_by_history,
                dropna_mode=dropna_mode,
            )
            if Xy_va.empty:
                continue

            # Build chronologically aligned series for forecasting and evaluation
            train_series = (
                Xy_tr[["DATE","QUANTITY"]]
                .set_index("DATE")
                .sort_index()["QUANTITY"]
                .asfreq("D")
                .fillna(0.0)
            )
            y_val_series = (
                Xy_va[["DATE","QUANTITY"]]
                .set_index("DATE")
                .sort_index()["QUANTITY"]
                .asfreq("D")
                .fillna(0.0)
            )
            y_val = y_val_series.to_numpy()
            horizon = len(y_val)

            # Baseline 1: Naive-1
            yhat_naive = _naive_last_value(train_series, horizon)
            rows.append({
                "CUSTOMER": cust, "fold": f.fold, "anchor": f.meta["anchor"].date(), "model": "Naive-1",
                "MAE": mae(y_val, yhat_naive), "RMSE": rmse(y_val, yhat_naive), "sMAPE": smape(y_val, yhat_naive),
                "n": horizon
            })

            # Baseline 2: Seasonal-7
            yhat_s7 = _seasonal_weekly(train_series, horizon)
            rows.append({
                "CUSTOMER": cust, "fold": f.fold, "anchor": f.meta["anchor"].date(), "model": "Seasonal-7",
                "MAE": mae(y_val, yhat_s7), "RMSE": rmse(y_val, yhat_s7), "sMAPE": smape(y_val, yhat_s7),
                "n": horizon
            })

            # Baseline 3: ETS (classical, no feature matrix)
            yhat_ets = _ets_forecast(train_series, horizon=horizon, seasonal=ets_seasonality, seasonal_periods=7)
            rows.append({
                "CUSTOMER": cust, "fold": f.fold, "anchor": f.meta["anchor"].date(), "model": f"ETS-{ets_seasonality}",
                "MAE": mae(y_val, yhat_ets), "RMSE": rmse(y_val, yhat_ets), "sMAPE": smape(y_val, yhat_ets),
                "n": horizon
            })

    per_fold = pd.DataFrame(rows).sort_values(["CUSTOMER","model","fold"]).reset_index(drop=True)
    summary = (
        per_fold.dropna(subset=["fold"])  # drop NO_FOLDS
        .groupby(["CUSTOMER","model"], as_index=False)[["MAE","RMSE","sMAPE"]]
        .mean()
        .sort_values(["CUSTOMER","sMAPE"])
        .reset_index(drop=True)
    )

    if save_csv:
        per_fold.to_csv(out_path / "baseline_results.csv", index=False)
        summary.to_csv(out_path / "baseline_summary.csv", index=False)

    return per_fold, summary
