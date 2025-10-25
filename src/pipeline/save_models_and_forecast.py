from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from prophet import Prophet
from prophet.serialize import model_to_json

from src.features.build_features import build_features
from src.features.future_features import build_future_features
from src.models.gbm import _align_columns, _make_roll_stats


@dataclass
class ForecastIntervals:
    mean: pd.Series
    p10: Optional[np.ndarray]
    p90: Optional[np.ndarray]


def _bootstrap_prediction_intervals(
    base_forecast: np.ndarray,
    residuals: np.ndarray,
    *,
    n_samples: int = 600,
    block_size: int = 7,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    if res.size == 0:
        return None, None

    rng = np.random.default_rng(42)
    horizon = len(base_forecast)
    block_size = max(1, min(block_size, res.size))
    draws = np.empty((n_samples, horizon), dtype=float)

    for i in range(n_samples):
        start = rng.integers(0, res.size)
        noise = np.empty(horizon, dtype=float)
        for h in range(horizon):
            idx = (start + h) % res.size
            noise[h] = res[idx]
        draws[i] = base_forecast + noise

    lower = np.quantile(draws, 0.10, axis=0)
    upper = np.quantile(draws, 0.90, axis=0)
    return lower, upper


def _compute_sigma(residuals: np.ndarray) -> float:
    if residuals.size == 0:
        return 1.0
    return float(np.nanstd(residuals, ddof=1) or 1.0)


def _extract_quantiles(bundle: ForecastIntervals, residuals: np.ndarray, quantile_z: float):
    p50 = bundle.mean.to_numpy()
    lower = bundle.p10
    upper = bundle.p90
    if (
        lower is not None
        and upper is not None
        and len(lower) == len(p50)
        and len(upper) == len(p50)
    ):
        p10 = np.clip(np.asarray(lower, dtype=float), 0, None)
        p90 = np.clip(np.asarray(upper, dtype=float), 0, None)
    else:
        sigma = _compute_sigma(np.asarray(residuals, dtype=float))
        p10 = np.clip(p50 - quantile_z * sigma, 0, None)
        p90 = np.clip(p50 + quantile_z * sigma, 0, None)
    return p50, p10, p90


def _fit_arima_model(
    series: pd.Series,
    params: Dict,
    transform: str,
    horizon: int,
    exog_train: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
):
    order = (
        params.get("p", 1),
        params.get("d", 0),
        params.get("q", 0),
    )
    seasonal_order = (
        params.get("P", 0),
        params.get("D", 0),
        params.get("Q", 0),
        params.get("sp", 0),
    )

    if transform == "log1p":
        endog = np.log1p(series.values)
    else:
        endog = series.values.astype(float)

    model = SARIMAX(
        endog,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    pred = results.get_forecast(steps=horizon, exog=exog_future)
    fc = pred.predicted_mean
    try:
        conf = pred.conf_int(alpha=0.2)
        if hasattr(conf, "iloc"):
            lower_arr = conf.iloc[:, 0].to_numpy()
            upper_arr = conf.iloc[:, 1].to_numpy()
        else:
            conf_np = np.asarray(conf)
            lower_arr = conf_np[:, 0]
            upper_arr = conf_np[:, 1]
    except Exception:
        lower_arr = upper_arr = None
    if transform == "log1p":
        fitted = np.expm1(results.fittedvalues)
        forecast = np.expm1(fc)
        if lower_arr is not None and upper_arr is not None:
            lower = np.expm1(lower_arr)
            upper = np.expm1(upper_arr)
        else:
            lower = upper = None
    else:
        fitted = results.fittedvalues
        forecast = fc
        lower = lower_arr
        upper = upper_arr
    residuals = series.values - fitted
    return results, ForecastIntervals(pd.Series(forecast, name="yhat"), lower, upper), residuals


def _fit_ets_model(series: pd.Series, params: Dict, transform: str, horizon: int):
    trend = params.get("trend", "add")
    seasonal = params.get("seasonal", "add")
    sp = params.get("sp", params.get("seasonal_periods", 7))

    if transform == "log1p":
        endog = np.log1p(series.values)
    else:
        endog = series.values.astype(float)

    model = ExponentialSmoothing(
        endog,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=sp,
        initialization_method="estimated",
    )
    results = model.fit(optimized=True)
    lower_arr = upper_arr = None
    try:
        pred = results.get_prediction(start=len(endog), end=len(endog) + horizon - 1)
        fc = pred.predicted_mean
        conf = pred.conf_int(alpha=0.2)
        if hasattr(conf, "iloc"):
            lower_arr = conf.iloc[:, 0].to_numpy()
            upper_arr = conf.iloc[:, 1].to_numpy()
        else:
            conf_np = np.asarray(conf)
            lower_arr = conf_np[:, 0]
            upper_arr = conf_np[:, 1]
    except Exception:
        fc = results.forecast(horizon)
    if transform == "log1p":
        fitted = np.expm1(results.fittedvalues)
        forecast = np.expm1(fc)
        if lower_arr is not None and upper_arr is not None:
            lower = np.expm1(lower_arr)
            upper = np.expm1(upper_arr)
        else:
            lower = upper = None
    else:
        fitted = results.fittedvalues
        forecast = fc
        lower = lower_arr
        upper = upper_arr
    residuals = series.values - fitted
    return results, ForecastIntervals(pd.Series(forecast, name="yhat"), lower, upper), residuals


def _fit_prophet_model(
    df: pd.DataFrame,
    transform: str,
    params: Dict,
    horizon: int,
    exog_train: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
):
    idx = pd.to_datetime(df["DATE"])
    y = df["QUANTITY"].astype(float)

    if transform == "log1p":
        y_fit = np.log1p(y)
    else:
        y_fit = y

    m = Prophet(
        seasonality_mode=params.get("seasonality_mode", "additive"),
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8,
    )

    hist = pd.DataFrame({"ds": idx, "y": y_fit})

    if exog_train is not None and not exog_train.empty:
        exog_train = exog_train.copy()
        exog_train.index = pd.to_datetime(exog_train.index)
        exog_train = exog_train.reindex(idx).fillna(0.0)
        for col in exog_train.columns:
            m.add_regressor(col)
        hist = pd.concat([hist.reset_index(drop=True), exog_train.reset_index(drop=True)], axis=1)
    m.fit(hist)
    hist_pred = m.predict(hist)
    future = m.make_future_dataframe(periods=horizon, freq="D", include_history=False)
    if exog_future is not None and not exog_future.empty:
        exog_future = exog_future.copy()
        exog_future.index = pd.to_datetime(exog_future.index)
        exog_future = exog_future.reindex(future["ds"]).fillna(0.0)
        future = pd.concat([future.reset_index(drop=True), exog_future.reset_index(drop=True)], axis=1)
    future_pred = m.predict(future)
    fc = future_pred["yhat"]
    lower_series = future_pred.get("yhat_lower")
    upper_series = future_pred.get("yhat_upper")
    if transform == "log1p":
        forecast = np.expm1(fc)
        fitted = np.expm1(hist_pred[["yhat"]]).values.flatten()
        lower = np.expm1(lower_series) if lower_series is not None else None
        upper = np.expm1(upper_series) if upper_series is not None else None
    else:
        forecast = fc
        fitted = hist_pred[["yhat"]].values.flatten()
        lower = lower_series.to_numpy() if lower_series is not None else None
        upper = upper_series.to_numpy() if upper_series is not None else None
    residuals = y.values - fitted
    lower_np = lower.to_numpy() if hasattr(lower, "to_numpy") else lower
    upper_np = upper.to_numpy() if hasattr(upper, "to_numpy") else upper
    return m, ForecastIntervals(pd.Series(forecast.values, name="yhat"), lower_np, upper_np), residuals


def _fit_gbm_model(
    train_df: pd.DataFrame,
    transform: str,
    params: Dict,
    use_lightgbm: bool,
    max_lag: int,
    roll_windows: Tuple[int, ...],
    horizon: int,
):
    from src.models.gbm import _get_xgb

    Xy_tr = build_features(
        train_df,
        max_lag=max_lag,
        roll_windows=list(roll_windows),
        holiday_country="FR",
        holiday_subdiv_map=None,
        holiday_window=3,
        trim_by_history=False,
        dropna_mode="none",
        feature_set="full",
    )
    y_tr = Xy_tr["QUANTITY"].to_numpy()
    X_tr = Xy_tr.drop(columns=["DATE", "CUSTOMER", "QUANTITY"])

    if transform == "log1p":
        y_fit = np.log1p(y_tr)
    else:
        y_fit = y_tr

    tag, Model = _get_xgb(use_lightgbm)
    model = Model(**params)
    model.fit(X_tr, y_fit)

    if transform == "log1p":
        fitted = np.expm1(model.predict(X_tr))
    else:
        fitted = model.predict(X_tr)
    residuals = y_tr - fitted

    fut_tpl = build_future_features(
        train_df,
        horizon=horizon,
        max_lag=max_lag,
        roll_windows=list(roll_windows),
        holiday_country="FR",
        holiday_subdiv_map=None,
        holiday_window=3,
        feature_set="full",
    ).sort_values("DATE").reset_index(drop=True)

    buffer = (
        train_df.sort_values("DATE")["QUANTITY"].tail(max_lag).to_numpy()[::-1]
        if len(train_df) > 0
        else np.zeros(max_lag, dtype=float)
    )
    from collections import deque

    buffer = deque(buffer.tolist(), maxlen=max_lag)
    preds: list[float] = []
    for i in range(horizon):
        base = fut_tpl.iloc[i].drop(labels=["DATE", "CUSTOMER"], errors="ignore").to_dict()
        for k in range(1, max_lag + 1):
            base[f"lag_{k}"] = buffer[k - 1] if len(buffer) >= k else np.nan
        base.update(_make_roll_stats(buffer, list(roll_windows)))

        row_df = pd.DataFrame([base])
        row_df = _align_columns(X_tr, row_df).fillna(0.0)

        step_pred = model.predict(row_df)[0]
        if transform == "log1p":
            step_pred = np.expm1(step_pred)
        step_pred = float(np.clip(step_pred, 0, None))
        preds.append(step_pred)

        if max_lag > 0:
            buffer.appendleft(step_pred)

    fut_pred = np.asarray(preds, dtype=float)
    lower, upper = _bootstrap_prediction_intervals(fut_pred, residuals)
    if lower is not None:
        lower = np.clip(lower, 0, None)
    if upper is not None:
        upper = np.clip(upper, 0, None)

    return model, ForecastIntervals(pd.Series(fut_pred, name="yhat"), lower, upper), residuals


def save_models_and_forecast(
    df_clean: pd.DataFrame,
    best_yaml_path: str | Path,
    *,
    horizon: int = 21,
    models_dir: str | Path = "outputs/models",
    forecasts_path: str | Path = "outputs/forecasts/forecast_quantiles.csv",
    quantile_z: float = 1.2815515655446004,
) -> pd.DataFrame:
    """
    Fit the best models per customer, save them to disk, and generate
    21-day forecasts with P10/P50/P90 quantiles. Wherever possible we use
    model-native predictive intervals (SARIMA/ETS/Prophet, GBM bootstrap);
    otherwise we fall back to a residual normal approximation.

    Args:
        df_clean: Cleaned historical data.
        best_yaml_path: Path to YAML file with best model specifications per customer.
        horizon: Forecast horizon in days (default: 21).
        models_dir: Directory to save the fitted models (default: "outputs/models").
        forecasts_path: Path to save the forecast quantiles CSV (default: "outputs/forecasts/forecast_quantiles.csv").
        quantile_z: Z-score for quantile calculation (default: 1.2815515655446004 for 10th/90th percentiles).
    
    Returns:
        DataFrame with forecast quantiles for all customers.
    """
    cfg = yaml.safe_load(Path(best_yaml_path).read_text())
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    forecast_records: list[dict] = []

    for cust, spec in cfg["customers"].items():
        df_c = df_clean[df_clean["CUSTOMER"] == cust].sort_values("DATE")
        if df_c.empty:
            continue

        transform = spec.get("transform", "raw")
        model_spec = spec.get("models", [])
        if not model_spec:
            continue
        model_cfg = model_spec[0]
        family = model_cfg["family"]
        params = model_cfg.get("params", {})
        use_lgbm = bool(model_cfg.get("use_lightgbm", False))

        series = df_c.set_index("DATE")["QUANTITY"].astype(float)
        last_date = pd.to_datetime(series.index.max())
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        det_train = build_features(
            df_c,
            max_lag=spec.get("max_lag", 30),
            roll_windows=list(spec.get("roll_windows", [7, 14, 30])),
            holiday_country=spec.get("holiday_country", "FR"),
            holiday_subdiv_map=None,
            holiday_window=spec.get("holiday_window", 3),
            trim_by_history=False,
            dropna_mode="none",
            feature_set="deterministic",
        )
        det_train = det_train.set_index("DATE").sort_index()
        exog_train_df = det_train.drop(columns=["CUSTOMER", "QUANTITY"], errors="ignore").reindex(series.index).fillna(0.0)
        exog_train_arr = None if exog_train_df.empty else exog_train_df.to_numpy()

        det_future = build_future_features(
            df_c,
            horizon=horizon,
            max_lag=spec.get("max_lag", 30),
            roll_windows=list(spec.get("roll_windows", [7, 14, 30])),
            holiday_country=spec.get("holiday_country", "FR"),
            holiday_subdiv_map=None,
            holiday_window=spec.get("holiday_window", 3),
            feature_set="deterministic",
        )
        det_future = det_future.set_index("DATE").sort_index()
        exog_future_df = det_future.drop(columns=["CUSTOMER"], errors="ignore").reindex(future_dates).fillna(0.0)
        exog_future_arr = None if exog_future_df.empty else exog_future_df.to_numpy()

        if family == "arima":
            model, fc_bundle, residuals = _fit_arima_model(
                series,
                params,
                transform,
                horizon,
                exog_train=exog_train_arr,
                exog_future=exog_future_arr,
            )
            model_path = models_dir / f"{cust}_ARIMA.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        elif family == "sarima":
            model, fc_bundle, residuals = _fit_arima_model(
                series,
                params,
                transform,
                horizon,
                exog_train=exog_train_arr,
                exog_future=exog_future_arr,
            )
            model_path = models_dir / f"{cust}_SARIMA.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        elif family == "ets":
            model, fc_bundle, residuals = _fit_ets_model(series, params, transform, horizon)
            model_path = models_dir / f"{cust}_ETS.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        elif family == "prophet":
            model, fc_bundle, residuals = _fit_prophet_model(
                df_c,
                transform,
                params,
                horizon,
                exog_train=exog_train_df,
                exog_future=exog_future_df,
            )
            model_path = models_dir / f"{cust}_Prophet.json"
            model_path.write_text(model_to_json(model))

        elif family == "gbm":
            model, fc_bundle, residuals = _fit_gbm_model(
                df_c,
                transform=transform,
                params=params,
                use_lightgbm=use_lgbm,
                max_lag=spec.get("max_lag", 30),
                roll_windows=tuple(spec.get("roll_windows", [7, 14, 30])),
                horizon=horizon,
            )
            model_path = models_dir / f"{cust}_GBM.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        else:
            raise ValueError(f"Unsupported family '{family}' for customer {cust}.")

        p50, p10, p90 = _extract_quantiles(fc_bundle, np.asarray(residuals, dtype=float), quantile_z)

        for date, qp50, qp10, qp90 in zip(future_dates, p50, p10, p90):
            forecast_records.append({
                "CUSTOMER": cust,
                "DATE": date,
                "model": model_cfg["name"],
                "family": family,
                "P10": float(qp10),
                "P50": float(qp50),
                "P90": float(qp90),
            })

    forecast_df = pd.DataFrame(forecast_records).sort_values(["CUSTOMER", "DATE"]).reset_index(drop=True)
    if forecasts_path:
        forecasts_path = Path(forecasts_path)
        forecasts_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(forecasts_path, index=False)

    return forecast_df
