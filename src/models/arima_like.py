from __future__ import annotations
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_forecast_sarima(
    train: pd.Series,
    horizon: int,
    *,
    p:int,d:int,q:int,P:int,D:int,Q:int, sp:int,
    exog_train: pd.DataFrame | np.ndarray | None = None,
    exog_future: pd.DataFrame | np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit and forecast using SARIMA model.
    Seasonal order is (P,D,Q,sp).
    Fallback to last value if fitting fails.

    Args:
        train: training time series
        horizon: forecast horizon
        p,d,q: ARIMA orders
        P,D,Q,sp: seasonal ARIMA orders
        exog_train: optional exogenous variables for training period
        exog_future: optional exogenous variables for forecast period
    
    Returns:
        Forecasted values as numpy array

    """
    model = SARIMAX(
        train.astype(float),
        order=(p,d,q),
        seasonal_order=(P,D,Q,sp),
        enforce_stationarity=False,
        enforce_invertibility=False,
        exog=None if exog_train is None else np.asarray(exog_train, dtype=float)
    )
    res = model.fit(disp=False)
    fc = res.forecast(
        steps=horizon,
        exog=None if exog_future is None else np.asarray(exog_future, dtype=float)
    )
    return np.asarray(fc, dtype=float)

def fit_forecast_arima(
    train: pd.Series,
    horizon: int,
    *,
    p:int,d:int,q:int,
    exog_train: pd.DataFrame | np.ndarray | None = None,
    exog_future: pd.DataFrame | np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit and forecast using ARIMA model (non-seasonal).
    Fallback to last value if fitting fails.

    Args:
        train: training time series
        horizon: forecast horizon
        p,d,q: ARIMA orders
        exog_train: optional exogenous variables for training period
        exog_future: optional exogenous variables for forecast period
    
    Returns:
        Forecasted values as numpy array
    """
    model = SARIMAX(
        train.astype(float),
        order=(p,d,q),
        seasonal_order=(0,0,0,0),
        enforce_stationarity=False,
        enforce_invertibility=False,
        exog=None if exog_train is None else np.asarray(exog_train, dtype=float)
    )
    res = model.fit(disp=False)
    fc = res.forecast(
        steps=horizon,
        exog=None if exog_future is None else np.asarray(exog_future, dtype=float)
    )
    return np.asarray(fc, dtype=float)

def fit_forecast_ets(train: pd.Series, horizon: int, *, trend:str="add", seasonal:str="add", sp:int=7) -> np.ndarray:
    """
    Fit and forecast using Holt–Winters ETS model.
    seasonal ∈ {"add","mul"}.

    Args:
        train: training time series
        horizon: forecast horizon
        trend: trend component type
        seasonal: seasonal component type
        sp: seasonal periods
    
    Returns:
        Forecasted values as numpy array
        
    """
    m = ExponentialSmoothing(
        train.astype(float),
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=sp,
        initialization_method="estimated"
    ).fit(optimized=True)
    fc = m.forecast(steps=horizon)
    return np.asarray(fc, dtype=float)
