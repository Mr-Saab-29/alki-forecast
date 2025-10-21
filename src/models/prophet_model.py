from __future__ import annotations
import numpy as np, pandas as pd

def fit_forecast_prophet(train: pd.Series, horizon: int, *, seasonality_mode:str="multiplicative", changepoint_prior_scale:float=0.5) -> np.ndarray:
    """
    Fit Facebook Prophet on a pandas Series with a DatetimeIndex and forecast horizon steps.
    """
    from prophet import Prophet

    df = pd.DataFrame({
        "ds": train.index.to_timestamp() if isinstance(train.index, pd.PeriodIndex) else train.index,
        "y": train.values.astype(float)
    })
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    m.fit(df)
    future = pd.DataFrame({
        "ds": pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    })
    fc = m.predict(future)["yhat"].to_numpy()
    return fc.astype(float)