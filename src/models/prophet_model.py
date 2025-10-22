from __future__ import annotations
import numpy as np, pandas as pd

def fit_forecast_prophet(
    train: pd.Series,
    horizon: int,
    *,
    seasonality_mode: str = "multiplicative",
    changepoint_prior_scale: float = 0.5,
    exog_train: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Fit Facebook Prophet on a pandas Series with a DatetimeIndex and forecast horizon steps.
    """
    from prophet import Prophet

    idx = train.index.to_timestamp() if isinstance(train.index, pd.PeriodIndex) else pd.to_datetime(train.index)
    df = pd.DataFrame({"ds": idx, "y": train.values.astype(float)})

    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
    )

    exog_cols: list[str] = []
    if exog_train is not None:
        exog_train = exog_train.copy()
        exog_train.index = pd.to_datetime(exog_train.index)
        exog_train = exog_train.sort_index()
        exog_cols = list(exog_train.columns)
        for col in exog_cols:
            m.add_regressor(col)
        aligned = exog_train.reindex(df["ds"]).fillna(method="ffill").fillna(0.0)
        df = pd.concat([df.reset_index(drop=True), aligned.reset_index(drop=True)], axis=1)

    m.fit(df)

    future = pd.DataFrame({
        "ds": pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    })
    if exog_future is not None and exog_cols:
        exog_future = exog_future.copy()
        exog_future.index = pd.to_datetime(exog_future.index)
        aligned_future = exog_future.reindex(future["ds"]).fillna(method="ffill").fillna(0.0)
        aligned_future = aligned_future.reindex(columns=exog_cols, fill_value=0.0)
        future = pd.concat([future.reset_index(drop=True), aligned_future.reset_index(drop=True)], axis=1)

    fc = m.predict(future)["yhat"].to_numpy()
    return fc.astype(float)
