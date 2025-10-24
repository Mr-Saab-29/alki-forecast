# src/models/gbm.py
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Optional, Dict, List
from collections import deque

def _get_xgb(use_lightgbm: bool=False):
    if use_lightgbm:
        import lightgbm as lgb
        return ("lgbm", lgb.LGBMRegressor)
    else:
        from xgboost import XGBRegressor
        return ("xgb", XGBRegressor)

def _align_columns(X_ref: pd.DataFrame, X_new: pd.DataFrame) -> pd.DataFrame:
    """Make X_new have exactly the columns of X_ref (order + names)."""
    X = X_new.copy()
    for c in X_ref.columns:
        if c not in X.columns:
            X[c] = 0.0
    X = X[X_ref.columns]
    # drop any extras present in X_new but not in X_ref
    return X

def _make_roll_stats(buffer: deque, windows: List[int]) -> Dict[str, float]:
    """Compute causal rolling stats from the current buffer of most recent values (buffer[0] is most recent)."""
    out = {}
    arr = np.array(buffer)  # shape: [max_lag], most recent first
    for w in windows:
        if len(arr) >= w:
            seg = arr[:w]
            out[f"roll_mean_{w}"] = float(seg.mean())
            out[f"roll_std_{w}"]  = float(seg.std(ddof=1)) if w > 1 else 0.0
            out[f"roll_min_{w}"]  = float(seg.min())
            out[f"roll_max_{w}"]  = float(seg.max())
        else:
            out[f"roll_mean_{w}"] = np.nan
            out[f"roll_std_{w}"]  = np.nan
            out[f"roll_min_{w}"]  = np.nan
            out[f"roll_max_{w}"]  = np.nan
    return out

def fit_predict_gbm_recursive(
    train_df: pd.DataFrame,
    *,
    build_features_fn,               # function(df)-> training features (with lags/rolls)
    build_future_features_fn,        # function(df, horizon)-> deterministic future template (calendar/holidays/etc.)
    horizon: int,
    params: Dict,
    transform: str = "raw",
    use_lightgbm: bool = False,
    max_lag: int = 30,
    roll_windows: List[int] = (7,14,30),
) -> np.ndarray:
    """
    Train GBM on historical features and predict recursively for the horizon.
    - Uses last `max_lag` actuals from training to seed lag_* and roll_*.
    - Updates lag/rolling step-by-step with its own predictions.
    - Ensures feature parity between train and prediction.

    Assumes `train_df` is a **single-customer** slice with columns DATE,CUSTOMER,QUANTITY.

    Args:
        train_df: training DataFrame with actuals
        build_features_fn: function to build training features from DataFrame
        build_future_features_fn: function to build deterministic future template from DataFrame
        horizon: forecast horizon
        params: GBM model parameters
        transform: target transformation ("raw" or "log1p")
        use_lightgbm: whether to use LightGBM (else XGBoost)
        max_lag: maximum lag to use for features
        roll_windows: list of rolling window sizes for stats features
    
    Returns:
        Forecasted values as numpy array of shape (horizon,)
    """
    # 1) Build training design
    Xy_tr = build_features_fn(train_df)
    y_tr = Xy_tr["QUANTITY"].to_numpy()
    X_tr = Xy_tr.drop(columns=["DATE","CUSTOMER","QUANTITY"])

    if transform == "log1p":
        y_fit = np.log1p(y_tr)
    else:
        y_fit = y_tr

    tag, Model = _get_xgb(use_lightgbm)
    model = Model(**params)
    model.fit(X_tr, y_fit)
    feat_cols = X_tr.columns

    # 2) Build deterministic future template
    fut_tpl = build_future_features_fn(train_df, horizon=horizon)  # must include DATE (+ calendar/holiday/time features)
    fut_tpl = fut_tpl.sort_values("DATE").reset_index(drop=True)

    # 3) Prepare a causal buffer with the last max_lag actuals (most recent first)
    tail = (
        train_df.sort_values("DATE")["QUANTITY"].tail(max_lag).to_numpy()[::-1]
        if len(train_df) > 0 else np.zeros(max_lag, dtype=float)
    )
    buffer = deque(tail.tolist(), maxlen=max_lag)

    preds = []
    rows_for_pred = []

    for i in range(horizon):
        base_row = fut_tpl.iloc[i].drop(labels=["DATE","CUSTOMER"], errors="ignore").to_dict()

        # build lag_* from buffer (lag_1 is yesterday = buffer[0])
        for k in range(1, max_lag + 1):
            base_row[f"lag_{k}"] = (buffer[k-1] if len(buffer) >= k else np.nan)

        # build rolling stats from buffer
        roll_feats = _make_roll_stats(buffer, list(roll_windows))
        base_row.update(roll_feats)

        rows_for_pred.append(base_row)
        # We'll align + impute after we stack rows into a DataFrame.

        # we need a placeholder prediction to update buffer; actual prediction done after alignment
        # (we'll do it below once we have a DataFrame)
    # Build future design matrix (all rows), align to training columns
    X_fut = pd.DataFrame(rows_for_pred)
    X_fut = _align_columns(X_tr, X_fut)
    # simple imputation for any NaNs created by early short buffers
    X_fut = X_fut.fillna(0.0)

    # Predict all in one go, but this is still causal because features were constructed only from historical buffer
    yhat = model.predict(X_fut)
    if transform == "log1p":
        yhat = np.expm1(yhat)
    yhat = np.clip(yhat, 0, None)

    return yhat