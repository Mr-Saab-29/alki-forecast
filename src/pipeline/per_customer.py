from __future__ import annotations
import pandas as pd
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import numpy as np

from src.features.build_features import build_features
from src.utils.timeseries_split import (
    compute_min_hist, rolling_time_series_cv, select_by_index
)

@dataclass
class CustomerConfig:
    # knobs you may vary per customer based on EDA
    max_lag: int
    roll_windows: List[int]
    holiday_country: str = "FR"
    holiday_subdiv: str | None = None
    holiday_window: int = 3
    # CV
    n_folds: int = 5
    step_days: int = 7
    horizon_days: int = 25
    window_type: str = "expanding"  # or "sliding"
    train_window_days: int | None = None  # for sliding
    initial_train_days: int = 90
    min_hist_override: int | None = None  # else computed from lags/rolls

def default_customer_config_map() -> Dict[str, CustomerConfig]:
    """
    Based on EDA, return a mapping of CUSTOMER to their specific configuration.
    Adjust max_lag, roll_windows, and other parameters as needed per customer.

    Args:
        None
    Returns:
        Dict[str, CustomerConfig]: Mapping of customer name to their config.
    """
    return {
        "ARGALYS": CustomerConfig(max_lag=30, roll_windows=[7,14,30]),
        "LES MIRACULEUX": CustomerConfig(max_lag=30, roll_windows=[7,14,30]),
        "MINCI DELICE": CustomerConfig(max_lag=30, roll_windows=[7,14,30]),
        "NUTRAVANCE": CustomerConfig(max_lag=30, roll_windows=[7,14,30]),
    }

def build_features_for_customer(df_clean: pd.DataFrame, cust: str, cfg: CustomerConfig) -> pd.DataFrame:
    """"
    Build features for a single customer using their specific configuration.
    Args:
        df_clean (pd.DataFrame): Cleaned DataFrame with 'DATE', 'CUSTOMER', and 'QUANTITY' columns.
        cust (str): Customer name.
        cfg (CustomerConfig): Configuration for the customer.

    Returns:
        pd.DataFrame: DataFrame with features for the specified customer.
    
    """
    df_c = df_clean[df_clean["CUSTOMER"] == cust].copy()
    if df_c.empty:
        return df_c
    feats = build_features(
        df_c,
        max_lag=cfg.max_lag,
        roll_windows=cfg.roll_windows,
        holiday_country=cfg.holiday_country,
        holiday_subdiv_map={cust: cfg.holiday_subdiv} if cfg.holiday_subdiv else None,
        holiday_window=cfg.holiday_window,
        trim_by_history=False,    # keep rows; we’ll let min_hist handle validity
        dropna_mode="none",
    )
    return feats

def make_customer_folds(df_clean: pd.DataFrame, cust: str, cfg: CustomerConfig):
    df_c = df_clean[df_clean["CUSTOMER"] == cust].copy()
    if df_c.empty:
        return []

    min_hist = cfg.min_hist_override or compute_min_hist(cfg.max_lag, cfg.roll_windows)

    folds = rolling_time_series_cv(
        df_c,
        n_folds=cfg.n_folds,
        window_type=cfg.window_type,
        train_window_days=cfg.train_window_days,
        step_days=cfg.step_days,
        horizon_days=cfg.horizon_days,
        gap_days=0,
        by_customer=True,     # still true, though there's only one customer now
        min_hist=min_hist,
        initial_train_days=cfg.initial_train_days,
    )
    return folds

def per_customer_cv(
    df_clean: pd.DataFrame,
    model_factory: Dict[str, Callable[[], Any]],   # CUSTOMER -> model() ctor
    config_map: Dict[str, CustomerConfig] | None = None,
)-> Dict[str, Any]:
    """
    Run CV per customer, with its own feature recipe + model.
    Returns a dict of results per customer.

    Args:
        df_clean (pd.DataFrame): Cleaned DataFrame with 'DATE', 'CUSTOMER', and 'QUANTITY' columns.
        model_factory (Dict[str, Callable[[], Any]]): Mapping of CUSTOMER to model constructor functions.
        config_map (Dict[str, CustomerConfig] | None): Mapping of CUSTOMER to their config. If None, uses defaults.
    
    Returns:
        Dict[str, Any]: Results per customer, including fold metrics.
    """
    cfg_map = config_map or default_customer_config_map()
    customers = sorted(df_clean["CUSTOMER"].unique())
    results = {}

    for cust in customers:
        cfg = cfg_map.get(cust, CustomerConfig(max_lag=30, roll_windows=[7,14,30]))
        folds = make_customer_folds(df_clean, cust, cfg)
        if not folds:
            results[cust] = {"folds": 0, "metrics": None, "notes": "no folds"}
            continue

        # Build once for the whole customer, then select rows per fold (fast & leak-safe)
        feats_all = build_features_for_customer(df_clean, cust, cfg)
        if feats_all.empty:
            results[cust] = {"folds": 0, "metrics": None, "notes": "no features"}
            continue

        # enforce min_hist by dropping early rows here as well (consistent with folds)
        min_hist = cfg.min_hist_override or compute_min_hist(cfg.max_lag, cfg.roll_windows)
        feats_all = feats_all[feats_all.groupby("CUSTOMER").cumcount() >= min_hist].reset_index(drop=True)

        # choose the model class for this customer
        make_model = model_factory.get(cust, model_factory.get("__default__"))
        if make_model is None:
            raise ValueError(f"No model specified for customer {cust} and no __default__ provided.")

        fold_metrics = []
        for f in folds:
            df_c = df_clean[df_clean["CUSTOMER"] == cust].reset_index(drop=True)
            train_df = select_by_index(df_c, f.train_idx)
            val_df   = select_by_index(df_c, f.val_idx)

            # Build features on the split slices (avoid any chance of leakage)
            Xy_tr = build_features_for_customer(train_df, cust, cfg)
            Xy_va = build_features_for_customer(val_df,   cust, cfg)

            # Align columns (in case some windows missing in val start)
            drop_cols = ["DATE","CUSTOMER","QUANTITY"]
            X_tr = Xy_tr.drop(columns=drop_cols)
            X_va = Xy_va.drop(columns=drop_cols)
            X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)

            y_tr = Xy_tr["QUANTITY"].values
            y_va = Xy_va["QUANTITY"].values

            model = make_model()
            model.fit(X_tr, y_tr)
            y_hat = model.predict(X_va)

            # simple metrics (replace with your metric suite)
            mae  = float(np.mean(np.abs(y_va - y_hat)))
            rmse = float(np.sqrt(np.mean((y_va - y_hat)**2)))
            smape = float(100*np.mean(2*np.abs(y_hat - y_va) / (np.abs(y_hat)+np.abs(y_va)+1e-8)))

            fold_metrics.append({"fold": f.fold, "MAE": mae, "RMSE": rmse, "sMAPE": smape, "anchor": f.meta["anchor"]})

        res_df = pd.DataFrame(fold_metrics).sort_values("fold")
        results[cust] = {"folds": len(res_df), "metrics": res_df}

    return results
