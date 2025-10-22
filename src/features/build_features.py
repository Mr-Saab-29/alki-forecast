from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from src.features.holiday_features import make_holiday_features

def build_features(
    df: pd.DataFrame,
    max_lag: int = 30,
    roll_windows: list[int] = [7, 14, 30],
    holiday_country: str = "FR",
    holiday_subdiv_map: Optional[Dict[str, str]] = None,
    holiday_window: int = 3,
    trim_by_history: bool = True,
    dropna_mode: str = "none", # options: "none", "features_only"
    feature_set: str = "full",  # "full" | "deterministic"
) -> pd.DataFrame:
    """ 
    Create time-series features capturing seasonality, recency, trend, business cycles, and holidays.
    Use feature_set="deterministic" to emit only calendar/time/holiday determinants (no target-derived lags).

    Args:
        df (pd.DataFrame): Input DataFrame with 'DATE', 'CUSTOMER', and 'QUANTITY' columns.
        max_lag (int): Maximum lag to create.
        roll_windows (list[int]): List of window sizes for rolling means.
        holiday_country (str): Country code for holiday features.
        holiday_subdiv_map (Optional[Dict[str, str]]): Mapping of CUSTOMER to holiday subdivision.
        holiday_window (int): Window size for holiday effect smoothing.
    
    Returns:
        pd.DataFrame: DataFrame with added lag and rolling mean features.
    """

    req_cols = {"DATE", "CUSTOMER", "QUANTITY"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"build_features: missing columns {missing}")

    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values(["CUSTOMER", "DATE"], kind="mergesort")

    feature_set = feature_set.lower()
    if feature_set not in {"full", "deterministic"}:
        raise ValueError(f"Unsupported feature_set '{feature_set}' (expected 'full' or 'deterministic').")

    include_lags = feature_set == "full"
    include_rolls = feature_set == "full"
    include_trend = feature_set == "full"

    frames: List[pd.DataFrame] = []
    longest_w = max(roll_windows) if (include_rolls and roll_windows) else 0
    min_hist = max(max_lag if include_lags else 0, longest_w)

    for cust, g in df.groupby("CUSTOMER", sort=False):
        # if your preprocess guaranteed daily contiguity this reindex is a no-op,
        # keep for safety (won't hurt); remove if you prefer strict 1:1 rows.
        g = g.sort_values("DATE").set_index("DATE")
        start, end = g.index.min(), g.index.max()
        idx = pd.date_range(start, end, freq="D")
        g = g.reindex(idx)
        g["CUSTOMER"] = cust

        if include_lags:
            # --- Lags from original cleaned signal (no fills) ---
            q_orig = g["QUANTITY"].astype(float)
            for lag in range(1, max_lag + 1):
                g[f"lag_{lag}"] = q_orig.shift(lag)

        if include_rolls:
            # --- Causal rolling stats (reference up to t-1) ---
            ref = g["QUANTITY"].shift(1)
            for w in roll_windows:
                g[f"roll_mean_{w}"] = ref.rolling(w, min_periods=w).mean()
                g[f"roll_std_{w}"]  = ref.rolling(w, min_periods=w).std()
                g[f"roll_min_{w}"]  = ref.rolling(w, min_periods=w).min()
                g[f"roll_max_{w}"]  = ref.rolling(w, min_periods=w).max()
        else:
            ref = g["QUANTITY"].shift(1)

        if include_trend:
            # --- Causal trend (NO leakage) ---
            g["diff_7"]  = ref.diff(6)   # (t-1) - (t-7) == ref - ref.shift(6)
            g["slope_7"] = ref.rolling(7, min_periods=7).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
            )

        # --- Calendar features ---
        g["dayofweek"]     = g.index.dayofweek
        g["month"]         = g.index.month
        g["dayofmonth"]    = g.index.day
        g["is_weekend"]    = g["dayofweek"].isin([5, 6]).astype(int)
        g["is_month_start"]= g.index.is_month_start.astype(int)
        g["is_month_end"]  = g.index.is_month_end.astype(int)

        # --- Cyclic encodings ---
        g["dow_sin"]   = np.sin(2 * np.pi * g["dayofweek"] / 7)
        g["dow_cos"]   = np.cos(2 * np.pi * g["dayofweek"] / 7)
        g["month_sin"] = np.sin(2 * np.pi * g["month"] / 12)
        g["month_cos"] = np.cos(2 * np.pi * g["month"] / 12)

        # --- Time index (per-customer) ---
        g["time_index"] = (g.index - g.index.min()).days
        denom = max(1, g["time_index"].max())
        g["time_norm"]  = g["time_index"] / denom

        # --- Holidays (safe, known in advance) ---
        subdiv = holiday_subdiv_map.get(cust) if holiday_subdiv_map else None
        hfe = make_holiday_features(g.index, country=holiday_country, subdiv=subdiv, window=holiday_window)
        g = g.join(hfe.drop(columns=["holiday_name"]))

        # --- Per-customer trim so all lags/rolls valid ---
        if trim_by_history and len(g) > min_hist:
            g = g.iloc[min_hist:].copy()

        # Optional targeted dropna (only on features we generated)
        if dropna_mode == "features_only":
            feat_cols = [c for c in g.columns if c.startswith(("lag_","roll_","diff_","slope_",
                                                               "dayof","month","is_","dow_","time_","is_holiday"))]
            g = g.dropna(subset=feat_cols)

        g = g.reset_index().rename(columns={"index": "DATE"})
        frames.append(g)

    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Consistent column order (target first)
    first_cols = ["DATE", "CUSTOMER", "QUANTITY"]
    other_cols = [c for c in features.columns if c not in first_cols]
    return features[first_cols + other_cols]
