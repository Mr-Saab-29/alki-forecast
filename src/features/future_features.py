from datetime import timedelta
import pandas as pd
import numpy as np
from src.features.holiday_features import make_holiday_features

def build_future_features(
    history_df: pd.DataFrame,
    horizon: int = 30,
    max_lag: int = 30,
    roll_windows: list[int] = [7, 14, 30],
    holiday_country: str = "FR",
    holiday_subdiv_map: dict | None = None,
    holiday_window: int = 3,
) -> pd.DataFrame:
    """
    Create the same feature set for future horizon dates using the latest known history.
    Lags/rolling features are initialized from the tail of history.

    Args:
        history_df (pd.DataFrame): Historical DataFrame with 'DATE', 'CUSTOMER', and 'QUANTITY' columns.
        horizon (int): Number of future days to forecast.
        max_lag (int): Maximum lag to create.
        roll_windows (list[int]): List of window sizes for rolling means.
        holiday_country (str): Country code for holiday features.
        holiday_subdiv_map (dict | None): Mapping of CUSTOMER to holiday subdivision.
        holiday_window (int): Window size for holiday effect smoothing.
    
    Returns:
        pd.DataFrame: DataFrame with future dates and features initialized.
    """
    frames = []

    for cust, g in history_df.groupby("CUSTOMER", sort=False):
        # --- ensure proper datetime type ---
        g = g.copy()
        g["DATE"] = pd.to_datetime(g["DATE"], errors="coerce")
        g = g.dropna(subset=["DATE"]).sort_values("DATE")
        g = g.set_index("DATE")

        if g.empty:
            continue  # skip customers with no valid dates

        last_date = g.index.max()

        # --- build future index ---
        future_idx = pd.date_range(last_date + pd.Timedelta(days=1),
                                   periods=horizon, freq="D")

        # --- container ---
        f = pd.DataFrame(index=future_idx)
        f["CUSTOMER"] = cust

        # --- Calendar / deterministic features ---
        f["dayofweek"] = f.index.dayofweek
        f["month"] = f.index.month
        f["dayofmonth"] = f.index.day
        f["is_weekend"] = f["dayofweek"].isin([5, 6]).astype(int)
        f["is_month_start"] = f.index.is_month_start.astype(int)
        f["is_month_end"]   = f.index.is_month_end.astype(int)
        f["dow_sin"]   = np.sin(2 * np.pi * f["dayofweek"] / 7)
        f["dow_cos"]   = np.cos(2 * np.pi * f["dayofweek"] / 7)
        f["month_sin"] = np.sin(2 * np.pi * f["month"] / 12)
        f["month_cos"] = np.cos(2 * np.pi * f["month"] / 12)

        # --- Time index ---
        base = (g.index - g.index.min()).days
        start_t = base.max() + 1
        f["time_index"] = np.arange(start_t, start_t + horizon)
        f["time_norm"]  = f["time_index"] / (start_t + horizon)

        # --- Holiday features ---
        subdiv = holiday_subdiv_map.get(cust) if holiday_subdiv_map else None
        hfe = make_holiday_features(f.index, country=holiday_country,
                                    subdiv=subdiv, window=holiday_window)
        f = f.join(hfe.drop(columns=["holiday_name"]))

        # --- Placeholder for lag / rolling ---
        for lag in range(1, max_lag + 1):
            f[f"lag_{lag}"] = np.nan
        for w in roll_windows:
            for stat in ["mean", "std", "min", "max"]:
                f[f"roll_{stat}_{w}"] = np.nan

        frames.append(f.reset_index().rename(columns={"index": "DATE"}))

    return pd.concat(frames, ignore_index=True)