from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict


# Helper Functions for Series Cleaning
def _sustained_bounds(
    s: pd.Series,
    *,
    min_nonzero_run: int = 7,
    min_nonzero_value: float = 1.0,
) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """
    Identify the first and last sustained activity periods in a time series.

    This function detects the first and last dates in a pandas Series where there are at least
    `min_nonzero_run` consecutive days with values greater than or equal to `min_nonzero_value`.
    It can be used to determine the sustained active period of a time series signal.

    Args:
        s : pd.Series
            Input time series with a DateTimeIndex.
        min_nonzero_run : int, optional
            Minimum number of consecutive days with value >= `min_nonzero_value`
            required to qualify as a sustained run. Default is 7.
        min_nonzero_value : float, optional
            Threshold value to consider an entry as "active". Default is 1.0.

    Returns:
        Tuple[pd.Timestamp | None, pd.Timestamp | None]
            A tuple containing:
            - `first_sustained`: The first date where a sustained run begins.
            - `last_sustained`: The last date where a sustained run ends.
            Returns `(None, None)` if no sustained run is found.
    """
    x = s.fillna(0).to_numpy()
    active = (x >= min_nonzero_value).astype(int)

    if active.sum() == 0:
        return None, None

    # rolling sum over active (consecutive run length)
    # forward
    cs = np.convolve(active, np.ones(min_nonzero_run, dtype=int), mode="valid")
    first_idx = np.argmax(cs >= min_nonzero_run) if (cs >= min_nonzero_run).any() else None
    first_ts = s.index[first_idx] if first_idx is not None else None

    # backward
    cs_back = np.convolve(active[::-1], np.ones(min_nonzero_run, dtype=int), mode="valid")
    last_pos_from_end = np.argmax(cs_back >= min_nonzero_run) if (cs_back >= min_nonzero_run).any() else None
    last_idx = len(s) - min_nonzero_run - last_pos_from_end if last_pos_from_end is not None else None
    last_ts = s.index[last_idx] if last_idx is not None else None

    return first_ts, last_ts



# Hybrid gap filler (causal-safe)
def _hybrid_fill(
        series: pd.Series,
        gap_limit: int = 7, *, 
        causal: bool = True) -> pd.Series:
    """
    Hybrid Filler for evenly spaced daily series. Assumes index is daily and monotonic
    - Time Interpolation: Small NaN gaps (<= gap_limit)
    - Set 0 for larger gaps and leading/trailing NaNs (> gap_limit)
    
    Args:
        series (pd.Series): Input time series with a DateTimeIndex.
        gap_limit (int): Maximum gap size to fill with interpolation.
        causal (bool): If True, use only past data for interpolation (forward fill then linear).
                       If False, use both past and future data (linear interpolation).
        
    Returns:
        pd.Series: Series with NaNs filled according to the hybrid strategy.
    """
    s = series.copy()

    isnan = s.isna().astype(int)
    run_id = (isnan.diff().fillna(0) == 1).cumsum() * isnan
    run_lengths = run_id.value_counts()

    # For each NaN run,  decide how to fill
    for rid, length in run_lengths.items():
        if rid == 0:
            continue  # skip non-NaN runs
        idx = s.index[run_id == rid]
        if length > gap_limit:
            s.loc[idx] = 0  # large gap or leading/trailing NaNs
        else:
            pass

    if causal:
        # forward-only (no lookahead)
        s = s.ffill(limit=gap_limit)
    else:
        # lookahead allowed for pretty EDA
        s = s.interpolate(method="linear", limit=gap_limit, limit_direction="both")
    
    return s.fillna(0)


def clean_and_truncate_series(
    series: pd.Series,
    *,
    freq: str = "D",
    gap_limit: int = 7,
    long_gap_days: int = 60,
    min_active_days: int = 30,
    causal: bool = True,
    verbose: bool = False,
    min_nonzero_run: int = 7,           
    min_nonzero_value: float = 1.0,     
) -> Tuple[pd.Series, Dict]:
    """
    Prepare a daily demand series for decomposition/modeling by aligning to daily frequency, trim it 
    according to inactivity and hybrid fill the gaps. Return cleaned series and metadata.

    Args:
        series (pd.Series): Input time series with a DateTimeIndex.
        freq (str): Frequency to reindex the series to (default 'D' for daily).
        gap_limit (int): Maximum gap size to fill with interpolation.
        long_gap_days (int): Threshold of inactivity days to trim from start/end.  # (1)
        min_active_days (int): Minimum number of active (non-zero) days required to keep the series.
        causal (bool): If True, use only past data for interpolation (forward fill then linear).
                       If False, use both past and future data (linear interpolation).
        verbose (bool): If True, print detailed processing information.
        min_nonzero_run (int): Require this many consecutive active days to declare 'start'/'end' of activity.
        min_nonzero_value (float): Values below this are treated as zero for activity detection.
    
    Returns:
        Tuple[pd.Series, Dict]: Cleaned series and metadata dictionary.
        - Cleaned daily series after truncation and gap filling.
        - Metadata dictionary containing original/cleaned lengths, date ranges, and parameters.
    """
    s_raw = series.asfreq(freq)
    orig_len = len(s_raw)
    orig_start, orig_end = s_raw.index.min(), s_raw.index.max()

    # Early exit if everything is missing
    if orig_len == 0:
        return pd.Series(dtype=float), {
            "active": False, "reason": "empty_series",
            "orig_len": 0, "orig_start": None, "orig_end": None
        }

    # Activity detection (treat NaN as 0 and ignore tiny noise)
    s_for_activity = s_raw.fillna(0)
    s_for_activity = s_for_activity.where(s_for_activity >= min_nonzero_value, 0)

    if not (s_for_activity != 0).any():
        return pd.Series(dtype=float), {
            "active": False, "reason": "no_nonzero",
            "orig_len": orig_len, "orig_start": orig_start, "orig_end": orig_end
        }

    # Sustained activity bounds
    first_sustained, last_sustained = _sustained_bounds(
        s_for_activity, 
        min_nonzero_run=min_nonzero_run,
        min_nonzero_value=min_nonzero_value,
    )

    # Fallback to first/last single non-zero if no sustained run found
    if first_sustained is None:
        first_sustained = s_for_activity.ne(0).idxmax()
    if last_sustained is None:
        last_sustained = s_for_activity[::-1].ne(0).idxmax()

    inactive_lead_days = int((first_sustained - orig_start).days)
    inactive_tail_days = int((orig_end - last_sustained).days)

    # Trim long inactive head/tail
    s_trim = s_raw
    if inactive_lead_days > long_gap_days:
        s_trim = s_raw.loc[first_sustained:]

    # Hybrid fill (causal-safe if causal=True)
    s_clean = _hybrid_fill(s_trim, gap_limit=gap_limit, causal=causal)

    active_days = int((s_clean != 0).sum())
    if active_days < min_active_days:
        return pd.Series(dtype=float), {
            "active": False, "reason": "too_short",
            "orig_len": orig_len, "clean_len": len(s_clean),
            "active_days": active_days,
            "orig_start": orig_start, "orig_end": orig_end,
            "clean_start": s_clean.index.min() if len(s_clean) else None,
            "clean_end": s_clean.index.max() if len(s_clean) else None,
            "inactive_lead_days": inactive_lead_days, "inactive_tail_days": inactive_tail_days,
            "gap_limit": gap_limit, "causal": causal,
            "min_nonzero_run": min_nonzero_run, "min_nonzero_value": min_nonzero_value,
        }

    meta = {
        "active": True,
        "orig_len": orig_len, "clean_len": len(s_clean),
        "active_days": active_days,
        "orig_start": orig_start, "orig_end": orig_end,
        "clean_start": s_clean.index.min(), "clean_end": s_clean.index.max(),
        "inactive_lead_days": inactive_lead_days, "inactive_tail_days": inactive_tail_days,
        "gap_limit": gap_limit, "causal": causal,
        "min_nonzero_run": min_nonzero_run, "min_nonzero_value": min_nonzero_value,
    }
    if verbose:
        print(f"[clean_and_truncate_series] {meta}")
    return s_clean, meta


def preprocess_all_customers(
    df: pd.DataFrame,
    *,
    gap_limit: int = 7,
    long_gap_days: int = 30,   
    min_active_days: int = 30,
    causal: bool = True,
    verbose: bool = True,
    min_nonzero_run: int = 5,
    min_nonzero_value: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Batch apply the cleaner to every customer.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'DATE', 'CUSTOMER', and 'QUANTITY' columns.
        gap_limit (int): Maximum gap size to fill with interpolation.
        long_gap_days (int): Threshold of inactivity days to trim from start/end.
        min_active_days (int): Minimum number of active (non-zero) days required to keep the series.
        causal (bool): If True, use only past data for interpolation (forward fill then linear).
                       If False, use both past and future data (linear interpolation).
        verbose (bool): If True, print detailed processing information. 
        min_nonzero_run (int): Require this many consecutive active days to declare 'start'/'end'.
        min_nonzero_value (float): Values below this are treated as zero for activity detection.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned DataFrame and metadata DataFrame
        - Cleaned DataFrame with columns ['DATE', 'CUSTOMER', 'QUANTITY'].
        - Metadata DataFrame summarizing cleaning decisions per customer
          (date ranges, inactive periods, parameters, etc.).
    """
    frames, metas = [], []
    for cust, g in df.groupby("CUSTOMER", sort=False):
        s = g.set_index("DATE")["QUANTITY"].sort_index()
        s_clean, meta = clean_and_truncate_series(
            s,
            gap_limit=gap_limit,
            long_gap_days=long_gap_days,
            min_active_days=min_active_days,
            causal=causal,
            verbose=verbose,
            min_nonzero_run=min_nonzero_run,
            min_nonzero_value=min_nonzero_value,
        )
        meta["CUSTOMER"] = cust
        metas.append(meta)
        if meta.get("active", False) and len(s_clean):
            frames.append(pd.DataFrame({"DATE": s_clean.index, "CUSTOMER": cust, "QUANTITY": s_clean.values}))
    df_clean = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["DATE","CUSTOMER","QUANTITY"])
    return df_clean, pd.DataFrame(metas)