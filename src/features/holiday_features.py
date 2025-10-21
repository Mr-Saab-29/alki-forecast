from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional, Iterable
import holidays


def _make_country_holiday_series(
    idx: pd.DatetimeIndex,
    country: str = "FR",
    subdiv: Optional[str] = None,
    years_pad: int = 2,
) -> pd.Series:
    """
    Build a boolean Series indexed by idx that marks public holidays for the given country/subdivision.
    years_pad controls how many years beyond the observed range to build (for inference horizon).

    Args:
        idx (pd.DatetimeIndex): Index of dates to mark.
        country (str): Country code for holidays.
        subdiv (Optional[str]): Subdivision code for holidays.
        years_pad (int): Number of years to pad before/after the index range.
    
    Returns:
        pd.Series: Boolean Series indicating holidays.
    """
    if idx.empty:
        return pd.Series([], dtype=bool, index=idx)

    years = list(range(idx.min().year - years_pad, idx.max().year + years_pad + 1))
    cal = holidays.country_holidays(country=country, subdiv=subdiv, years=years)
    is_h = idx.normalize().isin(pd.to_datetime(list(cal.keys())))
    return pd.Series(is_h, index=idx, name="is_holiday")


def _days_to_next_prev_flag(is_flag: pd.Series) -> pd.DataFrame:
    """
    Given a boolean Series (e.g., is_holiday) indexed by daily dates,
    compute days since previous flag and days to next flag (non-negative; inf if none).

    Args:
        is_flag (pd.Series): Boolean Series indicating flag days.
    
    Returns:
        pd.DataFrame: DataFrame with 'days_since_flag' and 'days_to_next_flag' columns.
    """
    idx = is_flag.index
    # days since previous
    prev = np.where(is_flag.values, 0, np.inf)
    for i in range(1, len(prev)):
        prev[i] = 0 if is_flag.iat[i] else (prev[i-1] + 1 if prev[i-1] != np.inf else np.inf)

    # days to next
    nxt = np.where(is_flag.values, 0, np.inf)
    for i in range(len(nxt)-2, -1, -1):
        nxt[i] = 0 if is_flag.iat[i] else (nxt[i+1] + 1 if nxt[i+1] != np.inf else np.inf)

    out = pd.DataFrame(
        {
            "days_since_holiday": prev,
            "days_to_next_holiday": nxt,
        },
        index=idx,
    )
    return out


def _window_around_flag(is_flag: pd.Series, window: int = 3) -> pd.DataFrame:
    """
    Flag days in a +/- `window` around the True days in is_flag (inclusive of the holiday).
    Produces is_holiday_window_{k} for k in [1..window].

    Args:
        is_flag (pd.Series): Boolean Series indicating flag days.
        window (int): Number of days to extend the flag in both directions.
    
    Returns:
        pd.DataFrame: DataFrame with is_holiday_window_{k} columns.
    """
    idx = is_flag.index
    base = is_flag.astype(int)
    feats = {}
    # dilate the flag by k days both directions
    for k in range(1, window + 1):
        shifted_fwd = base.shift(k, fill_value=0)
        shifted_bwd = base.shift(-k, fill_value=0)
        feats[f"is_holiday_window_{k}"] = ((base + shifted_fwd + shifted_bwd) > 0).astype(int)
    return pd.DataFrame(feats, index=idx)


def make_holiday_features(
    idx: pd.DatetimeIndex,
    *,
    country: str = "FR",
    subdiv: Optional[str] = None,
    window: int = 3,
) -> pd.DataFrame:
    """
    Build holiday features for the given date index.

    Args:
        idx (pd.DatetimeIndex): Index of dates to build features for.
        country (str): Country code for holidays.
        subdiv (Optional[str]): Subdivision code for holidays.
        window (int): Window size for holiday effect smoothing.
    
    Returns:
        pd.DataFrame: DataFrame with holiday features.
    """
    is_h = _make_country_holiday_series(idx, country=country, subdiv=subdiv)
    dist = _days_to_next_prev_flag(is_h)
    win = _window_around_flag(is_h, window=window)

    # Optional holiday names (string) — useful for target encoding later
    years = list(range(idx.min().year - 2, idx.max().year + 3))
    cal = holidays.country_holidays(country=country, subdiv=subdiv, years=years)
    name_map = {pd.to_datetime(d): n for d, n in cal.items()}
    names = pd.Series(
        [name_map.get(d.normalize(), "") for d in idx],
        index=idx,
        name="holiday_name",
        dtype="string",
    )

    feats = pd.concat([is_h.rename("is_holiday").astype(int), dist, win, names], axis=1)
    return feats