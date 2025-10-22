from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Dict, Literal

DateLike = pd.Timestamp | str

def compute_min_hist(max_lag: int, roll_windows: List[int]) -> int:
    """Minimal rows of history needed so all lag/rolling features are defined.
    Args:
        max_lag (int): Maximum lag to create.
        roll_windows (List[int]): List of window sizes for rolling means.
    Returns:
        int: Minimum history length required."""
    longest = max(roll_windows) if roll_windows else 0
    return max(max_lag, longest)
def temporal_train_test_split(
    df: pd.DataFrame,
    *,
    date_col: str = "DATE",
    customer_col: str = "CUSTOMER",
    cutoff: Optional[DateLike] = None,
    test_days: Optional[int] = None,
    test_frac: Optional[float] = None,
    by_customer: bool = True,
    min_hist: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split.

    Choose ONE of:
      - cutoff: all rows with date <= cutoff go to train, > cutoff to test
      - test_days: last N days per customer (or globally if by_customer=False)
      - test_frac: last fraction per customer (or globally if by_customer=False)

    If min_hist is given, rows within the first `min_hist` days per customer are removed from BOTH
    splits (to ensure all lag/rolling features can exist).

    Args:
        df (pd.DataFrame): Input DataFrame with date and customer columns.
        date_col (str): Name of the date column.
        customer_col (str): Name of the customer identifier column.
        cutoff (Optional[DateLike]): Date cutoff for splitting.
        test_days (Optional[int]): Number of days for test set.
        test_frac (Optional[float]): Fraction of data for test set.
        by_customer (bool): Whether to split per customer or globally.
        min_hist (Optional[int]): Minimum history rows to exclude from start of each customer.

    Returns: 
        Tuple[pd.DataFrame, pd.DataFrame]: (train DataFrame, test DataFrame)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values([customer_col, date_col], kind="mergesort")

    if sum(x is not None for x in (cutoff, test_days, test_frac)) != 1:
        raise ValueError("Specify exactly one of {cutoff, test_days, test_frac}.")

    if cutoff is not None:
        cutoff = pd.to_datetime(cutoff)
        if by_customer:
            # same absolute cutoff for all customers
            train = df[df[date_col] <= cutoff]
            test  = df[df[date_col] >  cutoff]
        else:
            # global chronological cutoff (same as by_customer=True here)
            train = df[df[date_col] <= cutoff]
            test  = df[df[date_col] >  cutoff]

    elif test_days is not None:
        if by_customer:
            parts = []
            parts_test = []
            for cust, g in df.groupby(customer_col, sort=False):
                g = g.sort_values(date_col)
                if len(g) == 0:
                    continue
                cutoff = g[date_col].max() - pd.Timedelta(days=test_days)
                parts.append(g[g[date_col] <= cutoff])
                parts_test.append(g[g[date_col] > cutoff])
            train = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
            test  = pd.concat(parts_test, ignore_index=True) if parts_test else df.iloc[0:0]
        else:
            cutoff = df[date_col].max() - pd.Timedelta(days=test_days)
            train = df[df[date_col] <= cutoff]
            test  = df[df[date_col] >  cutoff]

    else:  # test_frac
        if not (0 < test_frac < 1):
            raise ValueError("test_frac must be in (0, 1).")
        if by_customer:
            parts = []
            parts_test = []
            for cust, g in df.groupby(customer_col, sort=False):
                n = len(g)
                n_test = max(1, int(np.floor(n * test_frac)))
                parts.append(g.iloc[: n - n_test])
                parts_test.append(g.iloc[n - n_test :])
            train = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
            test  = pd.concat(parts_test, ignore_index=True) if parts_test else df.iloc[0:0]
        else:
            n = len(df)
            n_test = max(1, int(np.floor(n * test_frac)))
            train = df.iloc[: n - n_test]
            test  = df.iloc[n - n_test :]

    if min_hist is not None and min_hist > 0:
        # mask out first min_hist rows per customer from both splits
        def _mask(group: pd.DataFrame) -> pd.Series:
            return group.groupby(customer_col).cumcount() >= min_hist

        train = train[_mask(train)].copy()
        test  = test[_mask(test)].copy()

    return train.reset_index(drop=True), test.reset_index(drop=True)

@dataclass
class Fold:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    meta: Dict


def rolling_time_series_cv(
    df: pd.DataFrame,
    *,
    date_col: str = "DATE",
    customer_col: str = "CUSTOMER",
    n_folds: int = 3,
    window_type: Literal["expanding", "sliding"] = "expanding",
    train_window_days: Optional[int] = None,  # required for sliding
    step_days: int = 30,
    horizon_days: int = 30,
    gap_days: int = 0,               # gap between train end and val start
    by_customer: bool = True,
    min_hist: Optional[int] = None,  # enforce full-feature availability
    initial_train_days: Optional[int] = 90,  # warm-up length before first fold
) -> List[Fold]:
    """
    Build rolling CV folds.

    - expanding:
        fold k uses train: from first date .. anchor_k, then val: (anchor_k+gap, anchor_k+gap+horizon]
    - sliding:
        fold k uses a fixed-length train window of `train_window_days` ending at anchor_k

    Anchors advance by `step_days`. If by_customer=True, folds are created independently per
    customer (no leakage). Returned indices are positions in the ORIGINAL df (not per-group).

    Args:
        df (pd.DataFrame): Input DataFrame with date and customer columns.
        date_col (str): Name of the date column.
        customer_col (str): Name of the customer identifier column.
        n_folds (int): Number of folds to create.
        window_type (Literal): "expanding" or "sliding".
        train_window_days (Optional[int]): Length of training window for sliding type.
        step_days (int): Days to move anchor between folds.
        horizon_days (int): Length of validation horizon.
        gap_days (int): Gap days between train end and val start.
        by_customer (bool): Whether to create folds per customer or globally.
        min_hist (Optional[int]): Minimum history rows to exclude from start of each customer.
        initial_train_days (Optional[int]): Warm-up period (in days) before the first anchor.

    Returns: List[Fold] with (train_idx, val_idx, meta={cutoffs...})
    """
    if window_type == "sliding" and not train_window_days:
        raise ValueError("train_window_days must be provided for sliding windows.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values([customer_col, date_col], kind="mergesort")
    df = df.reset_index(drop=True)

    df["_hist_len"] = df.groupby(customer_col).cumcount()

    folds: List[Fold] = []

    # Helper to build per-customer indices for one anchor
    def _per_customer_indices(anchor: pd.Timestamp) -> Tuple[List[int], List[int]]:
        train_idx_list: List[int] = []
        val_idx_list: List[int] = []

        for cust, g in df.groupby(customer_col, sort=False):
            g = g.sort_values(date_col)
            # determine train window for this customer
            if window_type == "expanding":
                train_start = g[date_col].min()
            else:  # sliding
                train_start = anchor - pd.Timedelta(days=train_window_days)

            train_end = anchor

            # validation window
            val_start = anchor + pd.Timedelta(days=gap_days) + pd.Timedelta(days=1)
            val_end   = val_start + pd.Timedelta(days=horizon_days - 1)

            g_train = g[(g[date_col] >= train_start) & (g[date_col] <= train_end)]
            g_val   = g[(g[date_col] >= val_start)   & (g[date_col] <= val_end)]

            if min_hist is not None and min_hist > 0:
                g_train = g_train[df.loc[g_train.index, "_hist_len"] >= min_hist]
                g_val   = g_val[df.loc[g_val.index,   "_hist_len"] >= min_hist]

            train_idx_list.extend(g_train.index.tolist())
            val_idx_list.extend(g_val.index.tolist())

        return train_idx_list, val_idx_list

    # define anchor range
    global_start = df[date_col].min()
    global_end   = df[date_col].max()

    if initial_train_days is None:
        init_days = max(0, (min_hist or 0))
    else:
        init_days = max(0, int(initial_train_days))
        if min_hist is not None:
            init_days = max(init_days, int(min_hist))

    anchor = global_start + pd.Timedelta(days=init_days)
    latest_anchor = global_end - pd.Timedelta(days=gap_days + horizon_days)
    if anchor > latest_anchor:
        anchor = latest_anchor
    if anchor < global_start:
        anchor = global_start

    k = 1
    while anchor + pd.Timedelta(days=gap_days + horizon_days) <= global_end and k <= n_folds:
        tr_idx, va_idx = _per_customer_indices(anchor)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            anchor += pd.Timedelta(days=step_days)
            continue

        folds.append(Fold(
            fold=k,
            train_idx=np.array(tr_idx, dtype=int),
            val_idx=np.array(va_idx, dtype=int),
            meta={
                "anchor": anchor,
                "window_type": window_type,
                "train_window_days": train_window_days,
                "gap_days": gap_days,
                "horizon_days": horizon_days,
                "step_days": step_days,
            }
        ))
        anchor += pd.Timedelta(days=step_days)
        k += 1

    return folds

def select_by_index(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """Return df rows at integer positions `idx` (keeps original order).
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        idx (np.ndarray): Integer positions to select.  
    Returns:
        pd.DataFrame: Selected rows as a new DataFrame.
    """
    return df.iloc[idx].copy().reset_index(drop=True)


def add_split_flag(df: pd.DataFrame, idx_train: np.ndarray, idx_val: np.ndarray, col: str = "split") -> pd.DataFrame:
    """Annotate a DataFrame with a split flag: 'train'/'val' and return a copy.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        idx_train (np.ndarray): Integer positions for training set.
        idx_val (np.ndarray): Integer positions for validation set.
        col (str): Name of the column to add.  
    Returns:
        pd.DataFrame: DataFrame with added split flag column.
    """
    out = df.copy()
    out[col] = "unused"
    out.loc[idx_train, col] = "train"
    out.loc[idx_val, col] = "val"
    return out
