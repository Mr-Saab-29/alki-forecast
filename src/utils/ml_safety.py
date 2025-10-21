from __future__ import annotations
import numpy as np
import pandas as pd

def clean_design(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=0, how="any")
    return X

def align_like(X_ref: pd.DataFrame, X_new: pd.DataFrame) -> pd.DataFrame:
    X = X_new.copy()
    for c in X_ref.columns:
        if c not in X.columns:
            X[c] = 0.0
    return X[X_ref.columns]  # drop extras + reorder

def has_enough_rows(y, min_rows: int = 20) -> bool:
    return y is not None and len(y) >= min_rows