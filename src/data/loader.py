# src/data/loader.py
from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = ["DATE", "CUSTOMER", "QUANTITY"]

def load_raw(path: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file. Ensure required columns are present
    and parse dates.

    Args:
        path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with correct types.

    """
    # Read; keep_default_na=True makes 'nan','NaN','N/A' => NaN
    df = pd.read_csv(path, sep=";", dtype={"CUSTOMER": "string"})

    # Schema check
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse types
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", dayfirst=True)
    df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce")

    # 1) drop fully empty rows (often trailing garbage)
    df = df.dropna(how="all")

    # 2) drop rows missing DATE or CUSTOMER (still NA because 'string' dtype preserves NA)
    df = df.dropna(subset=["DATE", "CUSTOMER"])

    # 3) clean CUSTOMER text and remove sentinel empties
    df["CUSTOMER"] = df["CUSTOMER"].str.strip()
    df = df[~df["CUSTOMER"].str.lower().isin(["", "nan", "na", "none"])]

    # 4) final sanity: no NaT in DATE
    if df["DATE"].isna().any():
        bad = df[df["DATE"].isna()]
        # if you prefer to silently drop them, uncomment the next line and remove the error
        # df = df.dropna(subset=["DATE"])
        raise ValueError(f"Invalid dates remain after cleaning:\n{bad}")

    # Sort & return
    df = df.sort_values(["CUSTOMER", "DATE"]).reset_index(drop=True)
    return df

def reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    """ Reindex the DataFrame to ensure a continuous daily date range
    for each customer. Missing dates will have NaN quantities.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'DATE', 'CUSTOMER', and 'QUANTITY' columns.
    
    Returns:
        pd.DataFrame: Reindexed DataFrame with continuous daily dates.
    """
    out = []
    for cust, g in df.groupby("CUSTOMER", sort=False):
        full_idx = pd.date_range(g["DATE"].min(), g["DATE"].max(), freq="D")
        g = g.set_index("DATE").reindex(full_idx)
        g["CUSTOMER"] = cust
        g.index.name = "DATE"
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)[["DATE", "CUSTOMER", "QUANTITY"]]