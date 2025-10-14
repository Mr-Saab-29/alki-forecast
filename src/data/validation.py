from __future__ import annotations
import math
import numpy as np
import pandas as pd

def _is_integer_like(x: float) -> bool:
    """True if x is close to an integer (handles NaN and inf)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return False
    try:
        return float(x).is_integer()
    except Exception:
        return False

def _customer_gap_stats(g: pd.DataFrame) -> dict:
    # assumes DATE sorted
    dates = g["DATE"].to_numpy()
    if len(dates) <= 1:
        return {"missing_days": 0, "max_gap_days": 0}
    diffs = np.diff(dates.astype("datetime64[D]")).astype(int)
    missing = int(np.clip(diffs - 1, 0, None).sum())
    return {"missing_days": missing, "max_gap_days": int(diffs.max())}

def validate_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Return (report_df, ok)
    report_df: one row per CUSTOMER + a GLOBAL row with totals.
    Checks:
      - duplicates on (CUSTOMER, DATE)
      - NaN / negative / non-finite quantities
      - non-integer quantities (if business expects counts)
      - date coverage gaps (missing days, max gap)
      - date range per customer
    """
    # global duplicates
    dup_mask = df.duplicated(subset=["CUSTOMER", "DATE"], keep=False)
    n_dups = int(dup_mask.sum())

    rows = []
    for cust, g in df.groupby("CUSTOMER", sort=False):
        g = g.sort_values("DATE")
        # basic counts
        n_rows = len(g)
        n_na_qty = int(g["QUANTITY"].isna().sum())
        n_neg = int((g["QUANTITY"] < 0).sum())
        n_nonfinite = int(np.isinf(g["QUANTITY"].to_numpy()).sum())

        # integer-likeness (optional—comment out if decimals allowed)
        n_nonint = int((~g["QUANTITY"].dropna().apply(_is_integer_like)).sum())

        # duplicates within customer
        n_dups_c = int(g.duplicated(subset=["DATE"], keep=False).sum())

        # date coverage
        gaps = _customer_gap_stats(g)

        rows.append({
            "CUSTOMER": cust,
            "rows": n_rows,
            "start": g["DATE"].min().date() if n_rows else None,
            "end": g["DATE"].max().date() if n_rows else None,
            "duplicates_same_date": n_dups_c,
            "nan_qty": n_na_qty,
            "negative_qty": n_neg,
            "nonfinite_qty": n_nonfinite,
            "noninteger_qty": n_nonint,
            "missing_days": gaps["missing_days"],
            "max_gap_days": gaps["max_gap_days"],
        })

    rep = pd.DataFrame(rows).sort_values("CUSTOMER")

    # add a GLOBAL summary row
    global_row = {
        "CUSTOMER": "GLOBAL",
        "rows": int(len(df)),
        "start": df["DATE"].min().date() if len(df) else None,
        "end": df["DATE"].max().date() if len(df) else None,
        "duplicates_same_date": n_dups,
        "nan_qty": int(df["QUANTITY"].isna().sum()),
        "negative_qty": int((df["QUANTITY"] < 0).sum()),
        "nonfinite_qty": int(np.isinf(df["QUANTITY"].to_numpy()).sum()),
        "noninteger_qty": int((~df["QUANTITY"].dropna().apply(_is_integer_like)).sum()),
        "missing_days": int(
            rep["missing_days"].sum() if "missing_days" in rep else 0
        ),
        "max_gap_days": int(rep["max_gap_days"].max() if len(rep) else 0),
    }
    rep = pd.concat([rep, pd.DataFrame([global_row])], ignore_index=True)

    # Decide pass/fail (tune to your tolerance)
    ok = True
    if global_row["duplicates_same_date"] > 0: ok = False
    if global_row["nan_qty"] > 0: ok = False
    if global_row["negative_qty"] > 0: ok = False
    if global_row["nonfinite_qty"] > 0: ok = False
    # noninteger may be acceptable—don’t fail on it by default

    return rep, ok


def print_validation_report(rep: pd.DataFrame, ok: bool) -> None:
    cols = ["CUSTOMER","rows","start","end","duplicates_same_date",
            "nan_qty","negative_qty","nonfinite_qty","noninteger_qty",
            "missing_days","max_gap_days"]
    to_show = [c for c in cols if c in rep.columns]
    print(rep[to_show].to_string(index=False))
    print("\nValidation:", "PASS" if ok else "FAIL")