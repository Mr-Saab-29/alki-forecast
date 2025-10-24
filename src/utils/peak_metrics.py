from __future__ import annotations

import numpy as np
import pandas as pd


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.abs(y_pred - y_true)
    den = np.abs(y_true) + np.abs(y_pred)
    return float(200.0 * np.mean(num / (den + 1e-8)))


def compute_peak_metrics(detail_df: pd.DataFrame, *, peak_quantile: float = 0.9) -> pd.DataFrame:
    """Compute peak diagnostics (Peak sMAPE, Recall@TopK, under-prediction rate)."""
    if detail_df.empty:
        return pd.DataFrame(columns=[
            "CUSTOMER", "model", "peak_sMAPE", "recall_at_topk", "peak_under_rate", "k"
        ])

    df = detail_df.copy()
    thresholds = df.groupby("CUSTOMER")["y_true"].quantile(peak_quantile)
    df = df.join(thresholds.rename("peak_threshold"), on="CUSTOMER")
    df["is_peak"] = df["y_true"] >= df["peak_threshold"]

    records: list[dict] = []
    for (cust, model), grp in df.groupby(["CUSTOMER", "model"]):
        peak_grp = grp[grp["is_peak"]]
        k = int(peak_grp.shape[0])
        if k == 0:
            continue

        peak_smape = _smape(peak_grp["y_true"].to_numpy(), peak_grp["y_pred"].to_numpy())
        under_rate = float((peak_grp["y_pred"] < peak_grp["y_true"]).mean())

        actual_top_idx = grp.nlargest(k, "y_true").index
        pred_top_idx = grp.nlargest(k, "y_pred").index
        recall = len(set(actual_top_idx).intersection(pred_top_idx)) / k

        records.append({
            "CUSTOMER": cust,
            "model": model,
            "peak_sMAPE": peak_smape,
            "recall_at_topk": recall,
            "peak_under_rate": under_rate,
            "k": k,
        })

    return pd.DataFrame(records).sort_values(["CUSTOMER", "model"]).reset_index(drop=True)
