"""Evaluation utilities for abandonment prediction model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute production-facing binary classification metrics."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    try:
        auc_value = float(roc_auc_score(y_true_arr, y_prob))
    except ValueError:
        auc_value = None

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "auc": auc_value,
        "threshold": float(threshold),
        "support": int(len(y_true_arr)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    return metrics


def save_metrics_report(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Persist evaluation metrics to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
