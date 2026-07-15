#!/usr/bin/env python3
"""Retrain abandonment model using runtime-only dependencies (no pandas/matplotlib)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import LIGHTGBM_CONFIG, TRAINING_FEATURE_COLUMNS
from ml.training.derived_features import enrich_features_dict

RAW_COLS = [
    "timeOnPage",
    "pageTime",
    "scrollDepth",
    "clickMisses",
    "hesitations",
    "helpTriggered",
    "hasError",
    "multiplePages",
]


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "features" in payload:
        return payload["features"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported dataset format")


def row_to_vector(row: dict) -> list[float]:
    raw = {col: float(row.get(col, 0) or 0) for col in RAW_COLS}
    # Legacy rows without pageTime: fall back to session dwell.
    if row.get("pageTime") is None or row.get("pageTime") == "":
        raw["pageTime"] = float(row.get("timeOnPage", 0) or 0)
    raw["multiplePages"] = int(row.get("multiplePages", 0) or 0)
    enriched = enrich_features_dict(raw)
    return [float(enriched[col]) for col in TRAINING_FEATURE_COLUMNS]


def main() -> None:
    input_path = Path("ml/data/raw/dataset_realistic_labels.json")
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])

    rows = load_rows(input_path)
    labels = [int(row.get("label", 0) or 0) for row in rows]
    x = np.array([row_to_vector(row) for row in rows], dtype=np.float64)
    y = np.array(labels, dtype=np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )

    config = {k: v for k, v in LIGHTGBM_CONFIG.items() if k not in {"early_stopping_rounds", "eval_metric"}}
    model = LGBMClassifier(**config)
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.35).astype(int)
    accuracy = float((preds == y_test).mean())
    precision = float(preds[y_test == 1].sum() / max(preds.sum(), 1))
    recall = float(preds[y_test == 1].sum() / max(y_test.sum(), 1))

    model_path = Path("ml/models/lightgbm_model.pkl")
    feature_path = Path("ml/models/feature_names.json")
    metrics_path = Path("ml/results/metrics.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    feature_path.write_text(json.dumps(TRAINING_FEATURE_COLUMNS, indent=2), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "threshold": 0.35,
                "rows": len(rows),
                "test": {
                    "accuracy": round(accuracy, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "support": int(len(y_test)),
                },
                    "note": "Retrained with pageTime feature + page-scoped timePerPage/friction",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"model": str(model_path), "metrics": str(metrics_path), "test_accuracy": accuracy}))


if __name__ == "__main__":
    main()
