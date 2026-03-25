"""Train LightGBM model for abandonment prediction.

Examples:
    python ml/training/train_model.py --input ml/data/raw/dataset.json
    python ml/training/train_model.py --input ml/data/raw/dataset.csv --threshold 0.45
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import EVALUATION_PARAMS, LIGHTGBM_CONFIG, PATHS
from ml.training.evaluate import compute_classification_metrics, save_metrics_report
from ml.training.preprocess import load_exported_dataset, prepare_dataset, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM abandonment model")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to backend exported dataset (.json or .csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(EVALUATION_PARAMS.get("threshold", 0.5)),
        help="Decision threshold for binary metrics",
    )
    parser.add_argument(
        "--model-output",
        default=str(Path("ml") / "models" / "lightgbm_model.pkl"),
        help="Model output path",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(Path("ml") / "results" / "metrics.json"),
        help="Metrics report output path",
    )
    parser.add_argument(
        "--feature-output",
        default=str(Path("ml") / "models" / "feature_names.json"),
        help="Feature names output path",
    )
    parser.add_argument(
        "--prepared-output",
        default=str(Path("ml") / "data" / "processed" / "dataset_prepared.csv"),
        help="Prepared dataset output path",
    )
    return parser.parse_args()


def _prepare_directories() -> None:
    for key in ["models", "results", "data"]:
        Path(PATHS[key]).mkdir(parents=True, exist_ok=True)
    Path("ml/data/processed").mkdir(parents=True, exist_ok=True)


def _safe_training_config() -> dict:
    # Keep only constructor-safe keys for LGBMClassifier.
    disallowed = {"early_stopping_rounds", "eval_metric"}
    return {k: v for k, v in LIGHTGBM_CONFIG.items() if k not in disallowed}


def main() -> None:
    args = parse_args()
    _prepare_directories()

    raw_df = load_exported_dataset(args.input)
    prepared = prepare_dataset(raw_df)
    splits = split_dataset(prepared)

    model = LGBMClassifier(**_safe_training_config())
    model.fit(
        splits.x_train,
        splits.y_train,
        eval_set=[(splits.x_valid, splits.y_valid)],
        eval_metric="auc",
    )

    valid_prob = model.predict_proba(splits.x_valid)[:, 1]
    test_prob = model.predict_proba(splits.x_test)[:, 1]

    valid_metrics = compute_classification_metrics(
        splits.y_valid,
        valid_prob,
        threshold=args.threshold,
    )
    test_metrics = compute_classification_metrics(
        splits.y_test,
        test_prob,
        threshold=args.threshold,
    )

    metrics_payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": "LightGBM",
        "features": prepared.feature_columns,
        "dataset": {
            "rows": len(prepared.frame),
            "train_rows": len(splits.x_train),
            "valid_rows": len(splits.x_valid),
            "test_rows": len(splits.x_test),
        },
        "validation": valid_metrics,
        "test": test_metrics,
    }

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    save_metrics_report(metrics_payload, args.metrics_output)

    feature_path = Path(args.feature_output)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_path.open("w", encoding="utf-8") as f:
        json.dump(prepared.feature_columns, f, indent=2)

    prepared_output_path = Path(args.prepared_output)
    prepared_output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.frame.to_csv(prepared_output_path, index=False)

    print("Model training completed")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {args.metrics_output}")
    print(
        "Test metrics: "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"auc={test_metrics['auc']:.4f}"
    )


if __name__ == "__main__":
    main()
