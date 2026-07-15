"""Dataset preparation and feature engineering for LightGBM training.

This module consumes the dataset exported by the NestJS ML endpoints and
produces clean, structured training matrices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import (
    DATASET_PARAMS,
    PREPROCESSING_PARAMS,
    RAW_FEATURE_COLUMNS,
    TRAINING_FEATURE_COLUMNS,
)


DEFAULT_NUMERIC_VALUES = {
    "timeOnPage": 0.0,
    "pageTime": 0.0,
    "scrollDepth": 0.0,
    "clickMisses": 0.0,
    "hesitations": 0.0,
    "abandonmentRisk": 0.0,
}

DEFAULT_BINARY_VALUES = {
    "helpTriggered": 0,
    "hasError": 0,
    "multiplePages": 0,
    "label": 0,
}


@dataclass
class PreparedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str


@dataclass
class SplitDataset:
    x_train: pd.DataFrame
    x_valid: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def load_exported_dataset(input_path: str | Path) -> pd.DataFrame:
    """Load dataset exported by backend as CSV or JSON."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "features" in payload:
            return pd.DataFrame(payload["features"])

        if isinstance(payload, list):
            return pd.DataFrame(payload)

        raise ValueError("Unsupported JSON format. Expected list or {features: [...]}.")

    raise ValueError("Unsupported file format. Use .csv or .json")


def prepare_dataset(df: pd.DataFrame) -> PreparedDataset:
    """Create a clean training dataset with backend-aligned features."""
    frame = df.copy()

    _ensure_required_columns(frame)
    _clean_types_and_ranges(frame)
    _deduplicate_sessions(frame)
    _impute_missing(frame)
    _build_derived_features(frame)

    # Keep only known training columns + label to avoid leakage/noise.
    feature_columns = [c for c in TRAINING_FEATURE_COLUMNS if c in frame.columns]
    if "label" not in frame.columns:
        raise ValueError("Column 'label' is required for supervised training.")

    min_rows = int(PREPROCESSING_PARAMS["min_rows_for_training"])
    if len(frame) < min_rows:
        raise ValueError(
            f"Not enough rows for training. Found {len(frame)}, required at least {min_rows}."
        )

    # Ensure deterministic column order.
    frame = frame[feature_columns + ["label"] + _metadata_columns(frame)]

    return PreparedDataset(frame=frame, feature_columns=feature_columns, target_column="label")


def split_dataset(prepared: PreparedDataset) -> SplitDataset:
    """Split dataset into train/validation/test with stratification when possible."""
    test_size = float(DATASET_PARAMS["test_size"])
    valid_size = float(DATASET_PARAMS["validation_size"])
    random_state = int(DATASET_PARAMS["random_state"])

    x = prepared.frame[prepared.feature_columns]
    y = prepared.frame[prepared.target_column].astype(int)

    stratify = y if _can_stratify(y) else None
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    valid_ratio_in_train_valid = valid_size / (1.0 - test_size)
    stratify_train_valid = y_train_valid if _can_stratify(y_train_valid) else None
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid,
        y_train_valid,
        test_size=valid_ratio_in_train_valid,
        random_state=random_state,
        stratify=stratify_train_valid,
    )

    return SplitDataset(
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
    )


def _ensure_required_columns(frame: pd.DataFrame) -> None:
    for col in RAW_FEATURE_COLUMNS:
        if col not in frame.columns:
            default_value = DEFAULT_NUMERIC_VALUES.get(col, DEFAULT_BINARY_VALUES.get(col, 0))
            frame[col] = default_value


def _clean_types_and_ranges(frame: pd.DataFrame) -> None:
    numeric_cols = [
        "timeOnPage",
        "pageTime",
        "scrollDepth",
        "clickMisses",
        "hesitations",
        "abandonmentRisk",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Business constraints from behavioral metrics.
    frame["timeOnPage"] = frame["timeOnPage"].clip(lower=0)
    if "pageTime" in frame.columns:
        frame["pageTime"] = frame["pageTime"].fillna(frame["timeOnPage"]).clip(lower=0)
        # page dwell cannot exceed session dwell
        frame["pageTime"] = np.minimum(frame["pageTime"], frame["timeOnPage"])
    frame["scrollDepth"] = frame["scrollDepth"].clip(lower=0, upper=100)
    frame["clickMisses"] = frame["clickMisses"].clip(lower=0)
    frame["hesitations"] = frame["hesitations"].clip(lower=0)
    frame["abandonmentRisk"] = frame["abandonmentRisk"].clip(lower=0, upper=1)

    binary_cols = ["helpTriggered", "hasError", "multiplePages", "label"]
    for col in binary_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
        frame[col] = frame[col].clip(lower=0, upper=1)


def _deduplicate_sessions(frame: pd.DataFrame) -> None:
    if "sessionId" not in frame.columns:
        return

    if "analyzedAt" in frame.columns:
        frame["analyzedAt"] = pd.to_datetime(frame["analyzedAt"], errors="coerce")
        frame.sort_values(by=["sessionId", "analyzedAt"], ascending=[True, True], inplace=True)

    frame.drop_duplicates(subset=["sessionId"], keep="last", inplace=True)


def _impute_missing(frame: pd.DataFrame) -> None:
    strategy = str(PREPROCESSING_PARAMS["missing_strategy"]).lower()

    numeric_cols = [
        "timeOnPage",
        "pageTime",
        "scrollDepth",
        "clickMisses",
        "hesitations",
        "abandonmentRisk",
    ]

    for col in numeric_cols:
        if strategy == "median":
            fill_value = frame[col].median()
        else:
            fill_value = frame[col].mean()

        if pd.isna(fill_value):
            fill_value = DEFAULT_NUMERIC_VALUES[col]

        frame[col] = frame[col].fillna(fill_value)


def _build_derived_features(frame: pd.DataFrame) -> None:
    from ml.training.derived_features import build_derived_features_frame

    build_derived_features_frame(frame)


def _metadata_columns(frame: pd.DataFrame) -> list[str]:
    metadata = []
    for col in ["sessionId", "userId", "analyzedAt"]:
        if col in frame.columns:
            metadata.append(col)
    return metadata


def _can_stratify(target: pd.Series) -> bool:
    value_counts = target.value_counts(dropna=False)
    if len(value_counts) < 2:
        return False
    return bool((value_counts >= 2).all())
