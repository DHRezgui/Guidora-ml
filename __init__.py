"""
ML - Machine Learning Module
============================

LightGBM-based abandonment risk prediction model.

Usage:
    from ml.config import LIGHTGBM_CONFIG
    import lightgbm as lgb
    
    model = lgb.LGBMClassifier(**LIGHTGBM_CONFIG)
    model.fit(X_train, y_train)
"""

__version__ = "1.0.0"
__author__ = "Trustdev ML Team"

from .config import (
    LIGHTGBM_CONFIG,
    FEATURE_ENGINEERING,
    DATASET_PARAMS,
    EVALUATION_PARAMS,
    PATHS,
    RAW_FEATURE_COLUMNS,
    TRAINING_FEATURE_COLUMNS,
    PREPROCESSING_PARAMS,
)

__all__ = [
    "LIGHTGBM_CONFIG",
    "FEATURE_ENGINEERING",
    "DATASET_PARAMS",
    "EVALUATION_PARAMS",
    "PATHS",
    "RAW_FEATURE_COLUMNS",
    "TRAINING_FEATURE_COLUMNS",
    "PREPROCESSING_PARAMS",
]
