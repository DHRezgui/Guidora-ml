"""Real-time abandonment prediction using trained LightGBM model.

This script loads the trained model and generates abandonment risk predictions
for incoming user behavior features.

Usage:
    python ml/predict.py --input features.json
    
Input JSON format:
    {
        "features": {
            "timeOnPage": 45.5,
            "scrollDepth": 50,
            "clickMisses": 2,
            "hesitations": 1,
            "abandonmentRisk": 0.3,
            "helpTriggered": 0,
            "hasError": 0,
            "multiplePages": 1,
            "timePerPage": 22.75,
            "clickMissRate": 0.4,
            "hesitationRate": 0.2,
            "frictionScore": 0.35,
            "highFriction": 0,
            "multipleIssues": 0
        }
    }

Output JSON format:
    {
        "success": true,
        "prediction": {
            "abandonmentRisk": 0.325,
            "willAbandon": false,
            "confidence": 0.95,
            "threshold": 0.5
        }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import PATHS, TRAINING_FEATURE_COLUMNS
from ml.training.derived_features import enrich_features_dict


def compute_abandonment_confidence(abandonment_risk: float, threshold: float) -> float:
    """Confidence as normalized distance from the decision threshold."""
    denom = max(threshold, 1.0 - threshold)
    if denom <= 0:
        return 0.0
    return abs(abandonment_risk - threshold) / denom


def log_odds_to_probability(log_odds: float) -> float:
    """Convert SHAP / LightGBM log-odds baseline to a 0-1 probability."""
    if not np.isfinite(log_odds):
        return 0.0
    if log_odds >= 20:
        return 1.0
    if log_odds <= -20:
        return 0.0
    return float(1.0 / (1.0 + np.exp(-log_odds)))


class PredictionEngine:
    """Load trained model and make predictions."""

    def __init__(self, model_path: str | None = None, features_path: str | None = None):
        # Get ml directory path
        ml_dir = Path(__file__).resolve().parent
        
        # Use provided paths or defaults
        if model_path is None:
            self.model_path = ml_dir / "models" / "lightgbm_model.pkl"
        else:
            self.model_path = Path(model_path)
            
        if features_path is None:
            self.features_path = ml_dir / "models" / "feature_names.json"
        else:
            self.features_path = Path(features_path)
        
        self.model = None
        self.feature_names = None
        self._shap_explainer = None
        self.load_model()

    def load_model(self) -> None:
        """Load trained model and feature names from disk."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.model = joblib.load(str(self.model_path))
            
            if self.features_path.exists():
                with open(self.features_path, 'r') as f:
                    self.feature_names = json.load(f)
            else:
                # Fallback to config feature names
                self.feature_names = TRAINING_FEATURE_COLUMNS
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _prepare_feature_matrix(
        self,
        features: dict[str, float],
    ) -> tuple[np.ndarray, list[float]]:
        feature_values: list[float] = []
        missing_features: list[str] = []

        for feature_name in self.feature_names:
            if feature_name not in features:
                missing_features.append(feature_name)
                feature_values.append(0.0)
            else:
                feature_values.append(float(features[feature_name]))

        if missing_features:
            print(
                f"Warning: Missing features imputed with 0.0: {missing_features}",
                file=sys.stderr,
            )

        X = np.array([feature_values]).reshape(1, -1)
        return X, feature_values

    def _compute_shap_explanation(
        self,
        X: np.ndarray,
        feature_values: list[float],
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        import shap

        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)

        import pandas as pd

        X_df = pd.DataFrame(X, columns=self.feature_names)
        shap_values = self._shap_explainer.shap_values(X_df)
        if isinstance(shap_values, list):
            contributions = np.asarray(shap_values[1][0], dtype=float)
            expected_value = float(np.asarray(self._shap_explainer.expected_value)[1])
        else:
            contributions = np.asarray(shap_values[0], dtype=float)
            expected_raw = self._shap_explainer.expected_value
            expected_value = float(np.asarray(expected_raw).reshape(-1)[0])

        rows = [
            {
                "feature": feature_name,
                "contribution": round(float(contribution), 4),
                "value": round(float(value), 4),
            }
            for feature_name, contribution, value in zip(
                self.feature_names,
                contributions,
                feature_values,
            )
        ]
        rows.sort(key=lambda row: abs(row["contribution"]), reverse=True)

        return {
            "topFeatures": rows[:top_k],
            "expectedValue": round(expected_value, 4),
            "baseProbability": round(log_odds_to_probability(expected_value), 4),
        }

    def predict(
        self,
        features: dict[str, float],
        threshold: float = 0.5,
        *,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Make abandonment prediction.
        
        Args:
            features: Feature dictionary with all required fields
            threshold: Decision boundary for binary classification
            explain: When true, attach SHAP top-feature contributions
            
        Returns:
            Prediction result with risk score and classification
        """
        try:
            features = enrich_features_dict(features)
            X, feature_values = self._prepare_feature_matrix(features)
            
            # Get probability prediction
            proba = self.model.predict_proba(X)[0]
            abandonment_risk = float(proba[1])  # Probability of class 1 (abandon)
            
            # Binary prediction based on threshold
            will_abandon = abandonment_risk >= threshold
            
            # Confidence: certainty based on distance from the decision threshold.
            confidence = compute_abandonment_confidence(abandonment_risk, threshold)
            
            result: dict[str, Any] = {
                "abandonmentRisk": round(abandonment_risk, 4),
                "willAbandon": bool(will_abandon),
                "confidence": round(confidence, 4),
                "threshold": threshold,
            }

            if explain:
                result["explanation"] = self._compute_shap_explanation(X, feature_values)

            return result
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    def predict_batch(
        self,
        features_list: list[dict[str, float]],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Make batch predictions.
        
        Args:
            features_list: List of feature dictionaries
            threshold: Decision boundary
            
        Returns:
            List of prediction results
        """
        results = []
        for features in features_list:
            results.append(self.predict(features, threshold))
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time abandonment prediction")
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON file with features",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary classification (default: 0.5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (default: ml/models/lightgbm_model.pkl)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to feature names JSON (default: ml/models/feature_names.json)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Initialize engine
        engine = PredictionEngine(model_path=args.model, features_path=args.features)
        
        # Load input features
        if args.input:
            with open(args.input, 'r') as f:
                data = json.load(f)
                features = data.get("features", data)
        else:
            # Read from stdin for API calls
            input_data = sys.stdin.read()
            data = json.loads(input_data)
            features = data.get("features", data)
        
        # Make prediction
        prediction = engine.predict(features, threshold=args.threshold)
        
        # Output result
        result = {
            "success": True,
            "prediction": prediction,
        }
        
        print(json.dumps(result))
        sys.exit(0)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
        }
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
