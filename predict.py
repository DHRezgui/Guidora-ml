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


class PredictionEngine:
    """Load trained model and make predictions."""

    def __init__(self, model_path: str | None = None, features_path: str | None = None):
        """Initialize prediction engine.
        
        Args:
            model_path: Path to trained LightGBM model (.pkl)
            features_path: Path to feature names JSON
        """
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

    def predict(
        self,
        features: dict[str, float],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Make abandonment prediction.
        
        Args:
            features: Feature dictionary with all required fields
            threshold: Decision boundary for binary classification
            
        Returns:
            Prediction result with risk score and classification
        """
        try:
            # Extract features in correct order
            feature_values = []
            missing_features = []
            
            for feature_name in self.feature_names:
                if feature_name not in features:
                    missing_features.append(feature_name)
                    feature_values.append(0.0)  # Default imputation
                else:
                    feature_values.append(float(features[feature_name]))
            
            if missing_features:
                print(
                    f"Warning: Missing features imputed with 0.0: {missing_features}",
                    file=sys.stderr,
                )
            
            # Reshape for single sample prediction
            X = np.array([feature_values]).reshape(1, -1)
            
            # Get probability prediction
            proba = self.model.predict_proba(X)[0]
            abandonment_risk = float(proba[1])  # Probability of class 1 (abandon)
            
            # Binary prediction based on threshold
            will_abandon = abandonment_risk >= threshold
            
            # Confidence: distance from threshold (higher = more confident)
            confidence = 1.0 - abs(abandonment_risk - threshold)
            
            return {
                "abandonmentRisk": round(abandonment_risk, 4),
                "willAbandon": bool(will_abandon),
                "confidence": round(confidence, 4),
                "threshold": threshold,
            }
            
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
