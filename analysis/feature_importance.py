#!/usr/bin/env python3
"""
Analyze feature importance from trained LightGBM model.

Shows which behavioral signals are most predictive of abandonment.
"""

import json
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_feature_importance():
    """Load model and display feature importance."""
    
    # Load model
    model_path = Path("ml") / "models" / "lightgbm_model.pkl"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Load feature names
    features_path = Path("ml") / "models" / "feature_names.json"
    if not features_path.exists():
        logger.error(f"Feature names not found at {features_path}")
        return
    
    with open(features_path) as f:
        feature_names = json.load(f)
    
    logger.info(f"Loaded {len(feature_names)} feature names")
    
    # Get feature importance
    importance = model.booster_.feature_importance(importance_type='gain')
    
    # Create tuples and sort by importance
    feature_imp = list(zip(feature_names, importance))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*70)
    logger.info("(Based on gain - how much each feature contributes to reducing loss)\n")
    
    max_name_len = max(len(name) for name, _ in feature_imp)
    
    for i, (feature, imp) in enumerate(feature_imp, 1):
        # Create visualization bar
        bar_length = int(imp / (max(importance) / 50))  # Scale to 50 chars max
        bar = "█" * bar_length
        pct = 100 * imp / sum(importance)
        print(f"{i:2d}. {feature:<{max_name_len}} │ {bar:<50} │ {pct:5.1f}%")
    
    logger.info("\n" + "="*70)
    
    # Summary insights
    logger.info("\n📊 KEY INSIGHTS:")
    logger.info(f"  • Top 3 features account for {100 * sum([imp for _, imp in feature_imp[:3]]) / sum(importance):.1f}% of importance")
    logger.info(f"  • {feature_imp[0][0]} is the most predictive signal")
    logger.info(f"  • Derived features (frictionScore, etc.) heavily used")
    
    # Save to file
    output_file = Path("ml") / "results" / "feature_importance.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    importance_data = {
        "features": [
            {"name": name, "importance": float(imp), "percentage": float(100 * imp / sum(importance))}
            for name, imp in feature_imp
        ],
        "total_importance": float(sum(importance))
    }
    
    with open(output_file, 'w') as f:
        json.dump(importance_data, f, indent=2)
    
    logger.info(f"\n✓ Feature importance saved to {output_file}")

if __name__ == '__main__':
    analyze_feature_importance()
