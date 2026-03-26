#!/usr/bin/env python3
"""
Analyze and optimize decision threshold for abandonment prediction.

Different thresholds trade-off between sensitivity and specificity.
This helps find the optimal threshold for your use case.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
import pandas as pd
import logging
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_thresholds():
    """Test different decision thresholds and find optimal one."""
    
    # Load model and test data
    model_path = Path("ml") / "models" / "lightgbm_model.pkl"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    model = joblib.load(model_path)
    logger.info(f"Loaded model")
    
    # Load metrics to get test set info
    metrics_path = Path("ml") / "results" / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Reload the test set to compute predictions
    from ml.training.preprocess import load_exported_dataset, prepare_dataset, split_dataset
    
    logger.info("Loading and preparing dataset...")
    df = load_exported_dataset("ml/data/raw/dataset_realistic_labels.json")
    prepared = prepare_dataset(df)
    split = split_dataset(prepared)
    
    # Get test set predictions (probabilities)
    y_pred_proba = model.predict_proba(split.x_test)[:, 1]  # Probability of class 1 (abandoned)
    y_true = split.y_test.values
    
    # Test different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    threshold_results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        threshold_results.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })
    
    # Display results
    logger.info("\n" + "="*120)
    logger.info("THRESHOLD OPTIMIZATION ANALYSIS")
    logger.info("="*120)
    
    df_results = pd.DataFrame(threshold_results)
    print(df_results.to_string(index=False))
    
    logger.info("\n" + "="*120)
    
    # Find optimal thresholds for different use cases
    logger.info("\n🎯 RECOMMENDED THRESHOLDS:\n")
    
    # Best F1 (balanced)
    best_f1_idx = df_results['f1'].idxmax()
    best_f1 = df_results.loc[best_f1_idx]
    logger.info(f"✓ BALANCED (F1 Score):")
    logger.info(f"  Threshold: {best_f1['threshold']:.2f}")
    logger.info(f"  → Accuracy: {best_f1['accuracy']:.1%}, Recall: {best_f1['recall']:.1%}, Precision: {best_f1['precision']:.1%}")
    logger.info(f"  → Use when: You want equal balance between false positives and false negatives")
    
    # Maximum recall (catch everything)
    max_recall_idx = df_results['recall'].idxmax()
    max_recall = df_results.loc[max_recall_idx]
    logger.info(f"\n✓ MAXIMUM SENSITIVITY (Catch all abandons):")
    logger.info(f"  Threshold: {max_recall['threshold']:.2f}")
    logger.info(f"  → Accuracy: {max_recall['accuracy']:.1%}, Recall: {max_recall['recall']:.1%}, Precision: {max_recall['precision']:.1%}")
    logger.info(f"  → Use when: You MUST catch every potential abandonment (cost of missing > cost of false alert)")
    
    # High precision (few false positives) - if available
    high_prec_candidates = df_results[df_results['precision'] >= 0.7]
    if len(high_prec_candidates) > 0:
        high_precision_idx = high_prec_candidates['f1'].idxmax()
        high_precision = df_results.loc[high_precision_idx]
        logger.info(f"\n✓ HIGH PRECISION (Few false alarms):")
        logger.info(f"  Threshold: {high_precision['threshold']:.2f}")
        logger.info(f"  → Accuracy: {high_precision['accuracy']:.1%}, Recall: {high_precision['recall']:.1%}, Precision: {high_precision['precision']:.1%}")
        logger.info(f"  → Use when: Alert fatigue is a problem (only alert when very confident)")
    else:
        logger.info(f"\n⚠ HIGH PRECISION (Few false alarms):")
        logger.info(f"  Not available with this dataset")
        logger.info(f"  (No threshold achieves >= 70% precision)")
        logger.info(f"  → Consider: This model is better at recall than precision (good for risk detection)")

    
    # Current threshold
    current_idx = (df_results['threshold'] - 0.5).abs().idxmin()
    current = df_results.loc[current_idx]
    logger.info(f"\n✓ CURRENT THRESHOLD ({current['threshold']:.2f}):")
    logger.info(f"  → Accuracy: {current['accuracy']:.1%}, Recall: {current['recall']:.1%}, Precision: {current['precision']:.1%}")
    
    logger.info("\n" + "="*120)
    
    # Save results
    output_file = Path("ml") / "results" / "threshold_analysis.json"
    
    recommendations = {
        "balanced": {
            "threshold": float(best_f1['threshold']),
            "f1": float(best_f1['f1'])
        },
        "maximum_sensitivity": {
            "threshold": float(max_recall['threshold']),
            "recall": float(max_recall['recall'])
        }
    }
    
    if len(high_prec_candidates) > 0:
        recommendations["high_precision"] = {
            "threshold": float(high_precision['threshold']),
            "precision": float(high_precision['precision'])
        }
    
    with open(output_file, 'w') as f:
        json.dump({
            "analysis": threshold_results,
            "recommendations": recommendations
        }, f, indent=2)
    
    logger.info(f"\n✓ Threshold analysis saved to {output_file}")

if __name__ == '__main__':
    analyze_thresholds()
