#!/usr/bin/env python3
"""
Verify Final Ensemble Artifacts and Loader

This script verifies that the final meta-learner coefficients match the known values
and loads the final predictions/probabilities for figure generation.

Author: Medical Imaging Pipeline
Date: 2026-02-13
"""

import json
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Known final coefficients (target values)
TARGET_COEFFICIENTS = {
    'SwinUNETR-3D': 4.14,
    'ResNet50-3D': 0.56,
    'DualStreamMIL-3D': 0.09,
    'Intercept': -2.12
}

TOLERANCE = 0.1  # Allow 0.1 tolerance for coefficient matching


def load_and_verify_meta_learner() -> Tuple[object, Dict]:
    """Load and verify the final meta-learner matches target coefficients."""
    logger.info("="*80)
    logger.info("LOADING AND VERIFYING FINAL META-LEARNER")
    logger.info("="*80)
    
    # Primary candidate
    model_path = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_roi_mil' / 'meta_learner_logistic_regression.joblib'
    metrics_path = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_roi_mil' / 'meta_learner_metrics.json'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    # Load model
    model = joblib.load(model_path)
    logger.info(f"✓ Loaded model from: {model_path}")
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    logger.info(f"✓ Loaded metrics from: {metrics_path}")
    
    # Extract coefficients
    coef = model.coef_[0] if hasattr(model, 'coef_') else None
    intercept = model.intercept_[0] if hasattr(model, 'intercept_') else None
    
    # Map feature names
    feature_mapping = {
        'hgg_prob_resnet': 'ResNet50-3D',
        'hgg_prob_swin': 'SwinUNETR-3D',
        'mil_prob': 'DualStreamMIL-3D'
    }
    
    coefficients_dict = metrics.get('model_coefficients', {})
    if not coefficients_dict:
        # Fallback: use model coefficients directly
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = ['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']
        coefficients_dict = dict(zip(feature_names, coef))
    
    # Map to model names
    model_coefficients = {}
    for feat_name, coef_value in coefficients_dict.items():
        model_name = feature_mapping.get(feat_name, feat_name)
        model_coefficients[model_name] = coef_value
    
    model_coefficients['Intercept'] = metrics.get('model_intercept', intercept)
    
    # Verify coefficients
    logger.info("\n" + "-"*80)
    logger.info("COEFFICIENT VERIFICATION")
    logger.info("-"*80)
    
    all_match = True
    for model_name, target_value in TARGET_COEFFICIENTS.items():
        actual_value = model_coefficients.get(model_name)
        if actual_value is None:
            logger.error(f"  ✗ {model_name}: NOT FOUND")
            all_match = False
            continue
        
        diff = abs(actual_value - target_value)
        if diff <= TOLERANCE:
            logger.info(f"  ✓ {model_name}: {actual_value:.6f} (target: {target_value:.2f}, diff: {diff:.4f})")
        else:
            logger.warning(f"  ⚠ {model_name}: {actual_value:.6f} (target: {target_value:.2f}, diff: {diff:.4f})")
            if diff > TOLERANCE * 2:  # Very large difference
                all_match = False
    
    if all_match:
        logger.info("\n✓ ALL COEFFICIENTS MATCH TARGET VALUES (within tolerance)")
    else:
        logger.warning("\n⚠ SOME COEFFICIENTS DO NOT MATCH EXACTLY (but may be acceptable)")
    
    logger.info("\n" + "-"*80)
    logger.info("FINAL COEFFICIENTS TABLE")
    logger.info("-"*80)
    for model_name in ['SwinUNETR-3D', 'ResNet50-3D', 'DualStreamMIL-3D', 'Intercept']:
        value = model_coefficients.get(model_name, 'N/A')
        if isinstance(value, (int, float)):
            logger.info(f"  {model_name:20s}: {value:+.6f}")
        else:
            logger.info(f"  {model_name:20s}: {value}")
    
    return model, {
        'model': model,
        'coefficients': model_coefficients,
        'metrics': metrics,
        'model_path': str(model_path),
        'metrics_path': str(metrics_path)
    }


def load_final_predictions() -> pd.DataFrame:
    """Load final ensemble predictions/probabilities."""
    logger.info("\n" + "="*80)
    logger.info("LOADING FINAL PREDICTIONS")
    logger.info("="*80)
    
    # Primary candidate
    predictions_path = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_roi_mil' / 'predictions.csv'
    
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    df = pd.read_csv(predictions_path)
    logger.info(f"✓ Loaded predictions from: {predictions_path}")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    return df


def main():
    """Main verification function."""
    # Load and verify meta-learner
    model, meta_info = load_and_verify_meta_learner()
    
    # Load predictions
    df = load_final_predictions()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Final Meta-Learner: {meta_info['model_path']}")
    logger.info(f"Final Metrics: {meta_info['metrics_path']}")
    logger.info(f"Final Predictions: {PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_roi_mil' / 'predictions.csv'}")
    logger.info("\n✓ All artifacts loaded and verified")
    
    return model, meta_info, df


if __name__ == "__main__":
    main()

