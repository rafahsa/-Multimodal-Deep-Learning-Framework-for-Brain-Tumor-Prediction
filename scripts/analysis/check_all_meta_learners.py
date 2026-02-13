#!/usr/bin/env python3
"""
Check All Meta-Learner Files and Their Coefficients

This script loads all meta_learner*.joblib files and extracts their coefficients
to identify which one (if any) has DualStreamMIL-3D coefficient around 0.09.
"""

import joblib
import json
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Find all meta-learner joblib files
meta_learner_files = [
    PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_roi_mil' / 'meta_learner_logistic_regression.joblib',
    PROJECT_ROOT / 'ensemble' / 'models' / 'augmented' / 'meta_learner_logistic_regression_augmented.joblib',
    PROJECT_ROOT / 'ensemble' / 'models' / 'roi_mil' / 'meta_learner_logistic_regression_roi_mil.joblib',
    PROJECT_ROOT / 'ensemble' / 'models' / 'meta_learner_logistic_regression.joblib',
]

# Also check for JSON files with coefficients
json_files = [
    PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_roi_mil' / 'meta_learner_metrics.json',
    PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_augmented' / 'augmented_ensemble_metrics.json',
]

logger.info("="*80)
logger.info("CHECKING ALL META-LEARNER FILES")
logger.info("="*80)

results = []

# Check joblib files
for model_path in meta_learner_files:
    if not model_path.exists():
        logger.info(f"\n✗ Not found: {model_path}")
        continue
    
    try:
        model = joblib.load(model_path)
        logger.info(f"\n✓ Found: {model_path}")
        
        coef = model.coef_[0] if hasattr(model, 'coef_') else None
        intercept = model.intercept_[0] if hasattr(model, 'intercept_') else None
        
        # Try to get feature names
        feature_names = None
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        elif len(coef) == 3:
            # Assume standard order
            feature_names = ['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']
        
        # Map to model names
        model_mapping = {
            'hgg_prob_resnet': 'ResNet50-3D',
            'hgg_prob_swin': 'SwinUNETR-3D',
            'mil_prob': 'DualStreamMIL-3D',
            'hgg_prob_mil': 'DualStreamMIL-3D'
        }
        
        coefficients_dict = {}
        if feature_names and coef is not None:
            for feat_name, coef_value in zip(feature_names, coef):
                model_name = model_mapping.get(feat_name, feat_name)
                coefficients_dict[model_name] = float(coef_value)
        
        if intercept is not None:
            coefficients_dict['Intercept'] = float(intercept)
        
        results.append({
            'path': str(model_path),
            'type': 'joblib',
            'coefficients': coefficients_dict,
            'mil_coef': coefficients_dict.get('DualStreamMIL-3D', None)
        })
        
        logger.info(f"  Coefficients:")
        for name, value in coefficients_dict.items():
            logger.info(f"    {name}: {value:.6f}")
        
    except Exception as e:
        logger.error(f"  Error loading {model_path}: {e}")

# Check JSON files
for json_path in json_files:
    if not json_path.exists():
        logger.info(f"\n✗ Not found: {json_path}")
        continue
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"\n✓ Found: {json_path}")
        
        # Extract coefficients
        coefficients_dict = {}
        if 'model_coefficients' in data:
            model_mapping = {
                'hgg_prob_resnet': 'ResNet50-3D',
                'hgg_prob_swin': 'SwinUNETR-3D',
                'mil_prob': 'DualStreamMIL-3D',
                'hgg_prob_mil': 'DualStreamMIL-3D'
            }
            
            for feat_name, coef_value in data['model_coefficients'].items():
                model_name = model_mapping.get(feat_name, feat_name)
                coefficients_dict[model_name] = float(coef_value)
        
        if 'model_intercept' in data:
            coefficients_dict['Intercept'] = float(data['model_intercept'])
        
        results.append({
            'path': str(json_path),
            'type': 'json',
            'coefficients': coefficients_dict,
            'mil_coef': coefficients_dict.get('DualStreamMIL-3D', None)
        })
        
        logger.info(f"  Coefficients:")
        for name, value in coefficients_dict.items():
            logger.info(f"    {name}: {value:.6f}")
        
    except Exception as e:
        logger.error(f"  Error loading {json_path}: {e}")

# Summary
logger.info("\n" + "="*80)
logger.info("SUMMARY - DUALSTREAMMIL-3D COEFFICIENTS")
logger.info("="*80)

target_value = 0.09
tolerance = 0.01  # Within 0.01 of 0.09

found_match = False
for result in results:
    mil_coef = result['mil_coef']
    if mil_coef is not None:
        diff = abs(mil_coef - target_value)
        match = diff <= tolerance
        status = "✓ MATCH" if match else "✗"
        logger.info(f"{status} {result['path']}")
        logger.info(f"    DualStreamMIL-3D coefficient: {mil_coef:.6f} (target: {target_value:.2f}, diff: {diff:.6f})")
        if match:
            found_match = True

if not found_match:
    logger.info(f"\n✗ NO META-LEARNER FOUND WITH DualStreamMIL-3D COEFFICIENT ≈ {target_value}")
    logger.info(f"  Final coefficient is 0.020900 (from meta_learner_roi_mil)")
    logger.info(f"  The 0.09 value was NOT from the final model")

logger.info("\n" + "="*80)
logger.info("ALL META-LEARNER COEFFICIENTS TABLE")
logger.info("="*80)

for result in results:
    logger.info(f"\n{result['path']} ({result['type']}):")
    for name in ['ResNet50-3D', 'SwinUNETR-3D', 'DualStreamMIL-3D', 'Intercept']:
        value = result['coefficients'].get(name, 'N/A')
        if isinstance(value, float):
            logger.info(f"  {name:20s}: {value:+.6f}")
        else:
            logger.info(f"  {name:20s}: {value}")

