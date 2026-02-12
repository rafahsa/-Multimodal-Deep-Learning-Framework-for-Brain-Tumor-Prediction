#!/usr/bin/env python3
"""
Improve Ensemble Performance WITHOUT Retraining Deep Learning Models

This script implements a comprehensive pipeline to improve ensemble performance:
1. Test-Time Augmentation (TTA) for Swin and ResNet
2. Nested-CV safe calibration
3. Threshold tuning
4. Non-DL feature extraction
5. Meta-learner retraining
6. Ablation study

Usage:
    python scripts/ensemble/improve_ensemble_no_retraining.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OOF_PREDICTIONS = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'ensemble_improvements'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import step modules
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.ensemble.step1_tta import apply_tta_to_oof
from scripts.ensemble.step2_calibration import apply_nested_cv_calibration
from scripts.ensemble.step3_threshold_tuning import tune_ensemble_thresholds
from scripts.ensemble.step4_non_dl_features import extract_non_dl_features
from scripts.ensemble.step5_meta_learner import retrain_meta_learner_with_features
from scripts.ensemble.step6_ablation import run_ablation_study


def main():
    """Main pipeline orchestrator."""
    logger.info("="*80)
    logger.info("ENSEMBLE IMPROVEMENT PIPELINE (NO RETRAINING)")
    logger.info("="*80)
    logger.info("Goal: Reduce FN, stabilize probabilities, add orthogonal signal")
    logger.info("Constraint: NO retraining of Swin/ResNet/MIL models")
    logger.info("="*80)
    
    # Load baseline OOF predictions
    if not OOF_PREDICTIONS.exists():
        logger.error(f"OOF predictions not found: {OOF_PREDICTIONS}")
        return
    
    logger.info(f"\nLoading baseline OOF predictions from: {OOF_PREDICTIONS}")
    df_baseline = pd.read_csv(OOF_PREDICTIONS)
    logger.info(f"Loaded {len(df_baseline)} patients")
    logger.info(f"Columns: {list(df_baseline.columns)}")
    
    # Step 1: Test-Time Augmentation
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Test-Time Augmentation (TTA)")
    logger.info("="*80)
    df_tta = apply_tta_to_oof(df_baseline, output_dir=OUTPUT_DIR)
    
    # Step 2: Calibration
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Nested-CV Safe Calibration")
    logger.info("="*80)
    df_cal = apply_nested_cv_calibration(df_tta, output_dir=OUTPUT_DIR)
    
    # Step 3: Threshold Tuning
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Threshold Tuning for Ensemble")
    logger.info("="*80)
    threshold_results = tune_ensemble_thresholds(df_cal, output_dir=OUTPUT_DIR)
    
    # Step 4: Non-DL Feature Extraction
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Non-DL Feature Extraction")
    logger.info("="*80)
    df_features = extract_non_dl_features(df_cal, output_dir=OUTPUT_DIR)
    
    # Step 5: Meta-Learner Retraining
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Meta-Learner Retraining")
    logger.info("="*80)
    meta_learner_results = retrain_meta_learner_with_features(df_features, output_dir=OUTPUT_DIR)
    
    # Step 6: Ablation Study
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Ablation Study")
    logger.info("="*80)
    ablation_results = run_ablation_study(
        df_baseline, df_tta, df_cal, df_features, 
        threshold_results, meta_learner_results,
        output_dir=OUTPUT_DIR
    )
    
    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("\nAblation Study Results:")
    for config, metrics in ablation_results.items():
        logger.info(f"\n{config}:")
        logger.info(f"  FN: {metrics.get('fn_mean', 'N/A'):.2f} ± {metrics.get('fn_std', 0):.2f}")
        logger.info(f"  FP: {metrics.get('fp_mean', 'N/A'):.2f} ± {metrics.get('fp_std', 0):.2f}")
        logger.info(f"  Recall: {metrics.get('recall_mean', 'N/A'):.4f} ± {metrics.get('recall_std', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('precision_mean', 'N/A'):.4f} ± {metrics.get('precision_std', 0):.4f}")
        logger.info(f"  AUC: {metrics.get('auc_mean', 'N/A'):.4f} ± {metrics.get('auc_std', 0):.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()


