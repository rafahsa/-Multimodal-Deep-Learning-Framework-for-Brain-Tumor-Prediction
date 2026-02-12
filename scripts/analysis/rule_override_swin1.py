#!/usr/bin/env python3
"""
Rule-Based Override for Swin-1 False Negatives

This script implements a simple rule-based override to reduce Swin-1 false negatives
ONLY on uncertain LGG predictions, while controlling false positives.

NO RETRAINING - strictly post-hoc decision adjustment.
"""

import sys
from pathlib import Path

# Set project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UNCERTAIN_SAMPLES_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net' / 'uncertain_samples.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWIN1_THRESHOLD = 0.5  # Swin-1 decision threshold (unchanged)

# Rule parameters (configurable at the top)
SWIN_PROB_MIN = 0.40  # Minimum Swin-1 probability to consider override
ENTROPY_PERCENTILE = 90  # Percentile threshold for entropy (90th percentile)
REQUIRE_DISAGREEMENT = True  # Override only if ResNet OR MIL predicts HGG
ENTROPY_SCOPE = 'uncertain'  # Compute entropy threshold over 'uncertain' samples only, or 'all' for all samples


def compute_entropy(prob: float) -> float:
    """Compute binary entropy: -p*log2(p) - (1-p)*log2(1-p)."""
    if prob <= 0 or prob >= 1:
        return 0.0
    return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)


def apply_rule_override(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based override to reduce FN on uncertain LGG predictions.
    
    Override rule:
    - Only applies when: baseline_pred == 0 AND uncertainty_status == "uncertain"
    - Override conditions:
      1. hgg_prob_swin >= SWIN_PROB_MIN
      2. prediction_entropy >= entropy_threshold (percentile)
      3. (resnet_pred == 1 OR mil_pred == 1) if REQUIRE_DISAGREEMENT is True
    
    Args:
        df: DataFrame with columns: patient_id, fold, label, hgg_prob_swin, 
            hgg_prob_resnet, mil_prob, prediction_entropy, uncertainty_status
    
    Returns:
        DataFrame with added columns: baseline_pred, final_pred, overridden_flag
    """
    df = df.copy()
    
    # Ensure prediction_entropy exists (compute if missing)
    if 'prediction_entropy' not in df.columns or df['prediction_entropy'].isna().any():
        logger.info("Computing prediction_entropy from hgg_prob_swin...")
        df['prediction_entropy'] = df['hgg_prob_swin'].apply(compute_entropy)
    
    # Compute baseline predictions
    df['baseline_pred'] = (df['hgg_prob_swin'] >= SWIN1_THRESHOLD).astype(int)
    
    # Compute disagreement signals
    df['resnet_pred'] = (df['hgg_prob_resnet'] >= SWIN1_THRESHOLD).astype(int)
    df['mil_pred'] = (df['mil_prob'] >= SWIN1_THRESHOLD).astype(int)
    
    # Compute entropy threshold
    if ENTROPY_SCOPE == 'uncertain':
        # Compute threshold over uncertain samples only
        uncertain_mask = df['uncertainty_status'] == 'uncertain'
        if uncertain_mask.sum() > 0:
            entropy_threshold = np.percentile(df.loc[uncertain_mask, 'prediction_entropy'], ENTROPY_PERCENTILE)
        else:
            logger.warning("No uncertain samples found, using threshold from all samples")
            entropy_threshold = np.percentile(df['prediction_entropy'], ENTROPY_PERCENTILE)
    else:
        # Compute threshold over all samples
        entropy_threshold = np.percentile(df['prediction_entropy'], ENTROPY_PERCENTILE)
    
    logger.info(f"Entropy threshold ({ENTROPY_PERCENTILE}th percentile, scope={ENTROPY_SCOPE}): {entropy_threshold:.6f}")
    
    # Initialize final predictions (default to baseline)
    df['final_pred'] = df['baseline_pred'].copy()
    df['overridden_flag'] = 0
    
    # Apply override ONLY when: baseline_pred == 0 AND uncertainty_status == "uncertain"
    override_candidates = (df['baseline_pred'] == 0) & (df['uncertainty_status'] == 'uncertain')
    
    if override_candidates.sum() > 0:
        logger.info(f"\nEvaluating {override_candidates.sum()} override candidates...")
        
        # Check override conditions
        condition_1 = df.loc[override_candidates, 'hgg_prob_swin'] >= SWIN_PROB_MIN
        condition_2 = df.loc[override_candidates, 'prediction_entropy'] >= entropy_threshold
        
        if REQUIRE_DISAGREEMENT:
            condition_3 = (df.loc[override_candidates, 'resnet_pred'] == 1) | (df.loc[override_candidates, 'mil_pred'] == 1)
        else:
            condition_3 = pd.Series(True, index=df.loc[override_candidates].index)
        
        # Apply override when all conditions are met
        override_mask = override_candidates & condition_1 & condition_2 & condition_3
        
        df.loc[override_mask, 'final_pred'] = 1
        df.loc[override_mask, 'overridden_flag'] = 1
        
        n_overridden = override_mask.sum()
        logger.info(f"  Overridden: {n_overridden} predictions")
        logger.info(f"  Condition 1 (hgg_prob_swin >= {SWIN_PROB_MIN}): {condition_1.sum()}")
        logger.info(f"  Condition 2 (entropy >= {entropy_threshold:.6f}): {condition_2.sum()}")
        if REQUIRE_DISAGREEMENT:
            logger.info(f"  Condition 3 (resnet_pred==1 OR mil_pred==1): {condition_3.sum()}")
    else:
        logger.info("No override candidates found (no uncertain LGG predictions)")
    
    # Confident samples always keep baseline_pred (already set)
    confident_count = (df['uncertainty_status'] == 'confident').sum()
    logger.info(f"\nConfident samples (kept baseline): {confident_count}")
    
    return df


def main():
    logger.info("="*80)
    logger.info("RULE-BASED OVERRIDE FOR SWIN-1 FALSE NEGATIVES")
    logger.info("="*80)
    
    logger.info(f"\nRule Parameters:")
    logger.info(f"  SWIN_PROB_MIN: {SWIN_PROB_MIN}")
    logger.info(f"  ENTROPY_PERCENTILE: {ENTROPY_PERCENTILE}")
    logger.info(f"  REQUIRE_DISAGREEMENT: {REQUIRE_DISAGREEMENT}")
    logger.info(f"  ENTROPY_SCOPE: {ENTROPY_SCOPE}")
    
    # Load uncertain samples
    logger.info(f"\nLoading uncertain samples from: {UNCERTAIN_SAMPLES_FILE}")
    df = pd.read_csv(UNCERTAIN_SAMPLES_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    # Apply rule override
    logger.info("\nApplying rule-based override...")
    df_final = apply_rule_override(df)
    
    # Statistics
    baseline_fn = ((df_final['label'] == 1) & (df_final['baseline_pred'] == 0)).sum()
    final_fn = ((df_final['label'] == 1) & (df_final['final_pred'] == 0)).sum()
    fn_reduction = baseline_fn - final_fn
    
    baseline_fp = ((df_final['label'] == 0) & (df_final['baseline_pred'] == 1)).sum()
    final_fp = ((df_final['label'] == 0) & (df_final['final_pred'] == 1)).sum()
    fp_change = final_fp - baseline_fp
    
    logger.info(f"\nResults:")
    logger.info(f"  Baseline FN: {baseline_fn}")
    logger.info(f"  Final FN: {final_fn}")
    logger.info(f"  FN Reduction: {fn_reduction}")
    logger.info(f"  Baseline FP: {baseline_fp}")
    logger.info(f"  Final FP: {final_fp}")
    logger.info(f"  FP Change: {fp_change:+d}")
    logger.info(f"  Overridden predictions: {df_final['overridden_flag'].sum()}")
    
    # Save final predictions
    output_file = OUTPUT_DIR / 'rule_override_predictions.csv'
    
    # Prepare output DataFrame with required columns
    output_df = df_final[[
        'patient_id', 'fold', 'label', 'baseline_pred', 'final_pred',
        'hgg_prob_swin', 'prediction_entropy', 'hgg_prob_resnet', 'mil_prob',
        'uncertainty_status', 'overridden_flag'
    ]].copy()
    
    output_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved rule override predictions to: {output_file}")
    
    logger.info("\n" + "="*80)
    logger.info("RULE OVERRIDE COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

