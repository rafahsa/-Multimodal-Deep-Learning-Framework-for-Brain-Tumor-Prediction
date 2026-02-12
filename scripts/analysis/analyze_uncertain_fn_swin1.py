#!/usr/bin/env python3
"""
Analyze Baseline False Negatives (FN) for Swin-1

This script analyzes baseline FN predictions in detail, splitting them into
uncertain and confident categories, and computing descriptive statistics.

NO RETRAINING - strictly post-hoc analysis.
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


def compute_entropy(prob: float) -> float:
    """Compute binary entropy: -p*log2(p) - (1-p)*log2(1-p)."""
    if prob <= 0 or prob >= 1:
        return 0.0
    return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)


def print_descriptive_stats(series: pd.Series, name: str):
    """Print descriptive statistics for a series."""
    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Count: {len(series)}")
    logger.info(f"  Min: {series.min():.6f}")
    logger.info(f"  Max: {series.max():.6f}")
    logger.info(f"  Mean: {series.mean():.6f}")
    logger.info(f"  Median: {series.median():.6f}")
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    for q in quantiles:
        logger.info(f"  {q*100:.0f}th percentile: {series.quantile(q):.6f}")


def main():
    logger.info("="*80)
    logger.info("ANALYZE BASELINE FALSE NEGATIVES (FN) FOR SWIN-1")
    logger.info("="*80)
    
    # Load uncertain samples
    logger.info(f"\nLoading uncertain samples from: {UNCERTAIN_SAMPLES_FILE}")
    df = pd.read_csv(UNCERTAIN_SAMPLES_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    # Ensure prediction_entropy exists (compute if missing)
    if 'prediction_entropy' not in df.columns or df['prediction_entropy'].isna().any():
        logger.info("Computing prediction_entropy from hgg_prob_swin...")
        df['prediction_entropy'] = df['hgg_prob_swin'].apply(compute_entropy)
    
    # Create baseline predictions
    df['baseline_pred'] = (df['hgg_prob_swin'] >= SWIN1_THRESHOLD).astype(int)
    
    # Identify baseline FN: label==1 (HGG) and baseline_pred==0 (predicted as LGG)
    fn_mask = (df['label'] == 1) & (df['baseline_pred'] == 0)
    df_fn = df[fn_mask].copy()
    
    logger.info(f"\nBaseline False Negatives:")
    logger.info(f"  Total FN: {len(df_fn)}")
    
    # Split FN into uncertain and confident
    fn_uncertain_mask = df_fn['uncertainty_status'] == 'uncertain'
    df_fn_uncertain = df_fn[fn_uncertain_mask].copy()
    df_fn_confident = df_fn[~fn_uncertain_mask].copy()
    
    logger.info(f"  FN_uncertain: {len(df_fn_uncertain)}")
    logger.info(f"  FN_confident: {len(df_fn_confident)}")
    
    # Compute disagreement signals
    df_fn['resnet_pred'] = (df_fn['hgg_prob_resnet'] >= SWIN1_THRESHOLD).astype(int)
    df_fn['mil_pred'] = (df_fn['mil_prob'] >= SWIN1_THRESHOLD).astype(int)
    
    # For FN_uncertain: disagreement analysis
    if len(df_fn_uncertain) > 0:
        logger.info("\n" + "="*80)
        logger.info("FN_UNCERTAIN ANALYSIS")
        logger.info("="*80)
        
        # Descriptive stats for hgg_prob_swin
        print_descriptive_stats(df_fn_uncertain['hgg_prob_swin'], "hgg_prob_swin")
        
        # Descriptive stats for prediction_entropy
        print_descriptive_stats(df_fn_uncertain['prediction_entropy'], "prediction_entropy")
        
        # Disagreement signals
        logger.info("\nDisagreement Signals (FN_uncertain):")
        resnet_pred_1 = (df_fn_uncertain['resnet_pred'] == 1).sum()
        mil_pred_1 = (df_fn_uncertain['mil_pred'] == 1).sum()
        either_pred_1 = ((df_fn_uncertain['resnet_pred'] == 1) | (df_fn_uncertain['mil_pred'] == 1)).sum()
        
        logger.info(f"  ResNet predicts HGG (resnet_pred==1): {resnet_pred_1} / {len(df_fn_uncertain)} ({resnet_pred_1/len(df_fn_uncertain)*100:.1f}%)")
        logger.info(f"  MIL predicts HGG (mil_pred==1): {mil_pred_1} / {len(df_fn_uncertain)} ({mil_pred_1/len(df_fn_uncertain)*100:.1f}%)")
        logger.info(f"  Either ResNet OR MIL predicts HGG: {either_pred_1} / {len(df_fn_uncertain)} ({either_pred_1/len(df_fn_uncertain)*100:.1f}%)")
    
    # For FN_confident: descriptive stats
    if len(df_fn_confident) > 0:
        logger.info("\n" + "="*80)
        logger.info("FN_CONFIDENT ANALYSIS")
        logger.info("="*80)
        
        # Descriptive stats for hgg_prob_swin
        print_descriptive_stats(df_fn_confident['hgg_prob_swin'], "hgg_prob_swin")
        
        # Descriptive stats for prediction_entropy
        print_descriptive_stats(df_fn_confident['prediction_entropy'], "prediction_entropy")
    
    # Save analysis report
    output_file = OUTPUT_DIR / 'fn_uncertainty_analysis.csv'
    
    # Prepare output DataFrame with required columns
    output_df = df_fn[[
        'patient_id', 'fold', 'label', 'hgg_prob_swin', 'prediction_entropy',
        'hgg_prob_resnet', 'mil_prob', 'resnet_pred', 'mil_pred', 'uncertainty_status'
    ]].copy()
    
    output_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved FN analysis report to: {output_file}")
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

