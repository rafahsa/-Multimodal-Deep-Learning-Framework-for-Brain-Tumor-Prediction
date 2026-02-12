#!/usr/bin/env python3
"""
Hybrid Inference: Swin-1 + Safety-Net

This script implements the hybrid decision logic:
- If confident → use Swin-1 prediction
- If uncertain → use meta-decision prediction

NO RETRAINING - strictly post-hoc inference.
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
META_PREDICTIONS_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net' / 'uncertain_meta_predictions.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWIN1_THRESHOLD = 0.5  # Swin-1 decision threshold (unchanged)


def hybrid_inference(df: pd.DataFrame, meta_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Implement hybrid inference logic.
    
    For each patient:
    - If confident → Final prediction = Swin-1 prediction
    - If uncertain → Final prediction = meta-decision prediction
    """
    df = df.copy()
    
    # Merge meta-predictions
    df = df.merge(meta_predictions[['patient_id', 'meta_prob']], on='patient_id', how='left')
    
    # Initialize final predictions
    df['hybrid_prob'] = df['hgg_prob_swin'].copy()  # Default to Swin-1
    df['hybrid_pred'] = (df['hgg_prob_swin'] >= SWIN1_THRESHOLD).astype(int)  # Default to Swin-1
    
    # For uncertain samples, use meta-decision
    uncertain_mask = df['uncertainty_status'] == 'uncertain'
    df.loc[uncertain_mask, 'hybrid_prob'] = df.loc[uncertain_mask, 'meta_prob']
    df.loc[uncertain_mask, 'hybrid_pred'] = (df.loc[uncertain_mask, 'meta_prob'] >= SWIN1_THRESHOLD).astype(int)
    
    # Fill NaN meta_prob with Swin-1 prob (for confident samples)
    df['meta_prob'] = df['meta_prob'].fillna(df['hgg_prob_swin'])
    
    return df


def main():
    logger.info("="*80)
    logger.info("HYBRID INFERENCE: SWIN-1 + SAFETY-NET")
    logger.info("="*80)
    
    # Load uncertain samples
    logger.info(f"\nLoading uncertain samples from: {UNCERTAIN_SAMPLES_FILE}")
    df = pd.read_csv(UNCERTAIN_SAMPLES_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    # Load meta-predictions
    logger.info(f"Loading meta-predictions from: {META_PREDICTIONS_FILE}")
    meta_predictions = pd.read_csv(META_PREDICTIONS_FILE)
    logger.info(f"Loaded {len(meta_predictions)} meta-predictions")
    
    # Hybrid inference
    logger.info("\nApplying hybrid inference logic...")
    df_hybrid = hybrid_inference(df, meta_predictions)
    
    # Statistics
    confident_count = (df_hybrid['uncertainty_status'] == 'confident').sum()
    uncertain_count = (df_hybrid['uncertainty_status'] == 'uncertain').sum()
    
    confident_swin1 = (df_hybrid['uncertainty_status'] == 'confident').sum()
    uncertain_meta = (df_hybrid['uncertainty_status'] == 'uncertain').sum()
    
    logger.info(f"Confident samples (using Swin-1): {confident_count}")
    logger.info(f"Uncertain samples (using meta-decision): {uncertain_count}")
    
    # Save hybrid predictions
    output_file = OUTPUT_DIR / 'hybrid_predictions.csv'
    df_hybrid.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved hybrid predictions to: {output_file}")
    
    logger.info("\n" + "="*80)
    logger.info("HYBRID INFERENCE COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

