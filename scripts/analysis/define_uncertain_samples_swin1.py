#!/usr/bin/env python3
"""
Define Uncertain Samples for Swin-1 Hybrid Safety-Net

This script tags each sample as "confident" or "uncertain" based on:
- Probability-based uncertainty (default: 0.30 ≤ hgg_prob_swin ≤ 0.60)
- Entropy-based uncertainty (high prediction entropy)

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
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OOF_FILE = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_entropy(prob: float) -> float:
    """Compute binary entropy: -p*log(p) - (1-p)*log(1-p)."""
    if prob <= 0 or prob >= 1:
        return 0.0
    return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)


def define_uncertain_samples(
    df: pd.DataFrame,
    prob_lower: float = 0.30,
    prob_upper: float = 0.60,
    entropy_percentile: float = 75.0,
    use_entropy: bool = True
) -> pd.DataFrame:
    """
    Tag samples as confident or uncertain.
    
    Args:
        df: DataFrame with hgg_prob_swin column
        prob_lower: Lower bound for probability-based uncertainty
        prob_upper: Upper bound for probability-based uncertainty
        entropy_percentile: Percentile threshold for entropy-based uncertainty
        use_entropy: Whether to use entropy-based uncertainty
    
    Returns:
        DataFrame with 'uncertainty_status' column ('confident' or 'uncertain')
    """
    df = df.copy()
    
    # Compute entropy
    df['prediction_entropy'] = df['hgg_prob_swin'].apply(compute_entropy)
    
    # Probability-based uncertainty
    prob_uncertain = (df['hgg_prob_swin'] >= prob_lower) & (df['hgg_prob_swin'] <= prob_upper)
    
    # Entropy-based uncertainty
    if use_entropy:
        entropy_threshold = np.percentile(df['prediction_entropy'], entropy_percentile)
        entropy_uncertain = df['prediction_entropy'] >= entropy_threshold
        # Combine: uncertain if either condition is met
        uncertain_mask = prob_uncertain | entropy_uncertain
    else:
        uncertain_mask = prob_uncertain
    
    # Tag samples
    df['uncertainty_status'] = 'confident'
    df.loc[uncertain_mask, 'uncertainty_status'] = 'uncertain'
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Define uncertain samples for hybrid safety-net")
    parser.add_argument('--prob-lower', type=float, default=0.30,
                       help='Lower bound for probability-based uncertainty (default: 0.30)')
    parser.add_argument('--prob-upper', type=float, default=0.60,
                       help='Upper bound for probability-based uncertainty (default: 0.60)')
    parser.add_argument('--entropy-percentile', type=float, default=75.0,
                       help='Percentile threshold for entropy-based uncertainty (default: 75.0)')
    parser.add_argument('--no-entropy', action='store_true',
                       help='Disable entropy-based uncertainty (use only probability-based)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("DEFINE UNCERTAIN SAMPLES FOR SWIN-1 HYBRID SAFETY-NET")
    logger.info("="*80)
    
    # Load OOF predictions
    logger.info(f"\nLoading OOF predictions from: {OOF_FILE}")
    df = pd.read_csv(OOF_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    # Define uncertain samples
    logger.info(f"\nDefining uncertain samples:")
    logger.info(f"  Probability range: [{args.prob_lower}, {args.prob_upper}]")
    logger.info(f"  Entropy percentile: {args.entropy_percentile} (enabled: {not args.no_entropy})")
    
    df_tagged = define_uncertain_samples(
        df,
        prob_lower=args.prob_lower,
        prob_upper=args.prob_upper,
        entropy_percentile=args.entropy_percentile,
        use_entropy=not args.no_entropy
    )
    
    # Statistics
    confident_count = (df_tagged['uncertainty_status'] == 'confident').sum()
    uncertain_count = (df_tagged['uncertainty_status'] == 'uncertain').sum()
    
    logger.info(f"\nUncertainty Status:")
    logger.info(f"  Confident: {confident_count} ({confident_count/len(df_tagged)*100:.1f}%)")
    logger.info(f"  Uncertain: {uncertain_count} ({uncertain_count/len(df_tagged)*100:.1f}%)")
    
    # Save tagged samples
    output_file = OUTPUT_DIR / 'uncertain_samples.csv'
    df_tagged.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved tagged samples to: {output_file}")
    
    # Per-fold statistics
    logger.info("\n" + "="*80)
    logger.info("PER-FOLD STATISTICS")
    logger.info("="*80)
    for fold in sorted(df_tagged['fold'].unique()):
        fold_df = df_tagged[df_tagged['fold'] == fold]
        fold_confident = (fold_df['uncertainty_status'] == 'confident').sum()
        fold_uncertain = (fold_df['uncertainty_status'] == 'uncertain').sum()
        logger.info(f"Fold {fold}: Confident={fold_confident}, Uncertain={fold_uncertain}")
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

