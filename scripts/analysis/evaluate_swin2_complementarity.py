#!/usr/bin/env python3
"""
Evaluate Swin-2 Complementarity: Does Swin-2 Provide New Information vs Swin-1?

This script evaluates whether Swin-2 provides complementary signal to Swin-1,
specifically focusing on:
1. Does Swin-2 assign higher probabilities to Swin-1 FN cases than to TN cases?
2. Is the correlation(Swin-1 prob, Swin-2 prob) < 0.7?
3. Does Swin-2 rank Swin-1 FN cases higher than Swin-1 itself?

PRIMARY GOAL: Complementarity, NOT accuracy.
"""

import sys
from pathlib import Path

# Set project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GO/NO-GO criteria
FN_RANKING_AUC_THRESHOLD = 0.05  # Swin-2 FN ranking AUC must be ≥ Swin-1 + 0.05
CORRELATION_THRESHOLD = 0.70  # Correlation must be < 0.70


def load_predictions(swin1_oof_file: Path, swin2_predictions_file: Path, fold_id: int) -> pd.DataFrame:
    """Load and merge Swin-1 and Swin-2 predictions for specified fold."""
    logger.info(f"Loading Swin-1 OOF predictions from: {swin1_oof_file}")
    swin1_df = pd.read_csv(swin1_oof_file)
    swin1_fold = swin1_df[swin1_df['fold'] == fold_id].copy()
    
    logger.info(f"Loading Swin-2 predictions from: {swin2_predictions_file}")
    swin2_df = pd.read_csv(swin2_predictions_file)
    
    # Merge
    merged = swin1_fold.merge(
        swin2_df[['patient_id', 'swin2_prob']],
        on='patient_id',
        how='inner'
    )
    
    logger.info(f"Merged predictions: {len(merged)} patients")
    return merged


def identify_groups(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Identify Swin-1 FN, TP, TN, FP groups."""
    # Swin-1 predictions (threshold=0.5)
    swin1_pred = (df['hgg_prob_swin'] >= 0.5).astype(int)
    
    # Groups
    groups = {
        'swin1_fn': df[(df['label'] == 1) & (swin1_pred == 0)].index.values,  # HGG missed
        'swin1_tp': df[(df['label'] == 1) & (swin1_pred == 1)].index.values,  # HGG caught
        'swin1_tn': df[(df['label'] == 0) & (swin1_pred == 0)].index.values,  # LGG correct
        'swin1_fp': df[(df['label'] == 0) & (swin1_pred == 1)].index.values,  # LGG false alarm
    }
    
    logger.info(f"Swin-1 FN: {len(groups['swin1_fn'])}, TP: {len(groups['swin1_tp'])}, "
               f"TN: {len(groups['swin1_tn'])}, FP: {len(groups['swin1_fp'])}")
    
    return groups


def compute_probability_distributions(df: pd.DataFrame, groups: Dict[str, np.ndarray]) -> Dict:
    """Compute probability distributions for different groups."""
    distributions = {}
    
    for group_name, indices in groups.items():
        if len(indices) > 0:
            swin1_probs = df.loc[indices, 'hgg_prob_swin'].values
            swin2_probs = df.loc[indices, 'swin2_prob'].values
            
            distributions[group_name] = {
                'swin1_mean': float(np.mean(swin1_probs)),
                'swin1_std': float(np.std(swin1_probs)),
                'swin1_median': float(np.median(swin1_probs)),
                'swin2_mean': float(np.mean(swin2_probs)),
                'swin2_std': float(np.std(swin2_probs)),
                'swin2_median': float(np.median(swin2_probs)),
                'count': len(indices)
            }
        else:
            distributions[group_name] = {
                'swin1_mean': 0.0,
                'swin1_std': 0.0,
                'swin1_median': 0.0,
                'swin2_mean': 0.0,
                'swin2_std': 0.0,
                'swin2_median': 0.0,
                'count': 0
            }
    
    return distributions


def compute_fn_ranking_auc(df: pd.DataFrame, groups: Dict[str, np.ndarray]) -> Dict:
    """
    Compute ROC-AUC for ranking Swin-1 FN cases vs rest.
    
    Task: Can Swin-2 distinguish Swin-1 FN cases from all other cases?
    """
    fn_indices = groups['swin1_fn']
    other_indices = np.concatenate([
        groups['swin1_tp'],
        groups['swin1_tn'],
        groups['swin1_fp']
    ])
    
    if len(fn_indices) == 0 or len(other_indices) == 0:
        return {
            'swin1_auc': 0.0,
            'swin2_auc': 0.0,
            'improvement': 0.0
        }
    
    # Create binary labels: FN=1, others=0
    all_indices = np.concatenate([fn_indices, other_indices])
    y_true = np.concatenate([
        np.ones(len(fn_indices)),  # FN cases
        np.zeros(len(other_indices))  # Others
    ])
    
    # Swin-1 probabilities
    swin1_probs = df.loc[all_indices, 'hgg_prob_swin'].values
    swin1_auc = roc_auc_score(y_true, swin1_probs)
    
    # Swin-2 probabilities
    swin2_probs = df.loc[all_indices, 'swin2_prob'].values
    swin2_auc = roc_auc_score(y_true, swin2_probs)
    
    improvement = swin2_auc - swin1_auc
    
    return {
        'swin1_auc': float(swin1_auc),
        'swin2_auc': float(swin2_auc),
        'improvement': float(improvement)
    }


def compute_correlations(df: pd.DataFrame) -> Dict:
    """Compute Pearson and Spearman correlations between Swin-1 and Swin-2."""
    swin1_probs = df['hgg_prob_swin'].values
    swin2_probs = df['swin2_prob'].values
    
    pearson_r, pearson_p = pearsonr(swin1_probs, swin2_probs)
    spearman_r, spearman_p = spearmanr(swin1_probs, swin2_probs)
    
    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p)
    }


def evaluate_separation(df: pd.DataFrame, groups: Dict[str, np.ndarray]) -> Dict:
    """
    Evaluate separation between Swin-1 FN and TN in Swin-2 scores.
    
    Good complementarity: Swin-2 should assign higher scores to FN than TN.
    """
    fn_indices = groups['swin1_fn']
    tn_indices = groups['swin1_tn']
    
    if len(fn_indices) == 0 or len(tn_indices) == 0:
        return {
            'fn_mean': 0.0,
            'tn_mean': 0.0,
            'separation': 0.0,
            'clear_separation': False
        }
    
    fn_scores = df.loc[fn_indices, 'swin2_prob'].values
    tn_scores = df.loc[tn_indices, 'swin2_prob'].values
    
    fn_mean = float(np.mean(fn_scores))
    tn_mean = float(np.mean(tn_scores))
    separation = fn_mean - tn_mean
    
    # Clear separation: FN mean > TN mean (Swin-2 ranks FN higher)
    clear_separation = separation > 0.0
    
    return {
        'fn_mean': fn_mean,
        'tn_mean': tn_mean,
        'separation': separation,
        'clear_separation': bool(clear_separation)
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Swin-2 complementarity")
    parser.add_argument('--swin1-oof', type=str, required=True,
                       help='Path to Swin-1 OOF predictions CSV')
    parser.add_argument('--swin2-predictions', type=str, required=True,
                       help='Path to Swin-2 predictions CSV')
    parser.add_argument('--fold-id', type=int, required=True,
                       help='Fold ID (0-4)')
    parser.add_argument('--output-dir', type=str, default='ensemble/results/swin2_complementarity',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    swin1_oof_file = Path(args.swin1_oof)
    swin2_predictions_file = Path(args.swin2_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("SWIN-2 COMPLEMENTARITY EVALUATION")
    logger.info("="*80)
    
    # Load predictions
    df = load_predictions(swin1_oof_file, swin2_predictions_file, args.fold_id)
    
    # Identify groups
    groups = identify_groups(df)
    
    # Compute probability distributions
    logger.info("\n" + "="*80)
    logger.info("PROBABILITY DISTRIBUTIONS")
    logger.info("="*80)
    distributions = compute_probability_distributions(df, groups)
    
    for group_name, stats in distributions.items():
        logger.info(f"\n{group_name.upper()} (n={stats['count']}):")
        logger.info(f"  Swin-1: mean={stats['swin1_mean']:.4f}, median={stats['swin1_median']:.4f}")
        logger.info(f"  Swin-2: mean={stats['swin2_mean']:.4f}, median={stats['swin2_median']:.4f}")
    
    # Compute FN ranking AUC
    logger.info("\n" + "="*80)
    logger.info("FN RANKING AUC (FN vs Rest)")
    logger.info("="*80)
    fn_ranking = compute_fn_ranking_auc(df, groups)
    logger.info(f"Swin-1 AUC: {fn_ranking['swin1_auc']:.4f}")
    logger.info(f"Swin-2 AUC: {fn_ranking['swin2_auc']:.4f}")
    logger.info(f"Improvement: {fn_ranking['improvement']:+.4f}")
    logger.info(f"Threshold: ≥ {fn_ranking['swin1_auc']:.4f} + {FN_RANKING_AUC_THRESHOLD} = {fn_ranking['swin1_auc'] + FN_RANKING_AUC_THRESHOLD:.4f}")
    
    # Compute correlations
    logger.info("\n" + "="*80)
    logger.info("CORRELATION ANALYSIS")
    logger.info("="*80)
    correlations = compute_correlations(df)
    logger.info(f"Pearson r: {correlations['pearson_r']:.4f} (p={correlations['pearson_p']:.4e})")
    logger.info(f"Spearman r: {correlations['spearman_r']:.4f} (p={correlations['spearman_p']:.4e})")
    logger.info(f"Threshold: < {CORRELATION_THRESHOLD}")
    
    # Evaluate separation
    logger.info("\n" + "="*80)
    logger.info("FN vs TN SEPARATION")
    logger.info("="*80)
    separation = evaluate_separation(df, groups)
    logger.info(f"FN mean (Swin-2): {separation['fn_mean']:.4f}")
    logger.info(f"TN mean (Swin-2): {separation['tn_mean']:.4f}")
    logger.info(f"Separation: {separation['separation']:+.4f}")
    logger.info(f"Clear separation: {separation['clear_separation']}")
    
    # GO/NO-GO decision
    logger.info("\n" + "="*80)
    logger.info("GO/NO-GO DECISION")
    logger.info("="*80)
    
    criterion1 = fn_ranking['improvement'] >= FN_RANKING_AUC_THRESHOLD
    criterion2 = abs(correlations['pearson_r']) < CORRELATION_THRESHOLD
    criterion3 = separation['clear_separation']
    
    logger.info(f"Criterion 1 (FN ranking AUC improvement ≥ {FN_RANKING_AUC_THRESHOLD}): {criterion1} ({fn_ranking['improvement']:+.4f})")
    logger.info(f"Criterion 2 (Correlation < {CORRELATION_THRESHOLD}): {criterion2} ({correlations['pearson_r']:.4f})")
    logger.info(f"Criterion 3 (Clear FN/TN separation): {criterion3}")
    
    if criterion1 and criterion2 and criterion3:
        decision = "CONTINUE"
        reason = "All complementarity criteria met"
    else:
        decision = "STOP"
        reasons = []
        if not criterion1:
            reasons.append(f"FN ranking AUC improvement {fn_ranking['improvement']:+.4f} < {FN_RANKING_AUC_THRESHOLD}")
        if not criterion2:
            reasons.append(f"Correlation {correlations['pearson_r']:.4f} >= {CORRELATION_THRESHOLD}")
        if not criterion3:
            reasons.append("No clear FN/TN separation")
        reason = "; ".join(reasons)
    
    logger.info(f"\nDECISION: {decision}")
    logger.info(f"REASON: {reason}")
    
    # Compile results
    results = {
        'fold_id': args.fold_id,
        'distributions': distributions,
        'fn_ranking_auc': fn_ranking,
        'correlations': correlations,
        'separation': separation,
        'decision': {
            'result': decision,
            'reason': reason,
            'criterion1_met': bool(criterion1),
            'criterion2_met': bool(criterion2),
            'criterion3_met': bool(criterion3)
        }
    }
    
    # Save results
    json_path = output_dir / f'fold_{args.fold_id}_complementarity.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Saved results to: {json_path}")
    
    # Generate markdown report
    md_path = output_dir / f'fold_{args.fold_id}_complementarity.md'
    with open(md_path, 'w') as f:
        f.write(f"# Swin-2 Complementarity Evaluation: Fold {args.fold_id}\n\n")
        f.write(f"## Decision: **{decision}**\n\n")
        f.write(f"**Reason:** {reason}\n\n")
        f.write("## Criteria\n\n")
        f.write(f"1. **FN Ranking AUC Improvement ≥ {FN_RANKING_AUC_THRESHOLD}:** {criterion1} ({fn_ranking['improvement']:+.4f})\n")
        f.write(f"2. **Correlation < {CORRELATION_THRESHOLD}:** {criterion2} ({correlations['pearson_r']:.4f})\n")
        f.write(f"3. **Clear FN/TN Separation:** {criterion3}\n\n")
        f.write("## Results\n\n")
        f.write("### FN Ranking AUC\n\n")
        f.write(f"- Swin-1 AUC: {fn_ranking['swin1_auc']:.4f}\n")
        f.write(f"- Swin-2 AUC: {fn_ranking['swin2_auc']:.4f}\n")
        f.write(f"- Improvement: {fn_ranking['improvement']:+.4f}\n\n")
        f.write("### Correlations\n\n")
        f.write(f"- Pearson r: {correlations['pearson_r']:.4f}\n")
        f.write(f"- Spearman r: {correlations['spearman_r']:.4f}\n\n")
        f.write("### FN vs TN Separation\n\n")
        f.write(f"- FN mean (Swin-2): {separation['fn_mean']:.4f}\n")
        f.write(f"- TN mean (Swin-2): {separation['tn_mean']:.4f}\n")
        f.write(f"- Separation: {separation['separation']:+.4f}\n\n")
    
    logger.info(f"✓ Saved markdown report to: {md_path}")


if __name__ == '__main__':
    main()


