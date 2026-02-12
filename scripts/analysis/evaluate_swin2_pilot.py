#!/usr/bin/env python3
"""
Evaluate Swin-2 Pilot Experiment: GO/NO-GO Decision Gate

This script evaluates a single-fold Swin-2 pilot experiment against Swin-1
and makes a GO/NO-GO decision based on:
1. FN reduction >= 30%
2. Correlation < 0.70

Usage:
    python scripts/analysis/evaluate_swin2_pilot.py \
        --swin1-oof ensemble/oof_predictions/merged_oof_predictions.csv \
        --swin2-predictions results/SwinUNETR-3D-Swin2/fold_0/run_*/predictions/swin2_predictions.csv \
        --fold-id 0
"""

import sys
from pathlib import Path

# Set project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
from typing import Dict
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_predictions(swin1_oof_file: Path, swin2_predictions_file: Path, fold_id: int) -> pd.DataFrame:
    """Load Swin-1 and Swin-2 predictions for the specified fold."""
    logger.info(f"Loading Swin-1 OOF predictions from: {swin1_oof_file}")
    swin1_df = pd.read_csv(swin1_oof_file)
    
    # Filter to specified fold
    swin1_fold = swin1_df[swin1_df['fold'] == fold_id].copy()
    logger.info(f"Swin-1 predictions for fold {fold_id}: {len(swin1_fold)} patients")
    
    logger.info(f"Loading Swin-2 predictions from: {swin2_predictions_file}")
    swin2_df = pd.read_csv(swin2_predictions_file)
    logger.info(f"Swin-2 predictions: {len(swin2_df)} patients")
    
    # Merge on patient_id
    merged = swin1_fold.merge(
        swin2_df[['patient_id', 'swin2_prob']],
        on='patient_id',
        how='inner'
    )
    
    if len(merged) != len(swin1_fold):
        logger.warning(f"Patient ID mismatch: Swin-1 has {len(swin1_fold)}, merged has {len(merged)}")
    
    logger.info(f"Merged predictions: {len(merged)} patients")
    
    return merged


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute classification metrics at threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_proba))
    }


def evaluate_swin2_pilot(
    swin1_oof_file: Path,
    swin2_predictions_file: Path,
    fold_id: int,
    output_dir: Path
) -> Dict:
    """Evaluate Swin-2 pilot and make GO/NO-GO decision."""
    logger.info("="*80)
    logger.info("SWIN-2 PILOT EVALUATION")
    logger.info("="*80)
    
    # Load predictions
    merged = load_predictions(swin1_oof_file, swin2_predictions_file, fold_id)
    
    y_true = merged['label'].values
    swin1_proba = merged['hgg_prob_swin'].values
    swin2_proba = merged['swin2_prob'].values
    
    # Compute metrics for Swin-1
    logger.info("\n" + "="*80)
    logger.info("SWIN-1 METRICS (Baseline)")
    logger.info("="*80)
    swin1_metrics = compute_metrics(y_true, swin1_proba, threshold=0.5)
    logger.info(f"  TN: {swin1_metrics['tn']}, FP: {swin1_metrics['fp']}, FN: {swin1_metrics['fn']}, TP: {swin1_metrics['tp']}")
    logger.info(f"  Precision: {swin1_metrics['precision']:.4f}")
    logger.info(f"  Recall: {swin1_metrics['recall']:.4f}")
    logger.info(f"  F1: {swin1_metrics['f1']:.4f}")
    logger.info(f"  AUC: {swin1_metrics['auc']:.4f}")
    
    # Compute metrics for Swin-2
    logger.info("\n" + "="*80)
    logger.info("SWIN-2 METRICS (Pilot)")
    logger.info("="*80)
    swin2_metrics = compute_metrics(y_true, swin2_proba, threshold=0.5)
    logger.info(f"  TN: {swin2_metrics['tn']}, FP: {swin2_metrics['fp']}, FN: {swin2_metrics['fn']}, TP: {swin2_metrics['tp']}")
    logger.info(f"  Precision: {swin2_metrics['precision']:.4f}")
    logger.info(f"  Recall: {swin2_metrics['recall']:.4f}")
    logger.info(f"  F1: {swin2_metrics['f1']:.4f}")
    logger.info(f"  AUC: {swin2_metrics['auc']:.4f}")
    
    # Compute FN reduction
    fn_reduction = (swin1_metrics['fn'] - swin2_metrics['fn']) / swin1_metrics['fn'] if swin1_metrics['fn'] > 0 else 0.0
    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("="*80)
    logger.info(f"  FN Reduction: {fn_reduction:.2%} ({swin1_metrics['fn']} → {swin2_metrics['fn']})")
    logger.info(f"  FP Change: {swin2_metrics['fp'] - swin1_metrics['fp']} ({swin1_metrics['fp']} → {swin2_metrics['fp']})")
    logger.info(f"  Recall Change: {swin2_metrics['recall'] - swin1_metrics['recall']:.4f} ({swin1_metrics['recall']:.4f} → {swin2_metrics['recall']:.4f})")
    logger.info(f"  Precision Change: {swin2_metrics['precision'] - swin1_metrics['precision']:.4f} ({swin1_metrics['precision']:.4f} → {swin2_metrics['precision']:.4f})")
    logger.info(f"  AUC Change: {swin2_metrics['auc'] - swin1_metrics['auc']:.4f} ({swin1_metrics['auc']:.4f} → {swin2_metrics['auc']:.4f})")
    
    # Compute correlation
    correlation, p_value = pearsonr(swin1_proba, swin2_proba)
    logger.info(f"  Correlation (Swin-1 vs Swin-2): {correlation:.4f} (p={p_value:.4e})")
    
    # GO/NO-GO decision
    logger.info("\n" + "="*80)
    logger.info("GO/NO-GO DECISION")
    logger.info("="*80)
    
    criterion1_met = fn_reduction >= 0.30
    criterion2_met = abs(correlation) < 0.70
    
    logger.info(f"  Criterion 1 (FN reduction >= 30%): {criterion1_met} ({fn_reduction:.2%})")
    logger.info(f"  Criterion 2 (Correlation < 0.70): {criterion2_met} ({correlation:.4f})")
    
    if criterion1_met and criterion2_met:
        decision = "GO"
        decision_reason = "Both criteria met: FN reduction >= 30% and correlation < 0.70"
    else:
        decision = "NO_GO"
        reasons = []
        if not criterion1_met:
            reasons.append(f"FN reduction {fn_reduction:.2%} < 30%")
        if not criterion2_met:
            reasons.append(f"Correlation {correlation:.4f} >= 0.70")
        decision_reason = "; ".join(reasons)
    
    logger.info(f"\n  DECISION: {decision}")
    logger.info(f"  REASON: {decision_reason}")
    logger.info("="*80)
    
    # Compile results
    results = {
        'fold_id': fold_id,
        'swin1_metrics': swin1_metrics,
        'swin2_metrics': swin2_metrics,
        'improvement': {
            'fn_reduction': float(fn_reduction),
            'fp_change': int(swin2_metrics['fp'] - swin1_metrics['fp']),
            'recall_change': float(swin2_metrics['recall'] - swin1_metrics['recall']),
            'precision_change': float(swin2_metrics['precision'] - swin1_metrics['precision']),
            'auc_change': float(swin2_metrics['auc'] - swin1_metrics['auc'])
        },
        'correlation': {
            'pearson_r': float(correlation),
            'p_value': float(p_value)
        },
        'decision': {
            'result': decision,
            'reason': decision_reason,
            'criterion1_met': criterion1_met,
            'criterion2_met': criterion2_met
        }
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    json_path = output_dir / f'fold_{fold_id}_pilot_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Saved JSON results to: {json_path}")
    
    # Markdown
    md_path = output_dir / f'fold_{fold_id}_pilot_metrics.md'
    with open(md_path, 'w') as f:
        f.write(f"# Swin-2 Pilot Evaluation: Fold {fold_id}\n\n")
        f.write(f"## Decision: **{decision}**\n\n")
        f.write(f"**Reason:** {decision_reason}\n\n")
        f.write("## Metrics Comparison\n\n")
        f.write("| Metric | Swin-1 | Swin-2 | Change |\n")
        f.write("|--------|--------|--------|--------|\n")
        f.write(f"| FN | {swin1_metrics['fn']} | {swin2_metrics['fn']} | {swin2_metrics['fn'] - swin1_metrics['fn']} ({fn_reduction:.2%}) |\n")
        f.write(f"| FP | {swin1_metrics['fp']} | {swin2_metrics['fp']} | {swin2_metrics['fp'] - swin1_metrics['fp']} |\n")
        f.write(f"| Recall | {swin1_metrics['recall']:.4f} | {swin2_metrics['recall']:.4f} | {swin2_metrics['recall'] - swin1_metrics['recall']:+.4f} |\n")
        f.write(f"| Precision | {swin1_metrics['precision']:.4f} | {swin2_metrics['precision']:.4f} | {swin2_metrics['precision'] - swin1_metrics['precision']:+.4f} |\n")
        f.write(f"| F1 | {swin1_metrics['f1']:.4f} | {swin2_metrics['f1']:.4f} | {swin2_metrics['f1'] - swin1_metrics['f1']:+.4f} |\n")
        f.write(f"| AUC | {swin1_metrics['auc']:.4f} | {swin2_metrics['auc']:.4f} | {swin2_metrics['auc'] - swin1_metrics['auc']:+.4f} |\n")
        f.write("\n## Decision Criteria\n\n")
        f.write(f"- **Criterion 1 (FN reduction >= 30%):** {criterion1_met} ({fn_reduction:.2%})\n")
        f.write(f"- **Criterion 2 (Correlation < 0.70):** {criterion2_met} ({correlation:.4f})\n")
        f.write(f"\n## Correlation Analysis\n\n")
        f.write(f"- Pearson correlation: {correlation:.4f}\n")
        f.write(f"- P-value: {p_value:.4e}\n")
    logger.info(f"✓ Saved Markdown report to: {md_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Swin-2 pilot experiment")
    parser.add_argument('--swin1-oof', type=str, required=True,
                       help='Path to Swin-1 OOF predictions CSV')
    parser.add_argument('--swin2-predictions', type=str, required=True,
                       help='Path to Swin-2 predictions CSV')
    parser.add_argument('--fold-id', type=int, required=True,
                       help='Fold ID used for pilot (0-4)')
    parser.add_argument('--output-dir', type=str, default='ensemble/results/swin2_pilot',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    swin1_oof_file = Path(args.swin1_oof)
    swin2_predictions_file = Path(args.swin2_predictions)
    output_dir = Path(args.output_dir)
    
    if not swin1_oof_file.exists():
        raise FileNotFoundError(f"Swin-1 OOF file not found: {swin1_oof_file}")
    if not swin2_predictions_file.exists():
        raise FileNotFoundError(f"Swin-2 predictions file not found: {swin2_predictions_file}")
    
    results = evaluate_swin2_pilot(
        swin1_oof_file,
        swin2_predictions_file,
        args.fold_id,
        output_dir
    )
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Decision: {results['decision']['result']}")
    logger.info(f"Reason: {results['decision']['reason']}")


if __name__ == '__main__':
    main()

