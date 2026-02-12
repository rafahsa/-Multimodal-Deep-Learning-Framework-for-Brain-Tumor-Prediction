#!/usr/bin/env python3
"""
Post-Hoc Uncertainty-Aware Decision Analysis for Swin-1

This script evaluates different decision rules on Swin-1 predictions:
1. Simple threshold tuning (0.50, 0.45, 0.40, 0.35)
2. Uncertainty-aware reject zones ([0.40-0.60], [0.35-0.65], [0.45-0.65])

Goal: Reduce FN while keeping FP under control, study precision-recall tradeoff.

NO RETRAINING - strictly post-hoc decision logic analysis.
"""

import sys
from pathlib import Path

# Set project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OOF_FILE = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'posthoc_decision_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simple_threshold_decision(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Simple threshold: if prob >= threshold → predict HGG (1), else LGG (0)."""
    return (y_proba >= threshold).astype(int)


def reject_zone_decision(y_proba: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    """
    Uncertainty-aware reject zone decision rule.
    
    if prob > upper_bound → predict HGG (1)
    if prob < lower_bound → predict LGG (0)
    if lower_bound ≤ prob ≤ upper_bound → predict HGG (1)  # clinically justified
    """
    y_pred = np.zeros_like(y_proba, dtype=int)
    y_pred[y_proba > upper_bound] = 1
    y_pred[y_proba < lower_bound] = 0
    y_pred[(y_proba >= lower_bound) & (y_proba <= upper_bound)] = 1
    return y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute all required metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1)
    }


def main():
    logger.info("="*80)
    logger.info("POST-HOC DECISION ANALYSIS FOR SWIN-1")
    logger.info("="*80)
    
    logger.info(f"\nLoading OOF predictions from: {OOF_FILE}")
    df = pd.read_csv(OOF_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    required_cols = ['patient_id', 'label', 'hgg_prob_swin']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    y_true = df['label'].values
    y_proba = df['hgg_prob_swin'].values
    
    all_results = []
    
    # Experiment 1: Simple thresholds
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 1: SIMPLE THRESHOLDS")
    logger.info("="*80)
    
    for threshold in [0.50, 0.45, 0.40, 0.35]:
        y_pred = simple_threshold_decision(y_proba, threshold)
        metrics = compute_metrics(y_true, y_pred)
        all_results.append({
            'method': 'Simple Threshold',
            'configuration': f'threshold={threshold:.2f}',
            **metrics
        })
        logger.info(f"Threshold {threshold:.2f}: Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, "
                   f"FN={metrics['fn']}, FP={metrics['fp']}")
    
    # Experiment 2: Reject zones
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 2: UNCERTAINTY-AWARE REJECT ZONES")
    logger.info("="*80)
    
    for lower, upper in [(0.40, 0.60), (0.35, 0.65), (0.45, 0.65)]:
        y_pred = reject_zone_decision(y_proba, lower, upper)
        metrics = compute_metrics(y_true, y_pred)
        all_results.append({
            'method': 'Reject Zone',
            'configuration': f'[{lower:.2f} - {upper:.2f}]',
            **metrics
        })
        logger.info(f"Reject Zone [{lower:.2f} - {upper:.2f}]: Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, "
                   f"FN={metrics['fn']}, FP={metrics['fp']}")
    
    # Generate comparison table
    comparison_table = pd.DataFrame([
        {
            'Method': r['method'],
            'Configuration': r['configuration'],
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1': f"{r['f1']:.4f}",
            'FN': r['fn'],
            'FP': r['fp']
        }
        for r in all_results
    ])
    
    csv_path = OUTPUT_DIR / 'decision_analysis_comparison.csv'
    comparison_table.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Saved comparison table to: {csv_path}")
    
    logger.info("\n" + "="*80)
    logger.info("COMPARISON TABLE")
    logger.info("="*80)
    print("\n" + comparison_table.to_string(index=False))
    
    # Analysis
    best_fn = min(all_results, key=lambda x: x['fn'])
    best_fp = min([r for r in all_results if r['recall'] >= 0.80] or all_results, key=lambda x: x['fp'])
    high_recall = [r for r in all_results if r['recall'] >= 0.90]
    best_overall = max(high_recall, key=lambda x: x['f1']) if high_recall else max(all_results, key=lambda x: x['f1'])
    
    # Generate markdown report
    md_table = "| Method | Configuration | Precision | Recall | F1 | FN | FP |\n"
    md_table += "|--------|---------------|-----------|--------|----|----|----|\n"
    for _, row in comparison_table.iterrows():
        md_table += f"| {row['Method']} | {row['Configuration']} | {row['Precision']} | {row['Recall']} | {row['F1']} | {row['FN']} | {row['FP']} |\n"
    
    md_content = f"""# Post-Hoc Decision Analysis for Swin-1

## Comparison Table

{md_table}

## Analysis

### 1. Best FN Reduction

**Method:** {best_fn['method']} with {best_fn['configuration']}

- **FN:** {best_fn['fn']}
- **FP:** {best_fn['fp']}
- **Precision:** {best_fn['precision']:.4f}
- **Recall:** {best_fn['recall']:.4f}
- **F1:** {best_fn['f1']:.4f}

### 2. Best FP Control

**Method:** {best_fp['method']} with {best_fp['configuration']}

- **FN:** {best_fp['fn']}
- **FP:** {best_fp['fp']}
- **Precision:** {best_fp['precision']:.4f}
- **Recall:** {best_fp['recall']:.4f}
- **F1:** {best_fp['f1']:.4f}

### 3. Best Overall Tradeoff

**Method:** {best_overall['method']} with {best_overall['configuration']}

- **FN:** {best_overall['fn']}
- **FP:** {best_overall['fp']}
- **Precision:** {best_overall['precision']:.4f}
- **Recall:** {best_overall['recall']:.4f}
- **F1:** {best_overall['f1']:.4f}

## Written Analysis

### Which configuration gives the best FN reduction?

The {best_fn['method']} with {best_fn['configuration']} achieves the lowest FN count ({best_fn['fn']}). This configuration prioritizes recall ({best_fn['recall']:.4f}) over precision ({best_fn['precision']:.4f}), resulting in {best_fn['fp']} false positives.

### Which configuration keeps FP under control?

The {best_fp['method']} with {best_fp['configuration']} achieves the best FP control ({best_fp['fp']} FP) while maintaining reasonable recall ({best_fp['recall']:.4f}). This configuration has {best_fp['fn']} false negatives and precision of {best_fp['precision']:.4f}.

### Which configuration offers the best overall tradeoff?

The {best_overall['method']} with {best_overall['configuration']} offers the best overall tradeoff. It achieves:
- **FN:** {best_overall['fn']}
- **FP:** {best_overall['fp']}
- **Precision:** {best_overall['precision']:.4f}
- **Recall:** {best_overall['recall']:.4f}
- **F1:** {best_overall['f1']:.4f}

{'✅ **This configuration achieves strong recall (≥90%) while maintaining reasonable precision.**' if best_overall['recall'] >= 0.90 else '⚠️ **This configuration does not reach 90% recall. Consider if this meets clinical requirements.**'}

### Final Recommendation

**Selected Decision Rule:** {best_overall['method']} with {best_overall['configuration']}

**Rationale:**
- Minimizes FN as much as possible ({best_overall['fn']} FN)
- Keeps FP reasonably low ({best_overall['fp']} FP)
- Achieves {'strong' if best_overall['recall'] >= 0.90 else 'reasonable'} recall ({best_overall['recall']:.4f}) {'without destroying' if best_overall['recall'] >= 0.90 else 'with'} precision ({best_overall['precision']:.4f})
- Best overall F1-score ({best_overall['f1']:.4f})

---
*Analysis Date: 2026-02-10*
*Model: Swin-1 (no retraining, post-hoc decision logic only)*
"""
    
    md_path = OUTPUT_DIR / 'decision_analysis_report.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    logger.info(f"\n✓ Saved markdown report to: {md_path}")
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

