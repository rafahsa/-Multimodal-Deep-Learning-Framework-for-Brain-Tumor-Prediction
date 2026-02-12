#!/usr/bin/env python3
"""
Evaluate Meta-Decision Layer: Compare against Swin-1 Baseline

This script compares the meta-decision layer results against Swin-1 baseline
and provides a clear GO/NO-GO recommendation.
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
from typing import Dict
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OOF_FILE = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
META_PREDICTIONS_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_decision' / 'meta_decision_predictions.csv'
META_RESULTS_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_decision' / 'meta_decision_results.json'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_decision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target evaluation
FN_EXCELLENT = 25
FN_VERY_STRONG = 15
FN_RESEARCH_LEVEL = 10


def evaluate_baseline(df: pd.DataFrame) -> Dict:
    """Evaluate Swin-1 baseline (threshold=0.5)."""
    y_true = df['label'].values
    y_proba = df['hgg_prob_swin'].values
    y_pred = (y_proba >= 0.5).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'method': 'Swin-1 Baseline',
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn),
        'tp': int(tp),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_proba))
    }


def evaluate_meta_decision(df: pd.DataFrame) -> Dict:
    """Evaluate meta-decision layer."""
    y_true = df['label'].values
    y_proba = df['meta_prob'].values
    y_pred = df['meta_pred'].values
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'method': 'Swin-1 + Meta-Decision',
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn),
        'tp': int(tp),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_proba))
    }


def main():
    logger.info("="*80)
    logger.info("EVALUATE META-DECISION LAYER")
    logger.info("="*80)
    
    # Load OOF predictions (baseline)
    logger.info(f"\nLoading OOF predictions from: {OOF_FILE}")
    oof_df = pd.read_csv(OOF_FILE)
    
    # Load meta-decision predictions
    logger.info(f"Loading meta-decision predictions from: {META_PREDICTIONS_FILE}")
    meta_df = pd.read_csv(META_PREDICTIONS_FILE)
    
    # Merge
    df = oof_df.merge(meta_df[['patient_id', 'meta_prob', 'meta_pred']], on='patient_id', how='inner')
    
    # Evaluate baseline
    logger.info("\n" + "="*80)
    logger.info("SWIN-1 BASELINE (threshold=0.5)")
    logger.info("="*80)
    baseline_results = evaluate_baseline(df)
    logger.info(f"FN: {baseline_results['fn']}, FP: {baseline_results['fp']}")
    logger.info(f"Precision: {baseline_results['precision']:.4f}, Recall: {baseline_results['recall']:.4f}")
    logger.info(f"F1: {baseline_results['f1']:.4f}, AUC: {baseline_results['auc']:.4f}")
    
    # Evaluate meta-decision
    logger.info("\n" + "="*80)
    logger.info("SWIN-1 + META-DECISION")
    logger.info("="*80)
    meta_results = evaluate_meta_decision(df)
    logger.info(f"FN: {meta_results['fn']}, FP: {meta_results['fp']}")
    logger.info(f"Precision: {meta_results['precision']:.4f}, Recall: {meta_results['recall']:.4f}")
    logger.info(f"F1: {meta_results['f1']:.4f}, AUC: {meta_results['auc']:.4f}")
    
    # Comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)
    fn_reduction = baseline_results['fn'] - meta_results['fn']
    fp_change = meta_results['fp'] - baseline_results['fp']
    recall_improvement = meta_results['recall'] - baseline_results['recall']
    precision_change = meta_results['precision'] - baseline_results['precision']
    
    logger.info(f"FN Reduction: {fn_reduction} ({baseline_results['fn']} → {meta_results['fn']})")
    logger.info(f"FP Change: {fp_change:+d} ({baseline_results['fp']} → {meta_results['fp']})")
    logger.info(f"Recall Improvement: {recall_improvement:+.4f} ({baseline_results['recall']:.4f} → {meta_results['recall']:.4f})")
    logger.info(f"Precision Change: {precision_change:+.4f} ({baseline_results['precision']:.4f} → {meta_results['precision']:.4f})")
    
    # GO/NO-GO decision
    logger.info("\n" + "="*80)
    logger.info("GO/NO-GO DECISION")
    logger.info("="*80)
    
    # Evaluate targets
    fn_level = None
    if meta_results['fn'] < FN_RESEARCH_LEVEL:
        fn_level = "RESEARCH-LEVEL SUCCESS"
        fn_status = "✅"
    elif meta_results['fn'] < FN_VERY_STRONG:
        fn_level = "VERY STRONG"
        fn_status = "✅"
    elif meta_results['fn'] < FN_EXCELLENT:
        fn_level = "EXCELLENT"
        fn_status = "✅"
    else:
        fn_level = "INSUFFICIENT"
        fn_status = "❌"
    
    # Check if FP is acceptable (not too high)
    fp_acceptable = meta_results['fp'] <= baseline_results['fp'] + 5  # Allow small increase
    
    # Check if FN reduction is meaningful
    fn_reduction_meaningful = fn_reduction >= 5  # At least 5 FN reduction
    
    # Overall decision
    if fn_status == "✅" and fp_acceptable and fn_reduction_meaningful:
        decision = "GO"
        reason = f"FN reduction is {fn_level} ({meta_results['fn']} FN), FP is acceptable ({meta_results['fp']} FP), meaningful FN reduction ({fn_reduction} fewer FN)"
    else:
        decision = "NO-GO"
        reasons = []
        if fn_status == "❌":
            reasons.append(f"FN reduction insufficient ({meta_results['fn']} FN, target: <{FN_EXCELLENT})")
        if not fp_acceptable:
            reasons.append(f"FP too high ({meta_results['fp']} FP, baseline: {baseline_results['fp']})")
        if not fn_reduction_meaningful:
            reasons.append(f"FN reduction not meaningful ({fn_reduction} reduction, need ≥5)")
        reason = "; ".join(reasons)
    
    logger.info(f"\nFN Status: {fn_status} {fn_level} (FN = {meta_results['fn']})")
    logger.info(f"FP Acceptable: {'✅' if fp_acceptable else '❌'} (FP = {meta_results['fp']})")
    logger.info(f"FN Reduction Meaningful: {'✅' if fn_reduction_meaningful else '❌'} ({fn_reduction} reduction)")
    logger.info(f"\nDECISION: {decision}")
    logger.info(f"REASON: {reason}")
    
    # Generate comparison table
    comparison_data = [
        {
            'Method': baseline_results['method'],
            'FN': baseline_results['fn'],
            'FP': baseline_results['fp'],
            'Precision': f"{baseline_results['precision']:.4f}",
            'Recall': f"{baseline_results['recall']:.4f}",
            'F1': f"{baseline_results['f1']:.4f}",
            'AUC': f"{baseline_results['auc']:.4f}"
        },
        {
            'Method': meta_results['method'],
            'FN': meta_results['fn'],
            'FP': meta_results['fp'],
            'Precision': f"{meta_results['precision']:.4f}",
            'Recall': f"{meta_results['recall']:.4f}",
            'F1': f"{meta_results['f1']:.4f}",
            'AUC': f"{meta_results['auc']:.4f}"
        },
        {
            'Method': 'Improvement',
            'FN': f"{fn_reduction:+d}",
            'FP': f"{fp_change:+d}",
            'Precision': f"{precision_change:+.4f}",
            'Recall': f"{recall_improvement:+.4f}",
            'F1': f"{meta_results['f1'] - baseline_results['f1']:+.4f}",
            'AUC': f"{meta_results['auc'] - baseline_results['auc']:+.4f}"
        }
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = OUTPUT_DIR / 'comparison_table.csv'
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\n✓ Saved comparison table to: {comparison_file}")
    
    # Generate markdown report
    md_content = f"""# Meta-Decision Layer Evaluation for Swin-1

## Objective

Implement a lightweight meta-decision layer on top of Swin-1 to reduce False Negatives (FN) while keeping False Positives (FP) under control.

**Target Evaluation:**
- FN < {FN_RESEARCH_LEVEL} → research-level success
- FN < {FN_VERY_STRONG} → very strong
- FN < {FN_EXCELLENT} → excellent

---

## Comparison Table

| Method | FN | FP | Precision | Recall | F1 | AUC |
|--------|----|----|-----------|--------|----|-----|
| {baseline_results['method']} | {baseline_results['fn']} | {baseline_results['fp']} | {baseline_results['precision']:.4f} | {baseline_results['recall']:.4f} | {baseline_results['f1']:.4f} | {baseline_results['auc']:.4f} |
| {meta_results['method']} | {meta_results['fn']} | {meta_results['fp']} | {meta_results['precision']:.4f} | {meta_results['recall']:.4f} | {meta_results['f1']:.4f} | {meta_results['auc']:.4f} |
| Improvement | {fn_reduction:+d} | {fp_change:+d} | {precision_change:+.4f} | {recall_improvement:+.4f} | {meta_results['f1'] - baseline_results['f1']:+.4f} | {meta_results['auc'] - baseline_results['auc']:+.4f} |

---

## Analysis

### FN Reduction

- **Baseline FN:** {baseline_results['fn']}
- **Meta-Decision FN:** {meta_results['fn']}
- **FN Reduction:** {fn_reduction} ({fn_reduction/baseline_results['fn']*100:.1f}% reduction)
- **Status:** {fn_status} {fn_level}

### FP Control

- **Baseline FP:** {baseline_results['fp']}
- **Meta-Decision FP:** {meta_results['fp']}
- **FP Change:** {fp_change:+d}
- **Status:** {'✅ Acceptable' if fp_acceptable else '❌ Too High'}

### Overall Performance

- **Recall Improvement:** {recall_improvement:+.4f} ({baseline_results['recall']:.4f} → {meta_results['recall']:.4f})
- **Precision Change:** {precision_change:+.4f} ({baseline_results['precision']:.4f} → {meta_results['precision']:.4f})
- **F1 Improvement:** {meta_results['f1'] - baseline_results['f1']:+.4f} ({baseline_results['f1']:.4f} → {meta_results['f1']:.4f})

---

## GO/NO-GO Decision

### Decision: **{decision}**

### Reason: {reason}

### Criteria Evaluation:

1. **FN Reduction:** {fn_status} {fn_level} (FN = {meta_results['fn']}, target: <{FN_EXCELLENT})
2. **FP Acceptable:** {'✅' if fp_acceptable else '❌'} (FP = {meta_results['fp']}, baseline: {baseline_results['fp']})
3. **FN Reduction Meaningful:** {'✅' if fn_reduction_meaningful else '❌'} (Reduction = {fn_reduction}, need ≥5)

---

## Conclusion

{'✅ **GO:** The meta-decision layer provides meaningful FN reduction while keeping FP under control. Proceed with deployment.' if decision == 'GO' else '❌ **NO-GO:** The meta-decision layer does not meet the criteria. FN reduction is insufficient or FP is too high.'}

---

*Evaluation Date: 2026-02-10*  
*Method: Lightweight Logistic Regression meta-decision layer*  
*No deep learning training or Swin-1 modification*
"""
    
    md_file = OUTPUT_DIR / 'evaluation_report.md'
    with open(md_file, 'w') as f:
        f.write(md_content)
    logger.info(f"\n✓ Saved evaluation report to: {md_file}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

