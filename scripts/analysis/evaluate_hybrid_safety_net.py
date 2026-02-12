#!/usr/bin/env python3
"""
Evaluate Hybrid Safety-Net: Compare Swin-1 Baseline vs Hybrid System

This script evaluates the hybrid safety-net system and provides GO/NO-GO decision.
"""

import sys
from pathlib import Path

# Set project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HYBRID_PREDICTIONS_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net' / 'hybrid_predictions.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target evaluation
FN_EXCELLENT = 25
FN_VERY_STRONG = 15
FN_RESEARCH_LEVEL = 10


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, method_name: str) -> dict:
    """Evaluate predictions and compute all metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    
    return {
        'method': method_name,
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn),
        'tp': int(tp),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc)
    }


def main():
    logger.info("="*80)
    logger.info("EVALUATE HYBRID SAFETY-NET")
    logger.info("="*80)
    
    # Load hybrid predictions
    logger.info(f"\nLoading hybrid predictions from: {HYBRID_PREDICTIONS_FILE}")
    df = pd.read_csv(HYBRID_PREDICTIONS_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    y_true = df['label'].values
    
    # Evaluate Swin-1 baseline
    logger.info("\n" + "="*80)
    logger.info("SWIN-1 BASELINE (threshold=0.5)")
    logger.info("="*80)
    swin1_pred = (df['hgg_prob_swin'] >= 0.5).astype(int)
    swin1_results = evaluate_predictions(y_true, swin1_pred, df['hgg_prob_swin'].values, 'Swin-1 Baseline')
    logger.info(f"FN: {swin1_results['fn']}, FP: {swin1_results['fp']}")
    logger.info(f"Precision: {swin1_results['precision']:.4f}, Recall: {swin1_results['recall']:.4f}")
    logger.info(f"F1: {swin1_results['f1']:.4f}, AUC: {swin1_results['auc']:.4f}")
    
    # Evaluate Hybrid System
    logger.info("\n" + "="*80)
    logger.info("HYBRID SYSTEM (Swin-1 + Safety-Net)")
    logger.info("="*80)
    hybrid_pred = df['hybrid_pred'].values
    hybrid_results = evaluate_predictions(y_true, hybrid_pred, df['hybrid_prob'].values, 'Hybrid System')
    logger.info(f"FN: {hybrid_results['fn']}, FP: {hybrid_results['fp']}")
    logger.info(f"Precision: {hybrid_results['precision']:.4f}, Recall: {hybrid_results['recall']:.4f}")
    logger.info(f"F1: {hybrid_results['f1']:.4f}, AUC: {hybrid_results['auc']:.4f}")
    
    # Comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)
    fn_reduction = swin1_results['fn'] - hybrid_results['fn']
    fp_change = hybrid_results['fp'] - swin1_results['fp']
    recall_improvement = hybrid_results['recall'] - swin1_results['recall']
    precision_change = hybrid_results['precision'] - swin1_results['precision']
    
    logger.info(f"FN Reduction: {fn_reduction} ({swin1_results['fn']} → {hybrid_results['fn']}, {fn_reduction/swin1_results['fn']*100:.1f}% reduction)")
    logger.info(f"FP Change: {fp_change:+d} ({swin1_results['fp']} → {hybrid_results['fp']})")
    logger.info(f"Recall Improvement: {recall_improvement:+.4f} ({swin1_results['recall']:.4f} → {hybrid_results['recall']:.4f})")
    logger.info(f"Precision Change: {precision_change:+.4f} ({swin1_results['precision']:.4f} → {hybrid_results['precision']:.4f})")
    
    # GO/NO-GO decision
    logger.info("\n" + "="*80)
    logger.info("GO/NO-GO DECISION")
    logger.info("="*80)
    
    # Criteria
    fn_acceptable = hybrid_results['fn'] < FN_EXCELLENT
    fn_reduction_meaningful = fn_reduction >= 5
    fp_acceptable = fp_change <= 5
    
    # FN level
    if hybrid_results['fn'] < FN_RESEARCH_LEVEL:
        fn_level = "RESEARCH-LEVEL SUCCESS"
        fn_status = "✅"
    elif hybrid_results['fn'] < FN_VERY_STRONG:
        fn_level = "VERY STRONG"
        fn_status = "✅"
    elif hybrid_results['fn'] < FN_EXCELLENT:
        fn_level = "EXCELLENT"
        fn_status = "✅"
    else:
        fn_level = "INSUFFICIENT"
        fn_status = "❌"
    
    logger.info(f"FN Status: {fn_status} {fn_level} (FN = {hybrid_results['fn']}, target: <{FN_EXCELLENT})")
    logger.info(f"FN Reduction Meaningful: {'✅' if fn_reduction_meaningful else '❌'} ({fn_reduction} reduction, need ≥5)")
    logger.info(f"FP Acceptable: {'✅' if fp_acceptable else '❌'} (FP change = {fp_change:+d}, need ≤+5)")
    
    # Overall decision
    if fn_acceptable and fn_reduction_meaningful and fp_acceptable:
        decision = "GO"
        reason = f"FN reduction is {fn_level.lower()} ({hybrid_results['fn']} FN), meaningful FN reduction ({fn_reduction} fewer FN), FP acceptable ({fp_change:+d} change)"
    else:
        decision = "NO-GO"
        reasons = []
        if not fn_acceptable:
            reasons.append(f"FN reduction insufficient ({hybrid_results['fn']} FN, target: <{FN_EXCELLENT})")
        if not fn_reduction_meaningful:
            reasons.append(f"FN reduction not meaningful ({fn_reduction} reduction, need ≥5)")
        if not fp_acceptable:
            reasons.append(f"FP increase too high ({fp_change:+d} change, need ≤+5)")
        reason = "; ".join(reasons)
    
    logger.info(f"\nDECISION: {decision}")
    logger.info(f"REASON: {reason}")
    
    # Generate comparison table
    comparison_data = [
        {
            'Method': swin1_results['method'],
            'FN': swin1_results['fn'],
            'FP': swin1_results['fp'],
            'Precision': f"{swin1_results['precision']:.4f}",
            'Recall': f"{swin1_results['recall']:.4f}",
            'F1': f"{swin1_results['f1']:.4f}",
            'AUC': f"{swin1_results['auc']:.4f}"
        },
        {
            'Method': hybrid_results['method'],
            'FN': hybrid_results['fn'],
            'FP': hybrid_results['fp'],
            'Precision': f"{hybrid_results['precision']:.4f}",
            'Recall': f"{hybrid_results['recall']:.4f}",
            'F1': f"{hybrid_results['f1']:.4f}",
            'AUC': f"{hybrid_results['auc']:.4f}"
        },
        {
            'Method': 'Improvement',
            'FN': f"{fn_reduction:+d}",
            'FP': f"{fp_change:+d}",
            'Precision': f"{precision_change:+.4f}",
            'Recall': f"{recall_improvement:+.4f}",
            'F1': f"{hybrid_results['f1'] - swin1_results['f1']:+.4f}",
            'AUC': f"{hybrid_results['auc'] - swin1_results['auc']:+.4f}"
        }
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = OUTPUT_DIR / 'comparison_table.csv'
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\n✓ Saved comparison table to: {comparison_file}")
    
    # Generate markdown report
    md_content = f"""# Hybrid Safety-Net Evaluation for Swin-1

## Objective

Implement a focused hybrid safety-net on top of Swin-1:
- Swin-1 remains the main decision maker
- Meta-decision model acts as secondary safety-net, triggered only when Swin-1 is uncertain
- Final decision: confident → Swin-1, uncertain → meta-decision

**Target Evaluation:**
- FN < {FN_RESEARCH_LEVEL} → research-level success
- FN < {FN_VERY_STRONG} → very strong
- FN < {FN_EXCELLENT} → excellent

---

## Comparison Table

| Method | FN | FP | Precision | Recall | F1 | AUC |
|--------|----|----|-----------|--------|----|-----|
| {swin1_results['method']} | {swin1_results['fn']} | {swin1_results['fp']} | {swin1_results['precision']:.4f} | {swin1_results['recall']:.4f} | {swin1_results['f1']:.4f} | {swin1_results['auc']:.4f} |
| {hybrid_results['method']} | {hybrid_results['fn']} | {hybrid_results['fp']} | {hybrid_results['precision']:.4f} | {hybrid_results['recall']:.4f} | {hybrid_results['f1']:.4f} | {hybrid_results['auc']:.4f} |
| Improvement | {fn_reduction:+d} | {fp_change:+d} | {precision_change:+.4f} | {recall_improvement:+.4f} | {hybrid_results['f1'] - swin1_results['f1']:+.4f} | {hybrid_results['auc'] - swin1_results['auc']:+.4f} |

---

## Analysis

### FN Reduction

- **Baseline FN:** {swin1_results['fn']}
- **Hybrid FN:** {hybrid_results['fn']}
- **FN Reduction:** {fn_reduction} ({fn_reduction/swin1_results['fn']*100:.1f}% reduction)
- **Status:** {fn_status} {fn_level}

### FP Control

- **Baseline FP:** {swin1_results['fp']}
- **Hybrid FP:** {hybrid_results['fp']}
- **FP Change:** {fp_change:+d}
- **Status:** {'✅ Acceptable' if fp_acceptable else '❌ Too High'}

### Overall Performance

- **Recall Improvement:** {recall_improvement:+.4f} ({swin1_results['recall']:.4f} → {hybrid_results['recall']:.4f})
- **Precision Change:** {precision_change:+.4f} ({swin1_results['precision']:.4f} → {hybrid_results['precision']:.4f})
- **F1 Improvement:** {hybrid_results['f1'] - swin1_results['f1']:+.4f} ({swin1_results['f1']:.4f} → {hybrid_results['f1']:.4f})

---

## GO/NO-GO Decision

### Decision: **{decision}**

### Reason: {reason}

### Criteria Evaluation:

1. **FN < {FN_EXCELLENT}:** {'✅' if fn_acceptable else '❌'} (FN = {hybrid_results['fn']})
2. **FN Reduction ≥ 5:** {'✅' if fn_reduction_meaningful else '❌'} (Reduction = {fn_reduction})
3. **FP Increase ≤ +5:** {'✅' if fp_acceptable else '❌'} (Change = {fp_change:+d})

---

## Conclusion

{'✅ **GO:** The hybrid safety-net provides meaningful FN reduction while keeping FP under control. The system successfully acts as a clinical safety-net, catching hard FN cases while preserving Swin-1 precision.' if decision == 'GO' else '❌ **NO-GO:** The hybrid safety-net does not meet the criteria. FN reduction is insufficient, not meaningful, or FP increase is too high.'}

### Key Findings

1. **FN Reduction:** The hybrid system reduces FN by {fn_reduction} ({fn_reduction/swin1_results['fn']*100:.1f}% reduction)
2. **FP Control:** FP {'increased' if fp_change > 0 else 'decreased' if fp_change < 0 else 'remained unchanged'} by {abs(fp_change)} ({swin1_results['fp']} → {hybrid_results['fp']})
3. **Recall Improvement:** Recall improved by {recall_improvement:+.4f} ({swin1_results['recall']:.4f} → {hybrid_results['recall']:.4f})
4. **Precision Impact:** Precision {'improved' if precision_change > 0 else 'decreased' if precision_change < 0 else 'remained unchanged'} by {abs(precision_change):.4f} ({swin1_results['precision']:.4f} → {hybrid_results['precision']:.4f})

---

*Evaluation Date: 2026-02-10*  
*Method: Hybrid Safety-Net (Swin-1 + Meta-Decision on Uncertain Samples)*  
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

