#!/usr/bin/env python3
"""
Evaluate Rule Override vs Baseline

This script evaluates the rule-based override predictions against baseline,
computing confusion matrices and metrics for both.

NO RETRAINING - strictly post-hoc evaluation.
"""

import sys
from pathlib import Path

# Set project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RULE_OVERRIDE_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net' / 'rule_override_predictions.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """
    Compute classification metrics.
    
    Returns:
        Dictionary with metrics: FN, FP, TP, TN, Precision, Recall, F1, Accuracy
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = {
        'name': name,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy
    }
    
    return metrics


def print_metrics(metrics: dict):
    """Print metrics in a formatted way."""
    logger.info(f"\n{metrics['name']} Metrics:")
    logger.info(f"  TP: {metrics['TP']}")
    logger.info(f"  TN: {metrics['TN']}")
    logger.info(f"  FP: {metrics['FP']}")
    logger.info(f"  FN: {metrics['FN']}")
    logger.info(f"  Precision: {metrics['Precision']:.4f}")
    logger.info(f"  Recall: {metrics['Recall']:.4f}")
    logger.info(f"  F1: {metrics['F1']:.4f}")
    logger.info(f"  Accuracy: {metrics['Accuracy']:.4f}")


def generate_markdown_report(baseline_metrics: dict, final_metrics: dict) -> str:
    """Generate a markdown report comparing baseline and final metrics."""
    
    fn_reduction = baseline_metrics['FN'] - final_metrics['FN']
    fp_change = final_metrics['FP'] - baseline_metrics['FP']
    fn_reduction_pct = (fn_reduction / baseline_metrics['FN'] * 100) if baseline_metrics['FN'] > 0 else 0
    fp_change_pct = (fp_change / baseline_metrics['FP'] * 100) if baseline_metrics['FP'] > 0 else 0
    
    report = f"""# Rule Override Evaluation Report

## Summary

This report compares baseline Swin-1 predictions with rule-based override predictions.

### Key Changes

- **FN Reduction**: {fn_reduction} ({fn_reduction_pct:+.1f}%)
- **FP Change**: {fp_change:+d} ({fp_change_pct:+.1f}%)

## Metrics Comparison

| Metric | Baseline | Final | Change |
|--------|----------|-------|--------|
| **TP** | {baseline_metrics['TP']} | {final_metrics['TP']} | {final_metrics['TP'] - baseline_metrics['TP']:+d} |
| **TN** | {baseline_metrics['TN']} | {final_metrics['TN']} | {final_metrics['TN'] - baseline_metrics['TN']:+d} |
| **FP** | {baseline_metrics['FP']} | {final_metrics['FP']} | {fp_change:+d} |
| **FN** | {baseline_metrics['FN']} | {final_metrics['FN']} | {-fn_reduction:+d} |
| **Precision** | {baseline_metrics['Precision']:.4f} | {final_metrics['Precision']:.4f} | {final_metrics['Precision'] - baseline_metrics['Precision']:+.4f} |
| **Recall** | {baseline_metrics['Recall']:.4f} | {final_metrics['Recall']:.4f} | {final_metrics['Recall'] - baseline_metrics['Recall']:+.4f} |
| **F1** | {baseline_metrics['F1']:.4f} | {final_metrics['F1']:.4f} | {final_metrics['F1'] - baseline_metrics['F1']:+.4f} |
| **Accuracy** | {baseline_metrics['Accuracy']:.4f} | {final_metrics['Accuracy']:.4f} | {final_metrics['Accuracy'] - baseline_metrics['Accuracy']:+.4f} |

## Baseline Confusion Matrix

```
                Predicted
              LGG    HGG
Actual LGG    {baseline_metrics['TN']:4d}    {baseline_metrics['FP']:4d}
      HGG     {baseline_metrics['FN']:4d}    {baseline_metrics['TP']:4d}
```

## Final Confusion Matrix

```
                Predicted
              LGG    HGG
Actual LGG    {final_metrics['TN']:4d}    {final_metrics['FP']:4d}
      HGG     {final_metrics['FN']:4d}    {final_metrics['TP']:4d}
```

## Interpretation

- **False Negative Reduction**: The rule override reduced false negatives by {fn_reduction} ({fn_reduction_pct:+.1f}%), which means {fn_reduction} more HGG cases were correctly identified.
- **False Positive Impact**: The rule override {'increased' if fp_change > 0 else 'decreased' if fp_change < 0 else 'did not change'} false positives by {abs(fp_change)} ({abs(fp_change_pct):+.1f}%).
- **Overall Performance**: {'Improved' if final_metrics['F1'] > baseline_metrics['F1'] else 'Worsened' if final_metrics['F1'] < baseline_metrics['F1'] else 'Unchanged'} F1 score from {baseline_metrics['F1']:.4f} to {final_metrics['F1']:.4f}.

## Notes

- The rule override only applies to uncertain LGG predictions (baseline_pred==0 AND uncertainty_status=="uncertain").
- Confident samples always keep their baseline predictions.
- Swin-1 threshold remains fixed at 0.5 for baseline predictions.
"""
    
    return report


def main():
    logger.info("="*80)
    logger.info("EVALUATE RULE OVERRIDE VS BASELINE")
    logger.info("="*80)
    
    # Load rule override predictions
    logger.info(f"\nLoading rule override predictions from: {RULE_OVERRIDE_FILE}")
    df = pd.read_csv(RULE_OVERRIDE_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    # Extract ground truth and predictions
    y_true = df['label'].values
    y_baseline = df['baseline_pred'].values
    y_final = df['final_pred'].values
    
    # Compute metrics for baseline
    logger.info("\n" + "="*80)
    logger.info("BASELINE METRICS")
    logger.info("="*80)
    baseline_metrics = compute_metrics(y_true, y_baseline, "Baseline")
    print_metrics(baseline_metrics)
    
    # Compute metrics for final
    logger.info("\n" + "="*80)
    logger.info("FINAL METRICS (WITH RULE OVERRIDE)")
    logger.info("="*80)
    final_metrics = compute_metrics(y_true, y_final, "Final")
    print_metrics(final_metrics)
    
    # Compare metrics
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)
    
    fn_reduction = baseline_metrics['FN'] - final_metrics['FN']
    fp_change = final_metrics['FP'] - baseline_metrics['FP']
    
    logger.info(f"\nFN Reduction: {fn_reduction} ({fn_reduction/baseline_metrics['FN']*100:+.1f}%)")
    logger.info(f"FP Change: {fp_change:+d} ({fp_change/baseline_metrics['FP']*100:+.1f}%)")
    logger.info(f"Precision Change: {final_metrics['Precision'] - baseline_metrics['Precision']:+.4f}")
    logger.info(f"Recall Change: {final_metrics['Recall'] - baseline_metrics['Recall']:+.4f}")
    logger.info(f"F1 Change: {final_metrics['F1'] - baseline_metrics['F1']:+.4f}")
    logger.info(f"Accuracy Change: {final_metrics['Accuracy'] - baseline_metrics['Accuracy']:+.4f}")
    
    # Generate and save markdown report
    report = generate_markdown_report(baseline_metrics, final_metrics)
    report_file = OUTPUT_DIR / 'rule_override_report.md'
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"\n✓ Saved evaluation report to: {report_file}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

