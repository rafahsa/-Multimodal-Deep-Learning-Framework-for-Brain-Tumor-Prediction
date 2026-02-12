"""
Comprehensive comparison between OLD MIL (DualStreamMIL-3D) and NEW MIL (entropy-based sampling)
using OOF predictions for patient-level 5-fold CV.

This script:
1. Validates OOF integrity for both models
2. Computes comprehensive metrics (ROC-AUC, PR-AUC, Recall, Precision, F1, FN count)
3. Analyzes signal quality (probability distributions, calibration)
4. Provides recommendation on whether to replace OLD with NEW MIL
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# File paths
OLD_MIL_PATH = Path('ensemble/oof_predictions/dualstream_mil_3d_oof.csv')
NEW_MIL_PATH = Path('ensemble/results/mil_improvements/exp_1_1_entropy/oof_predictions.csv')
OUTPUT_DIR = Path('ensemble/results/mil_improvements/comparison_report')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_and_validate_oof(file_path: Path, prob_col: str, model_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load OOF predictions and validate integrity.
    
    Returns:
        DataFrame and validation report dictionary
    """
    print(f"\n{'='*60}")
    print(f"Loading and validating {model_name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(file_path)
    report = {
        'model_name': model_name,
        'file_path': str(file_path),
        'num_patients': len(df),
        'num_folds': df['fold'].nunique(),
        'has_nans': df[prob_col].isna().any(),
        'prob_range': (df[prob_col].min(), df[prob_col].max()),
        'unique_patients': df['patient_id'].nunique(),
        'duplicate_patients': df['patient_id'].duplicated().any(),
        'label_distribution': df['label'].value_counts().to_dict(),
        'fold_distribution': df['fold'].value_counts().sort_index().to_dict(),
        'validation_passed': True,
        'validation_errors': []
    }
    
    # Validation checks
    if report['num_patients'] != report['unique_patients']:
        report['validation_passed'] = False
        report['validation_errors'].append(f"Duplicate patients found: {report['num_patients']} rows but {report['unique_patients']} unique patients")
    
    if report['has_nans']:
        report['validation_passed'] = False
        report['validation_errors'].append(f"NaN values found in {prob_col}")
    
    if report['num_folds'] != 5:
        report['validation_errors'].append(f"Expected 5 folds, found {report['num_folds']}")
    
    if report['prob_range'][0] < 0 or report['prob_range'][1] > 1:
        report['validation_passed'] = False
        report['validation_errors'].append(f"Probabilities outside [0,1]: {report['prob_range']}")
    
    # Print validation report
    print(f"✓ Patients: {report['num_patients']}")
    print(f"✓ Unique patients: {report['unique_patients']}")
    print(f"✓ Folds: {report['num_folds']}")
    print(f"✓ Probability range: [{report['prob_range'][0]:.4f}, {report['prob_range'][1]:.4f}]")
    print(f"✓ Label distribution: {report['label_distribution']}")
    print(f"✓ Has NaNs: {report['has_nans']}")
    print(f"✓ Duplicate patients: {report['duplicate_patients']}")
    
    if report['validation_errors']:
        print(f"\n⚠ Validation Warnings:")
        for error in report['validation_errors']:
            print(f"  - {error}")
    else:
        print(f"\n✓ All validation checks passed!")
    
    return df, report


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute comprehensive metrics at given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        metrics['fn_count'] = int(fn)  # Critical for medical use
        metrics['fp_count'] = int(fp)
    else:
        # Edge case: only one class predicted
        metrics['tn'] = 0
        metrics['fp'] = 0
        metrics['fn'] = 0
        metrics['tp'] = 0
        metrics['fn_count'] = 0
        metrics['fp_count'] = 0
    
    return metrics


def analyze_probability_distribution(y_proba: np.ndarray, y_true: np.ndarray, model_name: str) -> Dict:
    """Analyze probability distribution and calibration."""
    stats = {
        'mean': float(np.mean(y_proba)),
        'std': float(np.std(y_proba)),
        'min': float(np.min(y_proba)),
        'max': float(np.max(y_proba)),
        'median': float(np.median(y_proba)),
        'q25': float(np.percentile(y_proba, 25)),
        'q75': float(np.percentile(y_proba, 75)),
        'iqr': float(np.percentile(y_proba, 75) - np.percentile(y_proba, 25)),
    }
    
    # Class-specific statistics
    hgg_probs = y_proba[y_true == 1]
    lgg_probs = y_proba[y_true == 0]
    
    stats['hgg_mean'] = float(np.mean(hgg_probs)) if len(hgg_probs) > 0 else 0.0
    stats['hgg_std'] = float(np.std(hgg_probs)) if len(hgg_probs) > 0 else 0.0
    stats['lgg_mean'] = float(np.mean(lgg_probs)) if len(lgg_probs) > 0 else 0.0
    stats['lgg_std'] = float(np.std(lgg_probs)) if len(lgg_probs) > 0 else 0.0
    
    # Separation quality (how well probabilities separate classes)
    stats['separation'] = float(stats['hgg_mean'] - stats['lgg_mean'])
    
    # Confidence on true positives (HGG cases)
    stats['hgg_high_confidence'] = int(np.sum(hgg_probs >= 0.7)) if len(hgg_probs) > 0 else 0
    stats['hgg_low_confidence'] = int(np.sum(hgg_probs < 0.5)) if len(hgg_probs) > 0 else 0
    
    # Collapse indicator (if too many predictions are in narrow range)
    prob_range = stats['max'] - stats['min']
    stats['range_width'] = prob_range
    stats['is_collapsed'] = prob_range < 0.3  # Heuristic: very narrow range suggests collapse
    
    return stats


def compare_models(old_df: pd.DataFrame, new_df: pd.DataFrame, 
                   old_prob_col: str, new_prob_col: str) -> Dict:
    """Comprehensive comparison between old and new MIL models."""
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE MODEL COMPARISON")
    print(f"{'='*60}")
    
    # Ensure same patients and order
    merged = pd.merge(
        old_df[['patient_id', 'fold', 'label']].rename(columns={}),
        old_df[[old_prob_col]].rename(columns={old_prob_col: 'old_prob'}),
        left_index=True, right_index=True
    )
    merged = pd.merge(
        merged,
        new_df[['patient_id', new_prob_col]].rename(columns={new_prob_col: 'new_prob'}),
        on='patient_id',
        how='inner'
    )
    
    if len(merged) != len(old_df):
        print(f"⚠ Warning: Patient mismatch. Old: {len(old_df)}, Merged: {len(merged)}")
    
    y_true = merged['label'].values
    old_proba = merged['old_prob'].values
    new_proba = merged['new_prob'].values
    
    # Compute metrics for both models
    print("\n" + "-"*60)
    print("METRICS @ Threshold 0.5")
    print("-"*60)
    
    old_metrics = compute_metrics(y_true, old_proba, threshold=0.5)
    new_metrics = compute_metrics(y_true, new_proba, threshold=0.5)
    
    # Print comparison table
    metric_names = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'fn_count']
    print(f"\n{'Metric':<20} {'OLD MIL':<15} {'NEW MIL':<15} {'Delta':<15} {'Winner':<10}")
    print("-" * 75)
    for metric in metric_names:
        old_val = old_metrics[metric]
        new_val = new_metrics[metric]
        delta = new_val - old_val
        winner = "NEW" if (delta > 0 and metric != 'fn_count') or (delta < 0 and metric == 'fn_count') else "OLD"
        if metric in ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1']:
            print(f"{metric:<20} {old_val:<15.4f} {new_val:<15.4f} {delta:+.4f}        {winner:<10}")
        else:
            print(f"{metric:<20} {old_val:<15} {new_val:<15} {delta:+d}        {winner:<10}")
    
    # Probability distribution analysis
    print("\n" + "-"*60)
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print("-"*60)
    
    old_dist = analyze_probability_distribution(old_proba, y_true, "OLD MIL")
    new_dist = analyze_probability_distribution(new_proba, y_true, "NEW MIL")
    
    print(f"\n{'Statistic':<25} {'OLD MIL':<20} {'NEW MIL':<20} {'Delta':<15}")
    print("-" * 80)
    dist_stats = ['mean', 'std', 'min', 'max', 'range_width', 'separation', 'hgg_mean', 'lgg_mean']
    for stat in dist_stats:
        old_val = old_dist[stat]
        new_val = new_dist[stat]
        delta = new_val - old_val
        print(f"{stat:<25} {old_val:<20.4f} {new_val:<20.4f} {delta:+.4f}")
    
    # Signal quality indicators
    print("\n" + "-"*60)
    print("SIGNAL QUALITY INDICATORS")
    print("-"*60)
    
    print(f"\nOLD MIL:")
    print(f"  - Is collapsed: {old_dist['is_collapsed']}")
    print(f"  - Range width: {old_dist['range_width']:.4f}")
    print(f"  - HGG high confidence (≥0.7): {old_dist['hgg_high_confidence']}/{np.sum(y_true==1)}")
    print(f"  - HGG low confidence (<0.5): {old_dist['hgg_low_confidence']}/{np.sum(y_true==1)}")
    
    print(f"\nNEW MIL:")
    print(f"  - Is collapsed: {new_dist['is_collapsed']}")
    print(f"  - Range width: {new_dist['range_width']:.4f}")
    print(f"  - HGG high confidence (≥0.7): {new_dist['hgg_high_confidence']}/{np.sum(y_true==1)}")
    print(f"  - HGG low confidence (<0.5): {new_dist['hgg_low_confidence']}/{np.sum(y_true==1)}")
    
    # Compile full comparison
    comparison = {
        'old_metrics': old_metrics,
        'new_metrics': new_metrics,
        'old_distribution': old_dist,
        'new_distribution': new_dist,
        'merged_data': merged,
        'y_true': y_true,
        'old_proba': old_proba,
        'new_proba': new_proba
    }
    
    return comparison


def create_visualizations(comparison: Dict, output_dir: Path):
    """Create comprehensive visualization plots."""
    
    y_true = comparison['y_true']
    old_proba = comparison['old_proba']
    new_proba = comparison['new_proba']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('OLD vs NEW MIL Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. ROC Curves
    ax = axes[0, 0]
    fpr_old, tpr_old, _ = roc_curve(y_true, old_proba)
    fpr_new, tpr_new, _ = roc_curve(y_true, new_proba)
    ax.plot(fpr_old, tpr_old, label=f"OLD MIL (AUC={comparison['old_metrics']['roc_auc']:.3f})", linewidth=2)
    ax.plot(fpr_new, tpr_new, label=f"NEW MIL (AUC={comparison['new_metrics']['roc_auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. PR Curves
    ax = axes[0, 1]
    prec_old, rec_old, _ = precision_recall_curve(y_true, old_proba)
    prec_new, rec_new, _ = precision_recall_curve(y_true, new_proba)
    ax.plot(rec_old, prec_old, label=f"OLD MIL (AP={comparison['old_metrics']['pr_auc']:.3f})", linewidth=2)
    ax.plot(rec_new, prec_new, label=f"NEW MIL (AP={comparison['new_metrics']['pr_auc']:.3f})", linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Probability Distributions (Histograms)
    ax = axes[0, 2]
    ax.hist(old_proba[y_true==0], bins=30, alpha=0.6, label='OLD MIL - LGG', color='blue', density=True)
    ax.hist(old_proba[y_true==1], bins=30, alpha=0.6, label='OLD MIL - HGG', color='red', density=True)
    ax.set_xlabel('Probability')
    ax.set_ylabel('Density')
    ax.set_title('OLD MIL: Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Probability Distributions (NEW)
    ax = axes[1, 0]
    ax.hist(new_proba[y_true==0], bins=30, alpha=0.6, label='NEW MIL - LGG', color='blue', density=True)
    ax.hist(new_proba[y_true==1], bins=30, alpha=0.6, label='NEW MIL - HGG', color='red', density=True)
    ax.set_xlabel('Probability')
    ax.set_ylabel('Density')
    ax.set_title('NEW MIL: Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Scatter: Old vs New Probabilities
    ax = axes[1, 1]
    scatter = ax.scatter(old_proba, new_proba, c=y_true, cmap='RdYlGn', alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('OLD MIL Probability')
    ax.set_ylabel('NEW MIL Probability')
    ax.set_title('Probability Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Label (0=LGG, 1=HGG)')
    
    # 6. Box plots by class
    ax = axes[1, 2]
    data_to_plot = [
        old_proba[y_true==0], old_proba[y_true==1],
        new_proba[y_true==0], new_proba[y_true==1]
    ]
    labels = ['OLD\nLGG', 'OLD\nHGG', 'NEW\nLGG', 'NEW\nHGG']
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Probability')
    ax.set_title('Probability Distribution by Class')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_dir / 'comparison_plots.png'}")
    plt.close()


def generate_recommendation(comparison: Dict) -> Dict:
    """Generate recommendation based on comprehensive analysis."""
    
    old_metrics = comparison['old_metrics']
    new_metrics = comparison['new_metrics']
    old_dist = comparison['old_distribution']
    new_dist = comparison['new_distribution']
    
    recommendation = {
        'decision': 'CONDITIONAL',  # YES, NO, or CONDITIONAL
        'confidence': 'medium',  # high, medium, low
        'reasoning': [],
        'improvements_needed': [],
        'statistical_significance': 'unknown'
    }
    
    # Key improvements
    roc_auc_delta = new_metrics['roc_auc'] - old_metrics['roc_auc']
    pr_auc_delta = new_metrics['pr_auc'] - old_metrics['pr_auc']
    recall_delta = new_metrics['recall'] - old_metrics['recall']
    fn_delta = new_metrics['fn_count'] - old_metrics['fn_count']
    separation_delta = new_dist['separation'] - old_dist['separation']
    range_delta = new_dist['range_width'] - old_dist['range_width']
    
    # Decision logic
    improvements = []
    concerns = []
    
    # ROC-AUC improvement
    if roc_auc_delta > 0.01:
        improvements.append(f"ROC-AUC improved by {roc_auc_delta:.4f} ({old_metrics['roc_auc']:.4f} → {new_metrics['roc_auc']:.4f})")
    elif roc_auc_delta < -0.01:
        concerns.append(f"ROC-AUC decreased by {abs(roc_auc_delta):.4f}")
    
    # PR-AUC improvement
    if pr_auc_delta > 0.01:
        improvements.append(f"PR-AUC improved by {pr_auc_delta:.4f} ({old_metrics['pr_auc']:.4f} → {new_metrics['pr_auc']:.4f})")
    elif pr_auc_delta < -0.01:
        concerns.append(f"PR-AUC decreased by {abs(pr_auc_delta):.4f}")
    
    # Recall improvement (critical for medical)
    if recall_delta > 0.02:
        improvements.append(f"Recall improved by {recall_delta:.4f} ({old_metrics['recall']:.4f} → {new_metrics['recall']:.4f})")
    elif recall_delta < -0.02:
        concerns.append(f"Recall decreased by {abs(recall_delta):.4f} (CRITICAL for medical use)")
    
    # False negatives (critical)
    if fn_delta < 0:
        improvements.append(f"False negatives REDUCED by {abs(fn_delta)} ({old_metrics['fn_count']} → {new_metrics['fn_count']})")
    elif fn_delta > 0:
        concerns.append(f"False negatives INCREASED by {fn_delta} (CRITICAL concern)")
    
    # Signal quality
    if separation_delta > 0.05:
        improvements.append(f"Class separation improved by {separation_delta:.4f} (better signal quality)")
    elif separation_delta < -0.05:
        concerns.append(f"Class separation decreased by {abs(separation_delta):.4f}")
    
    if range_delta > 0.1:
        improvements.append(f"Probability range expanded by {range_delta:.4f} (less collapsed)")
    elif range_delta < -0.1:
        concerns.append(f"Probability range narrowed by {abs(range_delta):.4f}")
    
    # High confidence on HGG
    hgg_high_conf_old = old_dist['hgg_high_confidence']
    hgg_high_conf_new = new_dist['hgg_high_confidence']
    if hgg_high_conf_new > hgg_high_conf_old:
        improvements.append(f"More HGG cases with high confidence (≥0.7): {hgg_high_conf_old} → {hgg_high_conf_new}")
    
    # Decision
    if len(concerns) == 0 and len(improvements) >= 2:
        recommendation['decision'] = 'YES'
        recommendation['confidence'] = 'high'
    elif len(concerns) == 0 and len(improvements) == 1:
        recommendation['decision'] = 'YES'
        recommendation['confidence'] = 'medium'
    elif len(concerns) > 0 and 'CRITICAL' in ' '.join(concerns):
        recommendation['decision'] = 'NO'
        recommendation['confidence'] = 'high'
    elif len(concerns) > len(improvements):
        recommendation['decision'] = 'NO'
        recommendation['confidence'] = 'medium'
    else:
        recommendation['decision'] = 'CONDITIONAL'
        recommendation['confidence'] = 'medium'
    
    recommendation['reasoning'] = improvements
    recommendation['concerns'] = concerns
    
    # Suggest improvements if conditional or no
    if recommendation['decision'] in ['CONDITIONAL', 'NO']:
        if new_dist['is_collapsed']:
            recommendation['improvements_needed'].append({
                'area': 'Probability distribution',
                'issue': 'Model predictions are collapsed (narrow range)',
                'suggestion': 'Increase entropy regularization or adjust bag sampling to encourage diversity',
                'expected_effect': 'Wider probability range, better calibration'
            })
        
        if new_metrics['recall'] < 0.85:
            recommendation['improvements_needed'].append({
                'area': 'Recall',
                'issue': f"Recall is {new_metrics['recall']:.3f}, should be >0.85 for medical use",
                'suggestion': 'Adjust loss function to penalize false negatives more, or use class weights',
                'expected_effect': 'Higher recall, fewer missed HGG cases'
            })
        
        if new_dist['hgg_low_confidence'] > len(comparison['y_true'][comparison['y_true']==1]) * 0.1:
            recommendation['improvements_needed'].append({
                'area': 'Confidence on HGG',
                'issue': f"{new_dist['hgg_low_confidence']} HGG cases have low confidence (<0.5)",
                'suggestion': 'Improve instance sampling to focus on discriminative patches, or increase bag size',
                'expected_effect': 'Higher confidence on true positives'
            })
        
        if separation_delta < 0.05:
            recommendation['improvements_needed'].append({
                'area': 'Class separation',
                'issue': 'Insufficient separation between HGG and LGG probabilities',
                'suggestion': 'Tune entropy-based sampling parameters, add confidence regularization',
                'expected_effect': 'Better separation, clearer decision boundary'
            })
    
    return recommendation


def generate_report(comparison: Dict, old_validation: Dict, new_validation: Dict, 
                   recommendation: Dict, output_dir: Path):
    """Generate comprehensive text report."""
    
    report_path = output_dir / 'comparison_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MIL MODEL COMPARISON REPORT: OLD vs NEW (Entropy-Based Sampling)\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Decision: {recommendation['decision']}\n")
        f.write(f"Confidence: {recommendation['confidence']}\n\n")
        
        if recommendation['decision'] == 'YES':
            f.write("The NEW MIL model (entropy-based sampling) should REPLACE the OLD MIL model.\n")
        elif recommendation['decision'] == 'NO':
            f.write("The NEW MIL model should NOT replace the OLD MIL model at this time.\n")
        else:
            f.write("The NEW MIL model shows promise but needs improvements before replacing OLD MIL.\n")
        f.write("\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n")
        for improvement in recommendation['reasoning']:
            f.write(f"✓ {improvement}\n")
        f.write("\n")
        
        if recommendation.get('concerns'):
            f.write("CONCERNS\n")
            f.write("-"*80 + "\n")
            for concern in recommendation['concerns']:
                f.write(f"⚠ {concern}\n")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED METRICS COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("Performance Metrics @ Threshold 0.5\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<20} {'OLD MIL':<15} {'NEW MIL':<15} {'Delta':<15}\n")
        f.write("-"*80 + "\n")
        
        metrics = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'fn_count', 'fp_count']
        for metric in metrics:
            old_val = comparison['old_metrics'][metric]
            new_val = comparison['new_metrics'][metric]
            delta = new_val - old_val
            if isinstance(old_val, float):
                f.write(f"{metric:<20} {old_val:<15.4f} {new_val:<15.4f} {delta:+.4f}\n")
            else:
                f.write(f"{metric:<20} {old_val:<15} {new_val:<15} {delta:+d}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SIGNAL QUALITY ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Probability Distribution Statistics\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Statistic':<25} {'OLD MIL':<20} {'NEW MIL':<20} {'Delta':<15}\n")
        f.write("-"*80 + "\n")
        
        dist_stats = ['mean', 'std', 'min', 'max', 'range_width', 'separation', 'hgg_mean', 'lgg_mean']
        for stat in dist_stats:
            old_val = comparison['old_distribution'][stat]
            new_val = comparison['new_distribution'][stat]
            delta = new_val - old_val
            f.write(f"{stat:<25} {old_val:<20.4f} {new_val:<20.4f} {delta:+.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATION & NEXT STEPS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Decision: {recommendation['decision']}\n")
        f.write(f"Confidence Level: {recommendation['confidence']}\n\n")
        
        if recommendation['improvements_needed']:
            f.write("Suggested Improvements:\n")
            f.write("-"*80 + "\n")
            for i, improvement in enumerate(recommendation['improvements_needed'], 1):
                f.write(f"\n{i}. {improvement['area']}\n")
                f.write(f"   Issue: {improvement['issue']}\n")
                f.write(f"   Suggestion: {improvement['suggestion']}\n")
                f.write(f"   Expected Effect: {improvement['expected_effect']}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("VALIDATION REPORTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("OLD MIL Validation:\n")
        f.write(f"  - Patients: {old_validation['num_patients']}\n")
        f.write(f"  - Folds: {old_validation['num_folds']}\n")
        f.write(f"  - Validation Passed: {old_validation['validation_passed']}\n")
        
        f.write("\nNEW MIL Validation:\n")
        f.write(f"  - Patients: {new_validation['num_patients']}\n")
        f.write(f"  - Folds: {new_validation['num_folds']}\n")
        f.write(f"  - Validation Passed: {new_validation['validation_passed']}\n")
    
    print(f"\n✓ Saved report: {report_path}")


def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("MIL MODEL COMPARISON: OLD vs NEW (Entropy-Based Sampling)")
    print("="*80)
    
    # Load and validate
    old_df, old_validation = load_and_validate_oof(OLD_MIL_PATH, 'hgg_prob', 'OLD MIL (DualStreamMIL-3D)')
    new_df, new_validation = load_and_validate_oof(NEW_MIL_PATH, 'hgg_prob_mil', 'NEW MIL (Entropy Sampling)')
    
    # Compare
    comparison = compare_models(old_df, new_df, 'hgg_prob', 'hgg_prob_mil')
    
    # Generate recommendation
    print("\n" + "="*60)
    print("GENERATING RECOMMENDATION")
    print("="*60)
    recommendation = generate_recommendation(comparison)
    
    print(f"\nDecision: {recommendation['decision']}")
    print(f"Confidence: {recommendation['confidence']}")
    print(f"\nKey Improvements:")
    for improvement in recommendation['reasoning']:
        print(f"  ✓ {improvement}")
    
    if recommendation.get('concerns'):
        print(f"\nConcerns:")
        for concern in recommendation['concerns']:
            print(f"  ⚠ {concern}")
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    create_visualizations(comparison, OUTPUT_DIR)
    
    # Generate report
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    generate_report(comparison, old_validation, new_validation, recommendation, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - comparison_plots.png")
    print(f"  - comparison_report.txt")


if __name__ == '__main__':
    main()

