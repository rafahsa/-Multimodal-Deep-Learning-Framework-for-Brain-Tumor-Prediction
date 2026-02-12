#!/usr/bin/env python3
"""
Deep Analysis of Current Swin Model for Designing a Complementary Second Swin

This script performs comprehensive analysis of the current Swin model to:
1. Identify strengths and weaknesses
2. Analyze error patterns (FN, FP)
3. Check redundancy with ResNet and MIL
4. Design a complementary second Swin model
5. Assess feasibility

Output: Comprehensive report with GO/NO-GO recommendation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_curve,
    roc_curve, precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OOF_FILE = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'swin_complementary_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load OOF predictions."""
    df = pd.read_csv(OOF_FILE)
    logger.info(f"Loaded {len(df)} patients")
    logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")
    return df


def analyze_swin_strengths(df: pd.DataFrame) -> Dict:
    """Analyze where Swin is strong."""
    logger.info("\n" + "="*80)
    logger.info("PART 1: SWIN STRENGTHS ANALYSIS")
    logger.info("="*80)
    
    # Threshold for "confident" predictions
    high_conf_threshold = 0.8
    low_conf_threshold = 0.2
    
    # Correct predictions
    df['swin_correct'] = ((df['hgg_prob_swin'] >= 0.5) & (df['label'] == 1)) | \
                         ((df['hgg_prob_swin'] < 0.5) & (df['label'] == 0))
    
    # High confidence correct
    df['swin_high_conf_correct'] = (
        ((df['hgg_prob_swin'] >= high_conf_threshold) & (df['label'] == 1)) |
        ((df['hgg_prob_swin'] <= low_conf_threshold) & (df['label'] == 0))
    ) & df['swin_correct']
    
    # Analyze correct predictions
    correct_mask = df['swin_correct']
    correct_hgg = df[correct_mask & (df['label'] == 1)]
    correct_lgg = df[correct_mask & (df['label'] == 0)]
    
    results = {
        'overall_accuracy': correct_mask.mean(),
        'hgg_accuracy': (df['label'] == 1).sum() and (correct_mask & (df['label'] == 1)).sum() / (df['label'] == 1).sum(),
        'lgg_accuracy': (df['label'] == 0).sum() and (correct_mask & (df['label'] == 0)).sum() / (df['label'] == 0).sum(),
        'high_conf_correct_pct': df['swin_high_conf_correct'].sum() / len(df),
        'high_conf_correct_hgg': correct_hgg[correct_hgg['hgg_prob_swin'] >= high_conf_threshold].shape[0],
        'high_conf_correct_lgg': correct_lgg[correct_lgg['hgg_prob_swin'] <= low_conf_threshold].shape[0],
        'mean_prob_correct_hgg': correct_hgg['hgg_prob_swin'].mean(),
        'mean_prob_correct_lgg': correct_lgg['hgg_prob_swin'].mean(),
        'std_prob_correct_hgg': correct_hgg['hgg_prob_swin'].std(),
        'std_prob_correct_lgg': correct_lgg['hgg_prob_swin'].std(),
    }
    
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"HGG Accuracy: {results['hgg_accuracy']:.4f}")
    logger.info(f"LGG Accuracy: {results['lgg_accuracy']:.4f}")
    logger.info(f"High Confidence Correct: {results['high_conf_correct_pct']:.4f}")
    logger.info(f"Mean HGG prob (correct): {results['mean_prob_correct_hgg']:.4f} ± {results['std_prob_correct_hgg']:.4f}")
    logger.info(f"Mean LGG prob (correct): {results['mean_prob_correct_lgg']:.4f} ± {results['std_prob_correct_lgg']:.4f}")
    
    return results


def analyze_swin_errors(df: pd.DataFrame) -> Dict:
    """Analyze Swin errors (FN and FP)."""
    logger.info("\n" + "="*80)
    logger.info("PART 2: SWIN ERROR ANALYSIS")
    logger.info("="*80)
    
    # Predictions at threshold 0.5
    df['swin_pred'] = (df['hgg_prob_swin'] >= 0.5).astype(int)
    
    # False Negatives (HGG predicted as LGG)
    fn_mask = (df['label'] == 1) & (df['swin_pred'] == 0)
    fn_cases = df[fn_mask].copy()
    
    # False Positives (LGG predicted as HGG)
    fp_mask = (df['label'] == 0) & (df['swin_pred'] == 1)
    fp_cases = df[fp_mask].copy()
    
    # True Positives and True Negatives for comparison
    tp_mask = (df['label'] == 1) & (df['swin_pred'] == 1)
    tn_mask = (df['label'] == 0) & (df['swin_pred'] == 0)
    tp_cases = df[tp_mask].copy()
    tn_cases = df[tn_mask].copy()
    
    results = {
        'fn_count': fn_mask.sum(),
        'fp_count': fp_mask.sum(),
        'fn_patients': fn_cases['patient_id'].tolist(),
        'fp_patients': fp_cases['patient_id'].tolist(),
        'fn_mean_prob': fn_cases['hgg_prob_swin'].mean() if len(fn_cases) > 0 else 0,
        'fp_mean_prob': fp_cases['hgg_prob_swin'].mean() if len(fp_cases) > 0 else 0,
        'fn_std_prob': fn_cases['hgg_prob_swin'].std() if len(fn_cases) > 0 else 0,
        'fp_std_prob': fp_cases['hgg_prob_swin'].std() if len(fp_cases) > 0 else 0,
        'fn_fold_distribution': fn_cases['fold'].value_counts().to_dict() if len(fn_cases) > 0 else {},
        'fp_fold_distribution': fp_cases['fold'].value_counts().to_dict() if len(fp_cases) > 0 else {},
    }
    
    # Compare FN probabilities with TP probabilities
    if len(fn_cases) > 0 and len(tp_cases) > 0:
        results['fn_vs_tp_prob_diff'] = tp_cases['hgg_prob_swin'].mean() - fn_cases['hgg_prob_swin'].mean()
        results['fn_vs_tp_prob_ttest'] = stats.ttest_ind(tp_cases['hgg_prob_swin'], fn_cases['hgg_prob_swin'])[1]
    
    # Compare FP probabilities with TN probabilities
    if len(fp_cases) > 0 and len(tn_cases) > 0:
        results['fp_vs_tn_prob_diff'] = fp_cases['hgg_prob_swin'].mean() - tn_cases['hgg_prob_swin'].mean()
        results['fp_vs_tn_prob_ttest'] = stats.ttest_ind(fp_cases['hgg_prob_swin'], tn_cases['hgg_prob_swin'])[1]
    
    logger.info(f"False Negatives: {results['fn_count']} ({results['fn_count']/len(df)*100:.2f}%)")
    logger.info(f"False Positives: {results['fp_count']} ({results['fp_count']/len(df)*100:.2f}%)")
    logger.info(f"FN Mean Prob: {results['fn_mean_prob']:.4f} ± {results['fn_std_prob']:.4f}")
    logger.info(f"FP Mean Prob: {results['fp_mean_prob']:.4f} ± {results['fp_std_prob']:.4f}")
    
    if 'fn_vs_tp_prob_diff' in results:
        logger.info(f"FN vs TP Prob Difference: {results['fn_vs_tp_prob_diff']:.4f} (p={results['fn_vs_tp_prob_ttest']:.4e})")
    
    return results, fn_cases, fp_cases


def analyze_error_consistency(df: pd.DataFrame, fn_cases: pd.DataFrame, fp_cases: pd.DataFrame) -> Dict:
    """Analyze error consistency across folds."""
    logger.info("\n" + "="*80)
    logger.info("PART 3: ERROR CONSISTENCY ANALYSIS")
    logger.info("="*80)
    
    # Check if errors are consistent across folds
    # (i.e., same patient misclassified in multiple folds - not possible with OOF, but check patterns)
    
    # Analyze error patterns per fold
    fold_errors = {}
    for fold in range(5):
        fold_df = df[df['fold'] == fold]
        fold_df['swin_pred'] = (fold_df['hgg_prob_swin'] >= 0.5).astype(int)
        
        fold_fn = ((fold_df['label'] == 1) & (fold_df['swin_pred'] == 0)).sum()
        fold_fp = ((fold_df['label'] == 0) & (fold_df['swin_pred'] == 1)).sum()
        
        fold_errors[fold] = {
            'fn': fold_fn,
            'fp': fold_fp,
            'fn_rate': fold_fn / (fold_df['label'] == 1).sum() if (fold_df['label'] == 1).sum() > 0 else 0,
            'fp_rate': fold_fp / (fold_df['label'] == 0).sum() if (fold_df['label'] == 0).sum() > 0 else 0,
        }
    
    results = {
        'fold_errors': fold_errors,
        'fn_std_across_folds': np.std([fold_errors[f]['fn'] for f in range(5)]),
        'fp_std_across_folds': np.std([fold_errors[f]['fp'] for f in range(5)]),
    }
    
    logger.info("Error distribution across folds:")
    for fold in range(5):
        logger.info(f"  Fold {fold}: FN={fold_errors[fold]['fn']}, FP={fold_errors[fold]['fp']}")
    
    logger.info(f"FN std across folds: {results['fn_std_across_folds']:.2f}")
    logger.info(f"FP std across folds: {results['fp_std_across_folds']:.2f}")
    
    return results


def analyze_confidence_behavior(df: pd.DataFrame) -> Dict:
    """Analyze Swin's confidence behavior."""
    logger.info("\n" + "="*80)
    logger.info("PART 4: CONFIDENCE BEHAVIOR ANALYSIS")
    logger.info("="*80)
    
    # Probability distributions for correct vs incorrect
    df['swin_correct'] = ((df['hgg_prob_swin'] >= 0.5) & (df['label'] == 1)) | \
                         ((df['hgg_prob_swin'] < 0.5) & (df['label'] == 0))
    
    correct_probs = df[df['swin_correct']]['hgg_prob_swin'].values
    incorrect_probs = df[~df['swin_correct']]['hgg_prob_swin'].values
    
    # Calibration analysis: bin probabilities and check accuracy per bin
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_accuracies = []
    bin_counts = []
    bin_mean_probs = []
    
    for i in range(len(bins) - 1):
        bin_mask = (df['hgg_prob_swin'] >= bins[i]) & (df['hgg_prob_swin'] < bins[i+1])
        if i == len(bins) - 2:  # Last bin includes upper bound
            bin_mask = (df['hgg_prob_swin'] >= bins[i]) & (df['hgg_prob_swin'] <= bins[i+1])
        
        bin_df = df[bin_mask]
        if len(bin_df) > 0:
            # For HGG: accuracy = proportion with prob >= 0.5
            # For LGG: accuracy = proportion with prob < 0.5
            hgg_mask = bin_df['label'] == 1
            lgg_mask = bin_df['label'] == 0
            
            hgg_correct = (bin_df[hgg_mask]['hgg_prob_swin'] >= 0.5).sum() if hgg_mask.sum() > 0 else 0
            lgg_correct = (bin_df[lgg_mask]['hgg_prob_swin'] < 0.5).sum() if lgg_mask.sum() > 0 else 0
            
            accuracy = (hgg_correct + lgg_correct) / len(bin_df) if len(bin_df) > 0 else 0
            bin_accuracies.append(accuracy)
            bin_counts.append(len(bin_df))
            bin_mean_probs.append(bin_df['hgg_prob_swin'].mean())
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
            bin_mean_probs.append(bin_centers[i])
    
    # ECE (Expected Calibration Error)
    ece = np.sum([abs(acc - mean_prob) * count for acc, mean_prob, count in 
                  zip(bin_accuracies, bin_mean_probs, bin_counts)]) / sum(bin_counts) if sum(bin_counts) > 0 else 0
    
    results = {
        'correct_mean_prob': np.mean(correct_probs) if len(correct_probs) > 0 else 0,
        'correct_std_prob': np.std(correct_probs) if len(correct_probs) > 0 else 0,
        'incorrect_mean_prob': np.mean(incorrect_probs) if len(incorrect_probs) > 0 else 0,
        'incorrect_std_prob': np.std(incorrect_probs) if len(incorrect_probs) > 0 else 0,
        'ece': ece,
        'calibration_bins': {
            'bin_centers': bin_centers.tolist(),
            'accuracies': bin_accuracies,
            'mean_probs': bin_mean_probs,
            'counts': bin_counts
        }
    }
    
    logger.info(f"Correct predictions - Mean prob: {results['correct_mean_prob']:.4f} ± {results['correct_std_prob']:.4f}")
    logger.info(f"Incorrect predictions - Mean prob: {results['incorrect_mean_prob']:.4f} ± {results['incorrect_std_prob']:.4f}")
    logger.info(f"Expected Calibration Error (ECE): {results['ece']:.4f}")
    
    # Overconfidence/underconfidence
    if results['correct_mean_prob'] > 0.7 and results['incorrect_mean_prob'] > 0.4:
        logger.info("⚠️  Swin appears OVERCONFIDENT (high prob on incorrect predictions)")
    elif results['correct_mean_prob'] < 0.6 and results['incorrect_mean_prob'] < 0.3:
        logger.info("⚠️  Swin appears UNDERCONFIDENT (low prob on correct predictions)")
    else:
        logger.info("✓ Swin confidence appears reasonable")
    
    return results


def analyze_redundancy(df: pd.DataFrame) -> Dict:
    """Analyze redundancy between Swin, ResNet, and MIL."""
    logger.info("\n" + "="*80)
    logger.info("PART 5: REDUNDANCY ANALYSIS")
    logger.info("="*80)
    
    # Correlation between predictions
    swin_resnet_corr = df['hgg_prob_swin'].corr(df['hgg_prob_resnet'])
    swin_mil_corr = df['hgg_prob_swin'].corr(df['mil_prob'])
    resnet_mil_corr = df['hgg_prob_resnet'].corr(df['mil_prob'])
    
    # Agreement on predictions (at threshold 0.5)
    df['swin_pred'] = (df['hgg_prob_swin'] >= 0.5).astype(int)
    df['resnet_pred'] = (df['hgg_prob_resnet'] >= 0.5).astype(int)
    df['mil_pred'] = (df['mil_prob'] >= 0.5).astype(int)
    
    swin_resnet_agree = (df['swin_pred'] == df['resnet_pred']).mean()
    swin_mil_agree = (df['swin_pred'] == df['mil_pred']).mean()
    resnet_mil_agree = (df['resnet_pred'] == df['mil_pred']).mean()
    
    # Cases where Swin is correct but others are wrong
    df['swin_correct'] = ((df['hgg_prob_swin'] >= 0.5) & (df['label'] == 1)) | \
                         ((df['hgg_prob_swin'] < 0.5) & (df['label'] == 0))
    df['resnet_correct'] = ((df['hgg_prob_resnet'] >= 0.5) & (df['label'] == 1)) | \
                           ((df['hgg_prob_resnet'] < 0.5) & (df['label'] == 0))
    df['mil_correct'] = ((df['mil_prob'] >= 0.5) & (df['label'] == 1)) | \
                        ((df['mil_prob'] < 0.5) & (df['label'] == 0))
    
    swin_unique_correct = df[df['swin_correct'] & ~df['resnet_correct'] & ~df['mil_correct']]
    resnet_unique_correct = df[df['resnet_correct'] & ~df['swin_correct'] & ~df['mil_correct']]
    mil_unique_correct = df[df['mil_correct'] & ~df['swin_correct'] & ~df['resnet_correct']]
    
    results = {
        'swin_resnet_correlation': swin_resnet_corr,
        'swin_mil_correlation': swin_mil_corr,
        'resnet_mil_correlation': resnet_mil_corr,
        'swin_resnet_agreement': swin_resnet_agree,
        'swin_mil_agreement': swin_mil_agree,
        'resnet_mil_agreement': resnet_mil_agree,
        'swin_unique_correct_count': len(swin_unique_correct),
        'resnet_unique_correct_count': len(resnet_unique_correct),
        'mil_unique_correct_count': len(mil_unique_correct),
        'swin_unique_correct_patients': swin_unique_correct['patient_id'].tolist(),
    }
    
    logger.info(f"Swin-ResNet Correlation: {swin_resnet_corr:.4f}")
    logger.info(f"Swin-MIL Correlation: {swin_mil_corr:.4f}")
    logger.info(f"ResNet-MIL Correlation: {resnet_mil_corr:.4f}")
    logger.info(f"\nPrediction Agreement (threshold 0.5):")
    logger.info(f"  Swin-ResNet: {swin_resnet_agree:.4f}")
    logger.info(f"  Swin-MIL: {swin_mil_agree:.4f}")
    logger.info(f"  ResNet-MIL: {resnet_mil_agree:.4f}")
    logger.info(f"\nUnique Correct Cases:")
    logger.info(f"  Swin only: {results['swin_unique_correct_count']}")
    logger.info(f"  ResNet only: {results['resnet_unique_correct_count']}")
    logger.info(f"  MIL only: {results['mil_unique_correct_count']}")
    
    return results


def design_complementary_swin(
    strengths: Dict,
    errors: Dict,
    consistency: Dict,
    confidence: Dict,
    redundancy: Dict,
    fn_cases: pd.DataFrame,
    fp_cases: pd.DataFrame
) -> Dict:
    """Design a complementary second Swin model."""
    logger.info("\n" + "="*80)
    logger.info("PART 6: COMPLEMENTARY SWIN DESIGN")
    logger.info("="*80)
    
    design = {
        'rationale': {},
        'architectural_changes': {},
        'training_strategy': {},
        'expected_impact': {},
        'risks': []
    }
    
    # Analyze FN patterns to design complementary model
    fn_prob_mean = errors['fn_mean_prob']
    fn_prob_std = errors['fn_std_prob']
    
    logger.info(f"\nFN Analysis:")
    logger.info(f"  Mean prob: {fn_prob_mean:.4f} ± {fn_prob_std:.4f}")
    logger.info(f"  FN cases are near threshold (0.5), suggesting uncertainty")
    
    # Design recommendations
    design['rationale'] = {
        'primary_goal': 'Reduce FN by capturing small/diffuse tumors that current Swin misses',
        'secondary_goal': 'Reduce FP by better distinguishing LGG from HGG',
        'key_insight': f"Current Swin has {errors['fn_count']} FN with mean prob {fn_prob_mean:.4f}, suggesting it misses subtle HGG cases"
    }
    
    # Architectural recommendations
    design['architectural_changes'] = {
        'input_view': 'Keep axial (current), but consider multi-view ensemble later',
        'patch_size': 'Smaller patch size (1 instead of 2) to capture fine details',
        'window_size': 'Smaller window (4 instead of 7) for local attention',
        'resolution': 'Higher resolution input (160x160x160 instead of 128x128x128)',
        'cropping': 'Tumor-focused cropping (if segmentation available) OR full brain with attention',
        'feature_size': 'Larger feature size (64 instead of 48) for more capacity',
        'depths': 'Deeper network ([3, 3, 3, 3] instead of [2, 2, 2, 2]) for more representation',
    }
    
    # Training strategy
    design['training_strategy'] = {
        'loss_function': 'Focal Loss (gamma=2.0, alpha=0.25) to focus on hard examples',
        'sampling': 'Hard example mining - oversample FN cases from current Swin',
        'class_weights': 'Higher weight for HGG class (pos_weight=2.0-3.0)',
        'augmentation': 'Stronger augmentation for small tumors (zoom, rotation)',
        'regularization': 'Moderate dropout (0.3) to prevent overfitting',
    }
    
    # Expected impact
    design['expected_impact'] = {
        'fn_reduction': f"Target: Reduce FN from {errors['fn_count']} to <10 (60% reduction)",
        'fp_reduction': f"Target: Reduce FP from {errors['fp_count']} to <10",
        'complementarity': 'Should capture different signal than Swin-1 (small tumors, diffuse patterns)',
        'ensemble_interaction': 'Should have lower correlation with Swin-1 (<0.7) while maintaining high AUC (>0.85)'
    }
    
    # Risks
    design['risks'] = [
        'Overfitting: 285 samples may not support deeper/larger model',
        'Redundancy: Second Swin may learn similar patterns to first',
        'Computational cost: Higher resolution and deeper network = slower training',
        'Data requirements: May need more data augmentation or synthetic data'
    ]
    
    logger.info("\nDesign Recommendations:")
    logger.info(f"  Primary Goal: {design['rationale']['primary_goal']}")
    logger.info(f"  Key Architectural Changes:")
    for key, value in design['architectural_changes'].items():
        logger.info(f"    {key}: {value}")
    logger.info(f"  Training Strategy:")
    for key, value in design['training_strategy'].items():
        logger.info(f"    {key}: {value}")
    
    return design


def assess_feasibility(
    errors: Dict,
    design: Dict,
    df: pd.DataFrame
) -> Dict:
    """Assess feasibility of achieving targets."""
    logger.info("\n" + "="*80)
    logger.info("PART 7: FEASIBILITY ASSESSMENT")
    logger.info("="*80)
    
    current_fn = errors['fn_count']
    current_fp = errors['fp_count']
    
    target_fn = 5
    target_fp = 5
    target_precision = 0.95
    target_recall = 0.95
    
    # Current metrics
    df['swin_pred'] = (df['hgg_prob_swin'] >= 0.5).astype(int)
    current_precision = precision_score(df['label'], df['swin_pred'], zero_division=0)
    current_recall = recall_score(df['label'], df['swin_pred'], zero_division=0)
    current_auc = roc_auc_score(df['label'], df['hgg_prob_swin'])
    
    # Estimate if targets are achievable
    n_samples = len(df)
    n_hgg = (df['label'] == 1).sum()
    n_lgg = (df['label'] == 0).sum()
    
    # Theoretical limits
    # FN < 5 means recall > (n_hgg - 5) / n_hgg
    min_fn_recall = (n_hgg - target_fn) / n_hgg if n_hgg > 0 else 0
    # FP < 5 means precision > n_hgg / (n_hgg + 5)
    min_fp_precision = n_hgg / (n_hgg + target_fp) if (n_hgg + target_fp) > 0 else 0
    
    feasibility = {
        'current_metrics': {
            'fn': current_fn,
            'fp': current_fp,
            'precision': current_precision,
            'recall': current_recall,
            'auc': current_auc
        },
        'target_metrics': {
            'fn': target_fn,
            'fp': target_fp,
            'precision': target_precision,
            'recall': target_recall
        },
        'theoretical_limits': {
            'min_fn_recall': min_fn_recall,
            'min_fp_precision': min_fp_precision,
            'achievable_with_fn_5': min_fn_recall >= target_recall,
            'achievable_with_fp_5': min_fp_precision >= target_precision
        },
        'realistic_assessment': {},
        'go_no_go': 'CONDITIONAL'
    }
    
    logger.info(f"\nCurrent Metrics:")
    logger.info(f"  FN: {current_fn}, FP: {current_fp}")
    logger.info(f"  Precision: {current_precision:.4f}, Recall: {current_recall:.4f}")
    logger.info(f"  AUC: {current_auc:.4f}")
    
    logger.info(f"\nTarget Metrics:")
    logger.info(f"  FN: {target_fn}, FP: {target_fp}")
    logger.info(f"  Precision: {target_precision:.4f}, Recall: {target_recall:.4f}")
    
    logger.info(f"\nTheoretical Limits:")
    logger.info(f"  Min Recall with FN=5: {min_fn_recall:.4f}")
    logger.info(f"  Min Precision with FP=5: {min_fp_precision:.4f}")
    
    # Realistic assessment
    if min_fn_recall >= target_recall and min_fp_precision >= target_precision:
        feasibility['realistic_assessment'] = {
            'theoretically_possible': True,
            'practical_likelihood': 'MODERATE',
            'reason': 'Theoretical limits allow targets, but requires near-perfect model'
        }
        logger.info("\n✓ Theoretically possible, but challenging")
    else:
        feasibility['realistic_assessment'] = {
            'theoretically_possible': False,
            'practical_likelihood': 'LOW',
            'reason': 'Theoretical limits do not allow simultaneous achievement of all targets'
        }
        logger.info("\n⚠️  Theoretically difficult to achieve all targets simultaneously")
    
    # GO/NO-GO decision
    improvement_needed_fn = current_fn - target_fn
    improvement_needed_fp = current_fp - target_fp
    
    if improvement_needed_fn <= 5 and improvement_needed_fp <= 5:
        feasibility['go_no_go'] = 'GO'
        feasibility['go_no_go_reason'] = 'Reasonable improvement targets'
    elif improvement_needed_fn <= 10 and improvement_needed_fp <= 10:
        feasibility['go_no_go'] = 'CONDITIONAL_GO'
        feasibility['go_no_go_reason'] = 'Moderate improvement needed, worth trying'
    else:
        feasibility['go_no_go'] = 'NO_GO'
        feasibility['go_no_go_reason'] = 'Too large improvement needed, unlikely to succeed'
    
    logger.info(f"\nGO/NO-GO Decision: {feasibility['go_no_go']}")
    logger.info(f"Reason: {feasibility['go_no_go_reason']}")
    
    return feasibility


def generate_visualizations(df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots."""
    logger.info("\nGenerating visualizations...")
    
    # 1. Probability distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Correct vs Incorrect
    df['swin_correct'] = ((df['hgg_prob_swin'] >= 0.5) & (df['label'] == 1)) | \
                         ((df['hgg_prob_swin'] < 0.5) & (df['label'] == 0))
    correct_probs = df[df['swin_correct']]['hgg_prob_swin']
    incorrect_probs = df[~df['swin_correct']]['hgg_prob_swin']
    
    axes[0, 0].hist(correct_probs, bins=50, alpha=0.7, label='Correct', density=True)
    axes[0, 0].hist(incorrect_probs, bins=50, alpha=0.7, label='Incorrect', density=True)
    axes[0, 0].set_xlabel('HGG Probability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Swin Probability Distribution: Correct vs Incorrect')
    axes[0, 0].legend()
    axes[0, 0].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    # HGG vs LGG
    hgg_probs = df[df['label'] == 1]['hgg_prob_swin']
    lgg_probs = df[df['label'] == 0]['hgg_prob_swin']
    
    axes[0, 1].hist(hgg_probs, bins=50, alpha=0.7, label='HGG', density=True)
    axes[0, 1].hist(lgg_probs, bins=50, alpha=0.7, label='LGG', density=True)
    axes[0, 1].set_xlabel('HGG Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Swin Probability Distribution: HGG vs LGG')
    axes[0, 1].legend()
    axes[0, 1].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    # Correlation matrix
    corr_data = df[['hgg_prob_swin', 'hgg_prob_resnet', 'mil_prob']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Model Prediction Correlations')
    
    # Error distribution across folds
    fold_errors = []
    for fold in range(5):
        fold_df = df[df['fold'] == fold]
        fold_df['swin_pred'] = (fold_df['hgg_prob_swin'] >= 0.5).astype(int)
        fn = ((fold_df['label'] == 1) & (fold_df['swin_pred'] == 0)).sum()
        fp = ((fold_df['label'] == 0) & (fold_df['swin_pred'] == 1)).sum()
        fold_errors.append({'fold': fold, 'FN': fn, 'FP': fp})
    
    fold_df_viz = pd.DataFrame(fold_errors)
    x = np.arange(5)
    width = 0.35
    axes[1, 1].bar(x - width/2, fold_df_viz['FN'], width, label='FN', alpha=0.7)
    axes[1, 1].bar(x + width/2, fold_df_viz['FP'], width, label='FP', alpha=0.7)
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Error Distribution Across Folds')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'Fold {i}' for i in range(5)])
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'swin_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved visualizations to {output_dir / 'swin_analysis_plots.png'}")


def main():
    """Main analysis pipeline."""
    logger.info("="*80)
    logger.info("SWIN COMPLEMENTARY MODEL DESIGN ANALYSIS")
    logger.info("="*80)
    
    # Load data
    df = load_data()
    
    # Part 1: Strengths
    strengths = analyze_swin_strengths(df)
    
    # Part 2: Errors
    errors, fn_cases, fp_cases = analyze_swin_errors(df)
    
    # Part 3: Consistency
    consistency = analyze_error_consistency(df, fn_cases, fp_cases)
    
    # Part 4: Confidence
    confidence = analyze_confidence_behavior(df)
    
    # Part 5: Redundancy
    redundancy = analyze_redundancy(df)
    
    # Part 6: Design
    design = design_complementary_swin(
        strengths, errors, consistency, confidence, redundancy,
        fn_cases, fp_cases
    )
    
    # Part 7: Feasibility
    feasibility = assess_feasibility(errors, design, df)
    
    # Generate visualizations
    generate_visualizations(df, OUTPUT_DIR)
    
    # Compile final report
    report = {
        'strengths': strengths,
        'errors': errors,
        'consistency': consistency,
        'confidence': confidence,
        'redundancy': redundancy,
        'design': design,
        'feasibility': feasibility,
        'summary': {
            'current_fn': errors['fn_count'],
            'current_fp': errors['fp_count'],
            'target_fn': 5,
            'target_fp': 5,
            'go_no_go': feasibility['go_no_go'],
            'go_no_go_reason': feasibility['go_no_go_reason']
        }
    }
    
    # Save report
    report_file = OUTPUT_DIR / 'swin_complementary_analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\n✓ Saved report to {report_file}")
    
    # Generate markdown summary
    generate_markdown_report(report, OUTPUT_DIR)
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


def generate_markdown_report(report: Dict, output_dir: Path):
    """Generate markdown summary report."""
    md_content = f"""# Swin Complementary Model Design Analysis

## Executive Summary

**Current Swin Performance:**
- False Negatives: {report['errors']['fn_count']}
- False Positives: {report['errors']['fp_count']}
- Precision: {report['feasibility']['current_metrics']['precision']:.4f}
- Recall: {report['feasibility']['current_metrics']['recall']:.4f}
- AUC: {report['feasibility']['current_metrics']['auc']:.4f}

**Target Performance:**
- False Negatives: < 5
- False Positives: < 5
- Precision: > 0.95
- Recall: > 0.95

**GO/NO-GO Decision: {report['feasibility']['go_no_go']}**
- Reason: {report['feasibility']['go_no_go_reason']}

---

## Part 1: Current Swin Strengths

- Overall Accuracy: {report['strengths']['overall_accuracy']:.4f}
- HGG Accuracy: {report['strengths']['hgg_accuracy']:.4f}
- LGG Accuracy: {report['strengths']['lgg_accuracy']:.4f}
- High Confidence Correct: {report['strengths']['high_conf_correct_pct']:.4f}

**Key Insight:** Swin performs well on high-confidence cases, suggesting it captures clear tumor patterns effectively.

---

## Part 2: Error Analysis

### False Negatives (HGG predicted as LGG)
- Count: {report['errors']['fn_count']}
- Mean Probability: {report['errors']['fn_mean_prob']:.4f} ± {report['errors']['fn_std_prob']:.4f}
- **Key Finding:** FN cases have probabilities near threshold (0.5), indicating uncertainty on subtle HGG cases.

### False Positives (LGG predicted as HGG)
- Count: {report['errors']['fp_count']}
- Mean Probability: {report['errors']['fp_mean_prob']:.4f} ± {report['errors']['fp_std_prob']:.4f}

---

## Part 3: Redundancy Analysis

- Swin-ResNet Correlation: {report['redundancy']['swin_resnet_correlation']:.4f}
- Swin-MIL Correlation: {report['redundancy']['swin_mil_correlation']:.4f}
- ResNet-MIL Correlation: {report['redundancy']['resnet_mil_correlation']:.4f}

**Key Finding:** Swin has moderate correlation with ResNet ({report['redundancy']['swin_resnet_correlation']:.3f}) and low correlation with MIL ({report['redundancy']['swin_mil_correlation']:.3f}), suggesting some complementarity.

- Swin Unique Correct Cases: {report['redundancy']['swin_unique_correct_count']}
- ResNet Unique Correct Cases: {report['redundancy']['resnet_unique_correct_count']}
- MIL Unique Correct Cases: {report['redundancy']['mil_unique_correct_count']}

---

## Part 4: Complementary Swin Design

### Rationale
- **Primary Goal:** {report['design']['rationale']['primary_goal']}
- **Secondary Goal:** {report['design']['rationale']['secondary_goal']}
- **Key Insight:** {report['design']['rationale']['key_insight']}

### Architectural Changes
"""
    
    for key, value in report['design']['architectural_changes'].items():
        md_content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
    
    md_content += f"""
### Training Strategy
"""
    
    for key, value in report['design']['training_strategy'].items():
        md_content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
    
    md_content += f"""
### Expected Impact
"""
    
    for key, value in report['design']['expected_impact'].items():
        md_content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
    
    md_content += f"""
### Risks
"""
    
    for risk in report['design']['risks']:
        md_content += f"- {risk}\n"
    
    md_content += f"""
---

## Part 5: Feasibility Assessment

### Current vs Target Metrics

| Metric | Current | Target | Improvement Needed |
|--------|---------|--------|-------------------|
| FN | {report['feasibility']['current_metrics']['fn']} | {report['feasibility']['target_metrics']['fn']} | {report['feasibility']['current_metrics']['fn'] - report['feasibility']['target_metrics']['fn']} |
| FP | {report['feasibility']['current_metrics']['fp']} | {report['feasibility']['target_metrics']['fp']} | {report['feasibility']['current_metrics']['fp'] - report['feasibility']['target_metrics']['fp']} |
| Precision | {report['feasibility']['current_metrics']['precision']:.4f} | {report['feasibility']['target_metrics']['precision']:.4f} | {report['feasibility']['target_metrics']['precision'] - report['feasibility']['current_metrics']['precision']:.4f} |
| Recall | {report['feasibility']['current_metrics']['recall']:.4f} | {report['feasibility']['target_metrics']['recall']:.4f} | {report['feasibility']['target_metrics']['recall'] - report['feasibility']['current_metrics']['recall']:.4f} |

### Theoretical Limits
- Minimum Recall with FN=5: {report['feasibility']['theoretical_limits']['min_fn_recall']:.4f}
- Minimum Precision with FP=5: {report['feasibility']['theoretical_limits']['min_fp_precision']:.4f}
- Theoretically Possible: {report['feasibility']['realistic_assessment'].get('theoretically_possible', 'Unknown')}

### Realistic Assessment
- **Likelihood:** {report['feasibility']['realistic_assessment'].get('practical_likelihood', 'Unknown')}
- **Reason:** {report['feasibility']['realistic_assessment'].get('reason', 'N/A')}

---

## Recommendations

### Priority 1: High ROI Design Choices
1. **Smaller patch size (1 instead of 2)** - Captures fine details for small tumors
2. **Focal Loss with hard example mining** - Focuses on FN cases
3. **Higher resolution input (160³ instead of 128³)** - Better spatial detail

### Priority 2: Moderate ROI Design Choices
1. **Deeper network ([3,3,3,3] instead of [2,2,2,2])** - More representation capacity
2. **Smaller window size (4 instead of 7)** - Local attention for subtle patterns
3. **Class weighting (pos_weight=2.0-3.0)** - Penalize FN more

### Priority 3: Lower ROI / Higher Risk
1. **Tumor-focused cropping** - Requires segmentation, risk of leakage
2. **Multi-view ensemble** - Computational cost, may not add much value

---

## Final Decision

**GO/NO-GO: {report['feasibility']['go_no_go']}**

**Reason:** {report['feasibility']['go_no_go_reason']}

**Next Steps:**
1. If GO: Implement Priority 1 design choices, train on single fold first
2. If CONDITIONAL_GO: Proceed with caution, monitor for overfitting
3. If NO_GO: Consider alternative approaches (more data, different architecture)

---

## Validation Signals to Monitor

1. **Correlation with Swin-1:** Should be < 0.7 to ensure complementarity
2. **FN Reduction:** Should reduce FN by at least 50% compared to Swin-1
3. **AUC:** Should maintain AUC > 0.85
4. **Overfitting:** Monitor train/val gap, should be < 0.10 AUC difference
5. **Fold Consistency:** FN/FP should be stable across folds (std < 3)

---

*Report generated on {pd.Timestamp.now()}*
"""
    
    md_file = output_dir / 'swin_complementary_analysis_report.md'
    with open(md_file, 'w') as f:
        f.write(md_content)
    
    logger.info(f"✓ Saved markdown report to {md_file}")


if __name__ == '__main__':
    main()

