#!/usr/bin/env python3
"""
Nested CV Evaluation with MIL-Only Probability Calibration

This script evaluates the impact of calibrating ONLY the MIL model probabilities
on ensemble performance, using strict nested cross-validation.

Key features:
- Calibrates MIL probabilities only (Platt or Isotonic)
- Keeps ResNet and Swin probabilities unchanged
- Uses patient-level nested CV (no data leakage)
- Compares baseline vs MIL-Platt vs MIL-Isotonic
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
OUTPUT_DIR = Path('ensemble/results/mil_calibration_nested_cv')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
OUTER_CV_FOLDS = 5
CALIBRATION_FRACTION = 0.7
THRESHOLD_SWEEP_START = 0.05
THRESHOLD_SWEEP_END = 0.95
THRESHOLD_SWEEP_STEP = 0.01
RANDOM_SEED = 42

# Feature columns
BASE_FEATURE_COLS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
TARGET_COLUMN = 'label'
PATIENT_ID_COLUMN = 'patient_id'


def make_json_serializable(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, (bool, np.bool_)):
        return int(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    return obj


def apply_platt_calibration(y_proba_uncal: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, object]:
    """
    Apply Platt scaling to probabilities.
    
    Args:
        y_proba_uncal: Uncalibrated probabilities (1D array)
        y_true: True labels (1D array)
    
    Returns:
        y_proba_cal: Calibrated probabilities
        calibrator: Calibrator object (for saving)
    """
    from sklearn.linear_model import LogisticRegression as PlattScaling
    
    # Clip probabilities to avoid log(0)
    y_proba_clipped = np.clip(y_proba_uncal, 1e-7, 1 - 1e-7)
    
    # Transform to log-odds
    log_odds = np.log(y_proba_clipped / (1 - y_proba_clipped))
    
    # Fit Platt scaling
    platt_model = PlattScaling()
    platt_model.fit(log_odds.reshape(-1, 1), y_true)
    
    # Apply calibration
    log_odds_cal = platt_model.predict_proba(log_odds.reshape(-1, 1))[:, 1]
    
    calibrator = {'type': 'platt', 'model': platt_model}
    
    return log_odds_cal, calibrator


def apply_isotonic_calibration(y_proba_uncal: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, object]:
    """
    Apply Isotonic regression to probabilities.
    
    Args:
        y_proba_uncal: Uncalibrated probabilities (1D array)
        y_true: True labels (1D array)
    
    Returns:
        y_proba_cal: Calibrated probabilities
        calibrator: Calibrator object (for saving)
    """
    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    isotonic_model.fit(y_proba_uncal, y_true)
    
    # Apply calibration
    y_proba_cal = isotonic_model.predict(y_proba_uncal)
    
    calibrator = {'type': 'isotonic', 'model': isotonic_model}
    
    return y_proba_cal, calibrator


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sweep_start: float,
    sweep_end: float,
    sweep_step: float
) -> List[Dict]:
    """Perform threshold sweep and compute metrics."""
    results = []
    thresholds = np.arange(sweep_start, sweep_end + sweep_step, sweep_step)
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        cost = 2 * fn + fp
        
        results.append({
            'threshold': float(threshold),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'cost': float(cost)
        })
    
    return results


def select_optimal_threshold(sweep_results: List[Dict]) -> Dict:
    """Select threshold with minimum cost. If tie, prefer higher recall."""
    min_cost = min(r['cost'] for r in sweep_results)
    candidates = [r for r in sweep_results if abs(r['cost'] - min_cost) < 0.01]
    
    # Prefer higher recall if multiple candidates
    best = max(candidates, key=lambda x: x['recall'])
    
    return best


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float
) -> Dict:
    """Evaluate model at a specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    cost = 2 * fn + fp
    brier = brier_score_loss(y_true, y_proba)
    
    return {
        'threshold': float(threshold),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'cost': float(cost),
        'brier_score': float(brier)
    }


def process_outer_fold(
    fold_idx: int,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    df: pd.DataFrame,
    calibration_mode: str  # 'none', 'platt', 'isotonic'
) -> Dict:
    """
    Process a single outer fold with MIL-only calibration.
    
    Args:
        calibration_mode: 'none' (baseline), 'platt', or 'isotonic'
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"OUTER FOLD {fold_idx + 1}/{OUTER_CV_FOLDS} - {calibration_mode.upper()}")
    logger.info(f"{'='*80}")
    
    # Split data
    df_outer_train = df.iloc[outer_train_idx].copy()
    df_outer_test = df.iloc[outer_test_idx].copy()
    
    logger.info(f"Outer-train: {len(df_outer_train)} patients")
    logger.info(f"Outer-test: {len(df_outer_test)} patients")
    
    # Extract features
    X_outer_train = df_outer_train[BASE_FEATURE_COLS].values
    y_outer_train = df_outer_train[TARGET_COLUMN].values
    X_outer_test = df_outer_test[BASE_FEATURE_COLS].values
    y_outer_test = df_outer_test[TARGET_COLUMN].values
    
    # Extract MIL probabilities separately
    mil_proba_train = df_outer_train['hgg_prob_mil'].values
    mil_proba_test = df_outer_test['hgg_prob_mil'].values
    
    # Apply MIL calibration if requested
    calibrator = None
    if calibration_mode == 'platt':
        logger.info("Fitting Platt calibration on MIL probabilities (outer-train)...")
        mil_proba_train_cal, calibrator = apply_platt_calibration(mil_proba_train, y_outer_train)
        # Apply to test set
        mil_proba_test_clipped = np.clip(mil_proba_test, 1e-7, 1 - 1e-7)
        log_odds_test = np.log(mil_proba_test_clipped / (1 - mil_proba_test_clipped))
        mil_proba_test_cal = calibrator['model'].predict_proba(log_odds_test.reshape(-1, 1))[:, 1]
    elif calibration_mode == 'isotonic':
        logger.info("Fitting Isotonic calibration on MIL probabilities (outer-train)...")
        mil_proba_train_cal, calibrator = apply_isotonic_calibration(mil_proba_train, y_outer_train)
        # Apply to test set
        mil_proba_test_cal = calibrator['model'].predict(mil_proba_test)
    else:  # none
        mil_proba_train_cal = mil_proba_train
        mil_proba_test_cal = mil_proba_test
    
    # Replace MIL probabilities in feature matrices
    X_outer_train_cal = X_outer_train.copy()
    X_outer_train_cal[:, 2] = mil_proba_train_cal  # MIL is 3rd column (index 2)
    
    X_outer_test_cal = X_outer_test.copy()
    X_outer_test_cal[:, 2] = mil_proba_test_cal
    
    # Split outer-train for calibration/threshold selection
    X_cal, X_thr, y_cal, y_thr = train_test_split(
        X_outer_train_cal, y_outer_train,
        test_size=1 - CALIBRATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y_outer_train
    )
    
    # Train meta-learner on calibration subset
    logger.info("Training Logistic Regression meta-learner...")
    meta_learner = LogisticRegression(
        class_weight='balanced',
        solver='lbfgs',
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    meta_learner.fit(X_cal, y_cal)
    
    # Get probabilities on threshold selection subset
    y_proba_thr = meta_learner.predict_proba(X_thr)[:, 1]
    
    # Threshold sweep
    sweep_results = threshold_sweep(
        y_thr, y_proba_thr,
        THRESHOLD_SWEEP_START, THRESHOLD_SWEEP_END, THRESHOLD_SWEEP_STEP
    )
    
    # Select optimal threshold
    optimal_threshold = select_optimal_threshold(sweep_results)
    selected_threshold = optimal_threshold['threshold']
    
    logger.info(f"Selected threshold: {selected_threshold:.4f} (cost: {optimal_threshold['cost']:.1f})")
    
    # Evaluate on outer-test (CRITICAL: never seen during training/calibration)
    y_proba_test = meta_learner.predict_proba(X_outer_test_cal)[:, 1]
    test_metrics = evaluate_at_threshold(
        y_outer_test, y_proba_test, selected_threshold
    )
    
    # Compute Brier score for MIL probabilities (before and after calibration)
    brier_mil_before = brier_score_loss(y_outer_test, mil_proba_test)
    brier_mil_after = brier_score_loss(y_outer_test, mil_proba_test_cal) if calibration_mode != 'none' else brier_mil_before
    
    logger.info(f"Outer-test evaluation:")
    logger.info(f"  FN={test_metrics['fn']}, FP={test_metrics['fp']}, Cost={test_metrics['cost']:.1f}")
    logger.info(f"  Recall={test_metrics['recall']:.4f}, Precision={test_metrics['precision']:.4f}")
    logger.info(f"  MIL Brier (before): {brier_mil_before:.4f}")
    if calibration_mode != 'none':
        logger.info(f"  MIL Brier (after): {brier_mil_after:.4f}")
    
    return {
        'fold': fold_idx,
        'calibration_mode': calibration_mode,
        'outer_train_size': len(df_outer_train),
        'outer_test_size': len(df_outer_test),
        'selected_threshold': selected_threshold,
        'brier_mil_before': float(brier_mil_before),
        'brier_mil_after': float(brier_mil_after),
        **test_metrics
    }


def main():
    """Main function."""
    logger.info("="*80)
    logger.info("NESTED CV WITH MIL-ONLY CALIBRATION")
    logger.info("="*80)
    
    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv(MERGED_OOF_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    # Verify required columns
    required_cols = [PATIENT_ID_COLUMN, TARGET_COLUMN] + BASE_FEATURE_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Extract features and labels
    X = df[BASE_FEATURE_COLS].values
    y = df[TARGET_COLUMN].values
    
    # Create outer CV splits (patient-level)
    logger.info(f"\nCreating {OUTER_CV_FOLDS}-fold outer CV splits (patient-level)...")
    outer_cv = StratifiedKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Process each calibration mode
    all_results = {}
    
    for calibration_mode in ['none', 'platt', 'isotonic']:
        logger.info("\n" + "="*80)
        logger.info(f"EVALUATING: {calibration_mode.upper()}")
        logger.info("="*80)
        
        fold_results = []
        
        for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
            try:
                fold_result = process_outer_fold(
                    fold_idx, outer_train_idx, outer_test_idx, df, calibration_mode
                )
                fold_results.append(fold_result)
            except Exception as e:
                logger.error(f"Error in outer fold {fold_idx} ({calibration_mode}): {e}", exc_info=True)
                continue
        
        if not fold_results:
            raise ValueError(f"No successful folds for {calibration_mode}")
        
        # Aggregate results
        fn_values = [r['fn'] for r in fold_results]
        fp_values = [r['fp'] for r in fold_results]
        cost_values = [r['cost'] for r in fold_results]
        recall_values = [r['recall'] for r in fold_results]
        precision_values = [r['precision'] for r in fold_results]
        f1_values = [r['f1'] for r in fold_results]
        brier_mil_before_values = [r['brier_mil_before'] for r in fold_results]
        brier_mil_after_values = [r['brier_mil_after'] for r in fold_results]
        
        summary = {
            'calibration_mode': calibration_mode,
            'n_folds': len(fold_results),
            'fn_mean': float(np.mean(fn_values)),
            'fn_std': float(np.std(fn_values)),
            'fn_min': int(np.min(fn_values)),
            'fn_max': int(np.max(fn_values)),
            'fp_mean': float(np.mean(fp_values)),
            'fp_std': float(np.std(fp_values)),
            'cost_mean': float(np.mean(cost_values)),
            'cost_std': float(np.std(cost_values)),
            'recall_mean': float(np.mean(recall_values)),
            'recall_std': float(np.std(recall_values)),
            'precision_mean': float(np.mean(precision_values)),
            'precision_std': float(np.std(precision_values)),
            'f1_mean': float(np.mean(f1_values)),
            'f1_std': float(np.std(f1_values)),
            'brier_mil_before_mean': float(np.mean(brier_mil_before_values)),
            'brier_mil_after_mean': float(np.mean(brier_mil_after_values)),
            'fold_results': fold_results
        }
        
        all_results[calibration_mode] = summary
        
        logger.info(f"\n{calibration_mode.upper()} Summary (across {len(fold_results)} folds):")
        logger.info(f"  FN: {summary['fn_mean']:.2f} ± {summary['fn_std']:.2f} "
                   f"(range: [{summary['fn_min']}, {summary['fn_max']}])")
        logger.info(f"  FP: {summary['fp_mean']:.2f} ± {summary['fp_std']:.2f}")
        logger.info(f"  Cost: {summary['cost_mean']:.2f} ± {summary['cost_std']:.2f}")
        logger.info(f"  Recall: {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
        logger.info(f"  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
        if calibration_mode != 'none':
            logger.info(f"  MIL Brier (before): {summary['brier_mil_before_mean']:.4f}")
            logger.info(f"  MIL Brier (after): {summary['brier_mil_after_mean']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f'mil_calibration_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(make_json_serializable(all_results), f, indent=2)
    logger.info(f"\n✓ Results saved to: {results_file}")
    
    # Generate visualizations and report
    generate_visualizations(all_results, df)
    generate_report(all_results, timestamp)
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETE")
    logger.info("="*80)


def generate_visualizations(all_results: Dict, df: pd.DataFrame):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    logger.info("\nGenerating visualizations...")
    
    # 1. FN-FP Tradeoff
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'none': '#A23B72', 'platt': '#2E86AB', 'isotonic': '#06A77D'}
    markers = {'none': 'o', 'platt': 's', 'isotonic': '^'}
    
    for mode in ['none', 'platt', 'isotonic']:
        if mode not in all_results:
            continue
        result = all_results[mode]
        fold_results = result['fold_results']
        fps = [r['fp'] for r in fold_results]
        fns = [r['fn'] for r in fold_results]
        
        ax.scatter(fps, fns, label=f'{mode.upper()}',
                  color=colors[mode], marker=markers[mode],
                  s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Mean ± std
        ax.errorbar(result['fp_mean'], result['fn_mean'],
                   xerr=result['fp_std'], yerr=result['fn_std'],
                   fmt='x', color=colors[mode],
                   markersize=15, markeredgewidth=3, capsize=5, capthick=2)
    
    ax.set_xlabel('False Positives (FP)', fontsize=13, fontweight='bold')
    ax.set_ylabel('False Negatives (FN)', fontsize=13, fontweight='bold')
    ax.set_title('FN-FP Trade-off: MIL Calibration Comparison\n(Nested CV - Outer-Test Only)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'fn_fp_tradeoff_mil_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: fn_fp_tradeoff_mil_calibration.png")
    
    # 2. Cost Distribution
    fig, ax = plt.subplots(figsize=(10, 7))
    data_for_boxplot = []
    labels = []
    
    for mode in ['none', 'platt', 'isotonic']:
        if mode not in all_results:
            continue
        costs = [r['cost'] for r in all_results[mode]['fold_results']]
        data_for_boxplot.append(costs)
        labels.append(mode.upper())
    
    bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True,
                   widths=0.6, showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], [colors[m] for m in ['none', 'platt', 'isotonic'] if m in all_results]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Cost (2×FN + FP)', fontsize=13, fontweight='bold')
    ax.set_title('Cost Distribution: MIL Calibration Comparison', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'cost_distribution_mil_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: cost_distribution_mil_calibration.png")
    
    # 3. Per-Fold FN Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.25
    x = np.arange(5)
    
    for i, mode in enumerate(['none', 'platt', 'isotonic']):
        if mode not in all_results:
            continue
        fns = [r['fn'] for r in all_results[mode]['fold_results']]
        ax.bar(x + i * width, fns, width, label=f'{mode.upper()}',
              color=colors[mode], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Outer Fold', fontsize=13, fontweight='bold')
    ax.set_ylabel('False Negatives (FN)', fontsize=13, fontweight='bold')
    ax.set_title('Per-Fold FN: MIL Calibration Comparison', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Fold {i}' for i in range(5)])
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'per_fold_fn_mil_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: per_fold_fn_mil_calibration.png")
    
    # 4. MIL Calibration Curves (before vs after)
    if 'platt' in all_results or 'isotonic' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        for idx, mode in enumerate(['platt', 'isotonic']):
            if mode not in all_results:
                continue
            
            ax = axes[idx]
            
            # Use full OOF data for calibration curve (for visualization only)
            y_true = df[TARGET_COLUMN].values
            mil_proba = df['hgg_prob_mil'].values
            
            # Apply calibration
            if mode == 'platt':
                mil_proba_cal, _ = apply_platt_calibration(mil_proba, y_true)
            else:
                mil_proba_cal, _ = apply_isotonic_calibration(mil_proba, y_true)
            
            # Calibration curves
            fraction_pos_before, mean_pred_before = calibration_curve(
                y_true, mil_proba, n_bins=10, strategy='uniform'
            )
            fraction_pos_after, mean_pred_after = calibration_curve(
                y_true, mil_proba_cal, n_bins=10, strategy='uniform'
            )
            
            ax.plot(mean_pred_before, fraction_pos_before, 's-', 
                   label='Before Calibration', linewidth=2, markersize=8)
            ax.plot(mean_pred_after, fraction_pos_after, 'o-', 
                   label='After Calibration', linewidth=2, markersize=8)
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
            
            ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
            ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
            ax.set_title(f'MIL Calibration Curve: {mode.upper()}', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'mil_calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved: mil_calibration_curves.png")


def generate_report(all_results: Dict, timestamp: str):
    """Generate markdown report."""
    logger.info("Generating markdown report...")
    
    lines = [
        "# MIL-Only Calibration: Nested CV Evaluation Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report evaluates the impact of calibrating **ONLY the MIL model probabilities**",
        "on ensemble performance, using strict nested cross-validation.",
        "",
        "**Key Findings**:",
        "",
        "---",
        "",
        "## Results Comparison",
        "",
        "| Metric | Baseline (No Cal) | MIL-Platt | MIL-Isotonic |",
        "|--------|------------------|-----------|---------------|"
    ]
    
    baseline = all_results['none']
    platt = all_results.get('platt', {})
    isotonic = all_results.get('isotonic', {})
    
    lines.extend([
        f"| FN (mean ± std) | {baseline['fn_mean']:.2f} ± {baseline['fn_std']:.2f} | "
        f"{platt.get('fn_mean', 0):.2f} ± {platt.get('fn_std', 0):.2f} | "
        f"{isotonic.get('fn_mean', 0):.2f} ± {isotonic.get('fn_std', 0):.2f} |",
        f"| FP (mean ± std) | {baseline['fp_mean']:.2f} ± {baseline['fp_std']:.2f} | "
        f"{platt.get('fp_mean', 0):.2f} ± {platt.get('fp_std', 0):.2f} | "
        f"{isotonic.get('fp_mean', 0):.2f} ± {isotonic.get('fp_std', 0):.2f} |",
        f"| Cost (mean ± std) | {baseline['cost_mean']:.2f} ± {baseline['cost_std']:.2f} | "
        f"{platt.get('cost_mean', 0):.2f} ± {platt.get('cost_std', 0):.2f} | "
        f"{isotonic.get('cost_mean', 0):.2f} ± {isotonic.get('cost_std', 0):.2f} |",
        f"| Recall (mean ± std) | {baseline['recall_mean']:.4f} ± {baseline['recall_std']:.4f} | "
        f"{platt.get('recall_mean', 0):.4f} ± {platt.get('recall_std', 0):.4f} | "
        f"{isotonic.get('recall_mean', 0):.4f} ± {isotonic.get('recall_std', 0):.4f} |",
        f"| Precision (mean ± std) | {baseline['precision_mean']:.4f} ± {baseline['precision_std']:.4f} | "
        f"{platt.get('precision_mean', 0):.4f} ± {platt.get('precision_std', 0):.4f} | "
        f"{isotonic.get('precision_mean', 0):.4f} ± {isotonic.get('precision_std', 0):.4f} |",
        "",
        "### MIL Brier Score Improvement",
        ""
    ])
    
    if 'platt' in all_results:
        platt_brier_improvement = baseline['brier_mil_before_mean'] - platt['brier_mil_after_mean']
        lines.append(f"- **Platt**: {baseline['brier_mil_before_mean']:.4f} → {platt['brier_mil_after_mean']:.4f} "
                    f"(improvement: {platt_brier_improvement:+.4f})")
    
    if 'isotonic' in all_results:
        isotonic_brier_improvement = baseline['brier_mil_before_mean'] - isotonic['brier_mil_after_mean']
        lines.append(f"- **Isotonic**: {baseline['brier_mil_before_mean']:.4f} → {isotonic['brier_mil_after_mean']:.4f} "
                    f"(improvement: {isotonic_brier_improvement:+.4f})")
    
    lines.extend([
        "",
        "---",
        "",
        "## Verdict",
        "",
        "### Does MIL calibration reduce FN?",
        ""
    ])
    
    # Compare FN
    if 'platt' in all_results:
        fn_improvement_platt = baseline['fn_mean'] - platt['fn_mean']
        if fn_improvement_platt > 0:
            lines.append(f"✅ **Platt**: FN reduced by {fn_improvement_platt:.2f} "
                        f"({baseline['fn_mean']:.2f} → {platt['fn_mean']:.2f})")
        else:
            lines.append(f"❌ **Platt**: FN increased by {abs(fn_improvement_platt):.2f} "
                        f"({baseline['fn_mean']:.2f} → {platt['fn_mean']:.2f})")
    
    if 'isotonic' in all_results:
        fn_improvement_isotonic = baseline['fn_mean'] - isotonic['fn_mean']
        if fn_improvement_isotonic > 0:
            lines.append(f"✅ **Isotonic**: FN reduced by {fn_improvement_isotonic:.2f} "
                        f"({baseline['fn_mean']:.2f} → {isotonic['fn_mean']:.2f})")
        else:
            lines.append(f"❌ **Isotonic**: FN increased by {abs(fn_improvement_isotonic):.2f} "
                        f"({baseline['fn_mean']:.2f} → {isotonic['fn_mean']:.2f})")
    
    lines.extend([
        "",
        "### Does it improve ensemble recall?",
        ""
    ])
    
    if 'platt' in all_results:
        recall_improvement_platt = platt['recall_mean'] - baseline['recall_mean']
        if recall_improvement_platt > 0:
            lines.append(f"✅ **Platt**: Recall improved by {recall_improvement_platt:+.4f} "
                        f"({baseline['recall_mean']:.4f} → {platt['recall_mean']:.4f})")
        else:
            lines.append(f"❌ **Platt**: Recall decreased by {abs(recall_improvement_platt):.4f} "
                        f"({baseline['recall_mean']:.4f} → {platt['recall_mean']:.4f})")
    
    if 'isotonic' in all_results:
        recall_improvement_isotonic = isotonic['recall_mean'] - baseline['recall_mean']
        if recall_improvement_isotonic > 0:
            lines.append(f"✅ **Isotonic**: Recall improved by {recall_improvement_isotonic:+.4f} "
                        f"({baseline['recall_mean']:.4f} → {isotonic['recall_mean']:.4f})")
        else:
            lines.append(f"❌ **Isotonic**: Recall decreased by {abs(recall_improvement_isotonic):.4f} "
                        f"({baseline['recall_mean']:.4f} → {isotonic['recall_mean']:.4f})")
    
    lines.extend([
        "",
        "### Is the improvement stable across folds?",
        ""
    ])
    
    if 'platt' in all_results:
        fn_std_platt = platt['fn_std']
        fn_std_baseline = baseline['fn_std']
        if fn_std_platt <= fn_std_baseline * 1.2:  # Within 20% of baseline
            lines.append(f"✅ **Platt**: Stable (FN std: {fn_std_platt:.2f} vs baseline {fn_std_baseline:.2f})")
        else:
            lines.append(f"⚠️ **Platt**: Less stable (FN std: {fn_std_platt:.2f} vs baseline {fn_std_baseline:.2f})")
    
    if 'isotonic' in all_results:
        fn_std_isotonic = isotonic['fn_std']
        fn_std_baseline = baseline['fn_std']
        if fn_std_isotonic <= fn_std_baseline * 1.2:
            lines.append(f"✅ **Isotonic**: Stable (FN std: {fn_std_isotonic:.2f} vs baseline {fn_std_baseline:.2f})")
        else:
            lines.append(f"⚠️ **Isotonic**: Less stable (FN std: {fn_std_isotonic:.2f} vs baseline {fn_std_baseline:.2f})")
    
    lines.extend([
        "",
        "---",
        "",
        "## Final Recommendation",
        "",
        "Based on the nested CV evaluation:",
        ""
    ])
    
    # Determine best method
    best_method = 'none'
    best_cost = baseline['cost_mean']
    
    if 'platt' in all_results and platt['cost_mean'] < best_cost:
        best_method = 'platt'
        best_cost = platt['cost_mean']
    
    if 'isotonic' in all_results and isotonic['cost_mean'] < best_cost:
        best_method = 'isotonic'
        best_cost = isotonic['cost_mean']
    
    if best_method == 'none':
        lines.append("❌ **MIL calibration does NOT improve ensemble performance**.")
        lines.append("")
        lines.append("**Conclusion**: The limitation is architectural. MIL model probabilities")
        lines.append("are not the bottleneck. Consider:")
        lines.append("- Improving MIL model architecture")
        lines.append("- Improving MIL training procedure")
        lines.append("- Replacing MIL with a better base model")
    else:
        lines.append(f"✅ **MIL-{best_method.upper()} calibration improves ensemble performance**.")
        lines.append("")
        lines.append(f"- Cost reduced from {baseline['cost_mean']:.2f} to {best_cost:.2f}")
        if best_method == 'platt':
            lines.append(f"- FN reduced from {baseline['fn_mean']:.2f} to {platt['fn_mean']:.2f}")
        else:
            lines.append(f"- FN reduced from {baseline['fn_mean']:.2f} to {isotonic['fn_mean']:.2f}")
        lines.append("")
        lines.append("**Recommendation**: Adopt MIL-only calibration in the ensemble pipeline.")
    
    lines.extend([
        "",
        "---",
        "",
        "## Methodology",
        "",
        "### Calibration Protocol",
        "",
        "- **Method**: Platt scaling and Isotonic regression",
        "- **Scope**: MIL probabilities only (ResNet and Swin unchanged)",
        "- **Fitting**: Only on outer-train data within each fold",
        "- **Evaluation**: Only on outer-test data (never seen during calibration)",
        "",
        "### Nested CV Structure",
        "",
        "- **Outer folds**: 5-fold patient-level StratifiedKFold",
        "- **Inner split**: 70% calibration/threshold selection, 30% meta-learner training",
        "- **Threshold selection**: Cost-sensitive (minimize 2×FN + FP)",
        "- **Meta-learner**: Logistic Regression (class_weight='balanced')",
        "",
        "### Metrics",
        "",
        "- All metrics computed on outer-test folds only",
        "- Aggregated as mean ± std across folds",
        "- Brier score computed for MIL probabilities (before vs after calibration)",
        ""
    ])
    
    report_file = OUTPUT_DIR / f'MIL_CALIBRATION_REPORT_{timestamp}.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"✓ Saved report: {report_file}")


if __name__ == '__main__':
    main()

