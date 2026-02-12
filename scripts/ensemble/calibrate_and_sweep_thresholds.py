#!/usr/bin/env python3
"""
Probability Calibration and Threshold Re-selection for Ensemble Meta-Learner

This script performs probability calibration (Platt/Isotonic) on the ensemble
meta-learner and re-selects optimal thresholds on calibrated probabilities.

Key scientific constraints:
- Calibration is trained on a subset (70%) of OOF data
- Threshold selection is performed on a disjoint held-out subset (30%)
- This prevents data leakage and ensures valid evaluation

Usage:
    python scripts/ensemble/calibrate_and_sweep_thresholds.py --calibration platt
    python scripts/ensemble/calibrate_and_sweep_thresholds.py --calibration none  # Sanity check
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
MODEL_FILE = Path('ensemble/models/meta_learner_logistic_regression.joblib')
FEATURE_COLUMNS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
TARGET_COLUMN = 'label'


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum(|acc_bin - conf_bin| * n_bin) / N
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for probability discretization
        
    Returns:
        ECE value (lower is better, 0 = perfect calibration)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    n_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            # Average confidence (predicted probability) in this bin
            avg_confidence_in_bin = y_proba[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return float(ece)


def load_data_and_model() -> Tuple[pd.DataFrame, object]:
    """Load OOF predictions and trained meta-learner model."""
    logger.info("Loading data and model...")
    
    # Load OOF predictions
    if not MERGED_OOF_FILE.exists():
        raise FileNotFoundError(
            f"Merged OOF file not found: {MERGED_OOF_FILE}\n"
            f"Please ensure merged_oof_predictions.csv exists."
        )
    
    df = pd.read_csv(MERGED_OOF_FILE)
    logger.info(f"Loaded {len(df)} samples from {MERGED_OOF_FILE}")
    
    # Validate required columns
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")
    
    # Load meta-learner model
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Meta-learner model not found: {MODEL_FILE}\n"
            f"Please train the meta-learner first using train_meta_learner.py"
        )
    
    meta_learner = joblib.load(MODEL_FILE)
    logger.info(f"Loaded meta-learner from {MODEL_FILE}")
    
    return df, meta_learner


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features (X) and target (y) from dataframe."""
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # Validate no missing values
    if np.isnan(X).any():
        raise ValueError("Found NaN values in features")
    if np.isnan(y).any():
        raise ValueError("Found NaN values in target")
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    calibration_fraction: float,
    split_seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform stratified split into calibration and threshold selection sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        calibration_fraction: Fraction for calibration (e.g., 0.70)
        split_seed: Random seed for reproducibility
        
    Returns:
        X_cal, X_thr, y_cal, y_thr
    """
    logger.info(f"Performing stratified split: {calibration_fraction:.1%} calibration, "
                f"{1-calibration_fraction:.1%} threshold selection (seed={split_seed})")
    
    X_cal, X_thr, y_cal, y_thr = train_test_split(
        X, y,
        test_size=1 - calibration_fraction,
        stratify=y,
        random_state=split_seed
    )
    
    logger.info(f"Calibration set: {len(X_cal)} samples")
    logger.info(f"Threshold selection set: {len(X_thr)} samples")
    logger.info(f"Calibration class distribution: {dict(zip(*np.unique(y_cal, return_counts=True)))}")
    logger.info(f"Threshold class distribution: {dict(zip(*np.unique(y_thr, return_counts=True)))}")
    
    return X_cal, X_thr, y_cal, y_thr


def apply_calibration(
    meta_learner: object,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_thr: np.ndarray,
    calibration_mode: str
) -> Tuple[Optional[object], np.ndarray, np.ndarray]:
    """
    Apply calibration to meta-learner probabilities.
    
    Args:
        meta_learner: Trained LogisticRegression meta-learner
        X_cal: Calibration features
        y_cal: Calibration labels
        X_thr: Threshold selection features
        calibration_mode: 'none', 'platt', or 'isotonic'
        
    Returns:
        calibrator (or None), calibrated probabilities for threshold set, uncalibrated probabilities
    """
    # Get uncalibrated probabilities
    y_proba_thr_uncal = meta_learner.predict_proba(X_thr)[:, 1]
    logger.info(f"Uncalibrated probabilities (threshold set): "
                f"min={y_proba_thr_uncal.min():.4f}, max={y_proba_thr_uncal.max():.4f}, "
                f"mean={y_proba_thr_uncal.mean():.4f}")
    
    if calibration_mode == 'none':
        logger.info("Calibration mode: none (using uncalibrated probabilities)")
        return None, y_proba_thr_uncal, y_proba_thr_uncal
    
    # Apply calibration manually (since cv='prefit' not supported in all sklearn versions)
    logger.info(f"Calibration mode: {calibration_mode}")
    logger.info(f"Fitting calibrator on {len(X_cal)} calibration samples...")
    
    # Get uncalibrated probabilities on calibration set
    y_proba_cal_uncal = meta_learner.predict_proba(X_cal)[:, 1]
    
    if calibration_mode == 'platt':
        # Platt scaling: logistic regression on log-odds
        # Transform probabilities to log-odds space
        # Avoid log(0) and log(1) by clipping
        y_proba_cal_uncal_clipped = np.clip(y_proba_cal_uncal, 1e-7, 1 - 1e-7)
        log_odds = np.log(y_proba_cal_uncal_clipped / (1 - y_proba_cal_uncal_clipped))
        
        # Fit logistic regression (Platt scaling)
        platt_model = PlattScaling()
        platt_model.fit(log_odds.reshape(-1, 1), y_cal)
        
        # Apply to threshold set
        y_proba_thr_uncal_clipped = np.clip(y_proba_thr_uncal, 1e-7, 1 - 1e-7)
        log_odds_thr = np.log(y_proba_thr_uncal_clipped / (1 - y_proba_thr_uncal_clipped))
        y_proba_thr_cal = platt_model.predict_proba(log_odds_thr.reshape(-1, 1))[:, 1]
        
        # Store calibrator as a simple wrapper
        calibrator = {'type': 'platt', 'model': platt_model}
        
    elif calibration_mode == 'isotonic':
        # Isotonic regression
        isotonic_model = IsotonicRegression(out_of_bounds='clip')
        isotonic_model.fit(y_proba_cal_uncal, y_cal)
        
        # Apply to threshold set
        y_proba_thr_cal = isotonic_model.transform(y_proba_thr_uncal)
        
        # Store calibrator
        calibrator = {'type': 'isotonic', 'model': isotonic_model}
    logger.info(f"Calibrated probabilities (threshold set): "
                f"min={y_proba_thr_cal.min():.4f}, max={y_proba_thr_cal.max():.4f}, "
                f"mean={y_proba_thr_cal.mean():.4f}")
    
    return calibrator, y_proba_thr_cal, y_proba_thr_uncal


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba_uncal: np.ndarray,
    y_proba_cal: np.ndarray,
    n_bins: int
) -> Dict:
    """Compute Brier score and ECE for both uncalibrated and calibrated probabilities."""
    brier_pre = brier_score_loss(y_true, y_proba_uncal)
    brier_post = brier_score_loss(y_true, y_proba_cal)
    
    ece_pre = compute_ece(y_true, y_proba_uncal, n_bins)
    ece_post = compute_ece(y_true, y_proba_cal, n_bins)
    
    improvement_brier = brier_pre - brier_post
    improvement_ece = ece_pre - ece_post
    
    logger.info(f"Brier Score: {brier_pre:.6f} (pre) -> {brier_post:.6f} (post), "
                f"improvement: {improvement_brier:.6f}")
    logger.info(f"ECE: {ece_pre:.6f} (pre) -> {ece_post:.6f} (post), "
                f"improvement: {improvement_ece:.6f}")
    
    return {
        'brier_pre': float(brier_pre),
        'brier_post': float(brier_post),
        'ece_pre': float(ece_pre),
        'ece_post': float(ece_post),
        'improvement_brier': float(improvement_brier),
        'improvement_ece': float(improvement_ece)
    }


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba_uncal: np.ndarray,
    y_proba_cal: np.ndarray,
    calibration_mode: str,
    n_bins: int,
    save_path: Path
):
    """Generate reliability diagram (calibration curve) comparing pre and post calibration."""
    logger.info(f"Generating reliability diagram with {n_bins} bins...")
    
    # Compute calibration curves
    fraction_pos_uncal, mean_pred_uncal = calibration_curve(
        y_true, y_proba_uncal, n_bins=n_bins, strategy='uniform'
    )
    fraction_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_proba_cal, n_bins=n_bins, strategy='uniform'
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot calibration curves
    ax.plot(mean_pred_uncal, fraction_pos_uncal, 'o-', label='Uncalibrated', 
            color='red', linewidth=2, markersize=6)
    ax.plot(mean_pred_cal, fraction_pos_cal, 's-', label=f'Calibrated ({calibration_mode})',
            color='blue', linewidth=2, markersize=6)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Reliability Diagram (Calibration Curve)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved reliability diagram to: {save_path}")


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sweep_start: float,
    sweep_end: float,
    sweep_step: float
) -> List[Dict]:
    """
    Perform threshold sweep and compute metrics at each threshold.
    
    Returns:
        List of dictionaries with metrics at each threshold
    """
    logger.info(f"Performing threshold sweep: [{sweep_start:.2f}, {sweep_end:.2f}] step {sweep_step:.2f}")
    
    thresholds = np.arange(sweep_start, sweep_end + sweep_step/2, sweep_step)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Edge case: all predictions same class
            if y_pred.sum() == 0:
                tn, fp, fn, tp = len(y_true) - y_true.sum(), 0, y_true.sum(), 0
            else:
                tn, fp, fn, tp = 0, len(y_true) - y_true.sum(), 0, y_true.sum()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        results.append({
            'threshold': float(threshold),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy)
        })
    
    logger.info(f"Completed threshold sweep: {len(results)} thresholds evaluated")
    return results


def select_recommended_thresholds(
    sweep_results: List[Dict],
    recall_target: float
) -> Dict:
    """
    Select recommended thresholds using two policies:
    1. Balanced: maximize F1
    2. High-sensitivity: max Precision subject to Recall >= recall_target
    
    Returns:
        Dictionary with recommended thresholds and their metrics
    """
    logger.info("Selecting recommended thresholds...")
    
    # Policy A: Balanced (maximize F1)
    best_f1_idx = np.argmax([r['f1_score'] for r in sweep_results])
    balanced = sweep_results[best_f1_idx].copy()
    logger.info(f"Balanced threshold: {balanced['threshold']:.4f} "
                f"(F1={balanced['f1_score']:.4f}, Precision={balanced['precision']:.4f}, "
                f"Recall={balanced['recall']:.4f})")
    
    # Policy B: High-sensitivity (Recall >= recall_target, maximize Precision)
    candidates = [r for r in sweep_results if r['recall'] >= recall_target]
    
    if candidates:
        best_precision_idx = np.argmax([c['precision'] for c in candidates])
        high_sensitivity = candidates[best_precision_idx].copy()
        logger.info(f"High-sensitivity threshold: {high_sensitivity['threshold']:.4f} "
                    f"(Precision={high_sensitivity['precision']:.4f}, "
                    f"Recall={high_sensitivity['recall']:.4f}, "
                    f"FN={high_sensitivity['fn']})")
    else:
        # Fallback: highest recall achievable
        best_recall_idx = np.argmax([r['recall'] for r in sweep_results])
        high_sensitivity = sweep_results[best_recall_idx].copy()
        logger.warning(f"No threshold met recall target {recall_target:.2f}. "
                      f"Using highest recall: {high_sensitivity['recall']:.4f}")
    
    return {
        'balanced': balanced,
        'high_sensitivity': high_sensitivity
    }


def save_outputs(
    output_dir: Path,
    calibration_mode: str,
    calibration_summary: Dict,
    sweep_results: List[Dict],
    recommended_thresholds: Dict,
    args: argparse.Namespace
):
    """Save all outputs to the run directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving outputs to: {output_dir}")
    
    # Save calibration summary
    summary_file = output_dir / 'calibration_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(calibration_summary, f, indent=2)
    logger.info(f"Saved calibration summary to: {summary_file}")
    
    # Save threshold sweep results
    sweep_file = output_dir / f'threshold_sweep_{calibration_mode}.json'
    with open(sweep_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    logger.info(f"Saved threshold sweep to: {sweep_file}")
    
    # Save recommended thresholds
    thresholds_file = output_dir / f'recommended_thresholds_{calibration_mode}.json'
    output_data = {
        'calibration_mode': calibration_mode,
        'timestamp': datetime.now().isoformat(),
        'args': {
            'calibration': args.calibration,
            'split_seed': args.split_seed,
            'calibration_fraction': args.calibration_fraction,
            'recall_target': args.recall_target,
            'sweep_start': args.sweep_start,
            'sweep_end': args.sweep_end,
            'sweep_step': args.sweep_step,
            'n_bins': args.n_bins
        },
        'thresholds': recommended_thresholds
    }
    with open(thresholds_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved recommended thresholds to: {thresholds_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Probability calibration and threshold re-selection for ensemble meta-learner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--calibration',
        type=str,
        choices=['none', 'platt', 'isotonic'],
        default='platt',
        help='Calibration method (default: platt)'
    )
    parser.add_argument(
        '--split-seed',
        type=int,
        default=42,
        help='Random seed for calibration/threshold split (default: 42)'
    )
    parser.add_argument(
        '--calibration-fraction',
        type=float,
        default=0.70,
        help='Fraction of data for calibration (default: 0.70, so 30%% held out)'
    )
    parser.add_argument(
        '--recall-target',
        type=float,
        default=0.94,
        help='Target recall for high-sensitivity threshold (default: 0.94)'
    )
    parser.add_argument(
        '--sweep-start',
        type=float,
        default=0.05,
        help='Start threshold for sweep (default: 0.05)'
    )
    parser.add_argument(
        '--sweep-end',
        type=float,
        default=0.95,
        help='End threshold for sweep (default: 0.95)'
    )
    parser.add_argument(
        '--sweep-step',
        type=float,
        default=0.01,
        help='Step size for threshold sweep (default: 0.01)'
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=10,
        help='Number of bins for calibration curve and ECE (default: 10)'
    )
    parser.add_argument(
        '--out-root',
        type=str,
        default='ensemble/results/calibration',
        help='Root directory for outputs (default: ensemble/results/calibration)'
    )
    parser.add_argument(
        '--save-calibrator',
        action='store_true',
        help='Save calibrator joblib file (only if calibration != none)'
    )
    parser.add_argument(
        '--plot-format',
        type=str,
        choices=['png'],
        default='png',
        help='Plot format (default: png)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Probability Calibration and Threshold Re-selection")
    logger.info("=" * 80)
    logger.info(f"Calibration mode: {args.calibration}")
    logger.info(f"Split seed: {args.split_seed}")
    logger.info(f"Calibration fraction: {args.calibration_fraction:.2f}")
    logger.info(f"Recall target: {args.recall_target:.2f}")
    
    # Load data and model
    df, meta_learner = load_data_and_model()
    X, y = prepare_features_and_target(df)
    
    # Split data
    X_cal, X_thr, y_cal, y_thr = split_data(
        X, y, args.calibration_fraction, args.split_seed
    )
    
    # Apply calibration
    calibrator, y_proba_thr_cal, y_proba_thr_uncal = apply_calibration(
        meta_learner, X_cal, y_cal, X_thr, args.calibration
    )
    
    # Compute calibration metrics
    metrics = compute_calibration_metrics(
        y_thr, y_proba_thr_uncal, y_proba_thr_cal, args.n_bins
    )
    
    # Generate reliability diagram
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_root = Path(args.out_root)
    run_dir = output_root / f"{timestamp}_{args.calibration}_seed{args.split_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    reliability_plot = run_dir / f'reliability_diagram_{args.calibration}.{args.plot_format}'
    plot_reliability_diagram(
        y_thr, y_proba_thr_uncal, y_proba_thr_cal,
        args.calibration, args.n_bins, reliability_plot
    )
    
    # Threshold sweep on calibrated probabilities
    sweep_results = threshold_sweep(
        y_thr, y_proba_thr_cal,
        args.sweep_start, args.sweep_end, args.sweep_step
    )
    
    # Select recommended thresholds
    recommended = select_recommended_thresholds(sweep_results, args.recall_target)
    
    # Prepare calibration summary
    calibration_summary = {
        'timestamp': timestamp,
        'calibration_mode': args.calibration,
        'split_seed': args.split_seed,
        'n_calibration': len(X_cal),
        'n_threshold_selection': len(X_thr),
        **metrics,
        'args': {
            'calibration_fraction': args.calibration_fraction,
            'recall_target': args.recall_target,
            'sweep_start': args.sweep_start,
            'sweep_end': args.sweep_end,
            'sweep_step': args.sweep_step,
            'n_bins': args.n_bins
        }
    }
    
    # Save calibrator if requested
    if args.save_calibrator and calibrator is not None:
        calibrator_file = run_dir / f'calibrator_{args.calibration}.joblib'
        joblib.dump(calibrator, calibrator_file)
        logger.info(f"Saved calibrator to: {calibrator_file}")
        logger.info("Note: Use apply_calibrator_to_proba() function to apply saved calibrator")
    
    # Save all outputs
    save_outputs(run_dir, args.calibration, calibration_summary, sweep_results, recommended, args)
    
    logger.info("=" * 80)
    logger.info("Calibration and threshold re-selection complete!")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

