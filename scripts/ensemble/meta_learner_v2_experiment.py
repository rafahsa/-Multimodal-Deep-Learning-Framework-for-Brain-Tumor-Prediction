#!/usr/bin/env python3
"""
Meta-Learner V2 Experiment

This script tests alternative meta-learners against the baseline LogisticRegression
to determine if a better meta-learner can reduce both FN and FP.

The experiment follows the same protocol as the baseline:
- Train on full OOF set (n=285)
- Split 70/30 (seed=42) for calibration/threshold selection
- Apply Platt calibration
- Select threshold by cost (2*FN + FP)
- Evaluate on full OOF set

Usage:
    python scripts/ensemble/meta_learner_v2_experiment.py
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Dict as TypedDict

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)
from sklearn.calibration import CalibratedClassifierCV

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
FEATURE_COLUMNS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
TARGET_COLUMN = 'label'

# Output directories
RESULTS_DIR = Path('ensemble/results/meta_learner_v2')
MODELS_DIR = Path('ensemble/models/meta_learner_v2')

# Baseline metrics (from threshold 0.35 on full OOF)
BASELINE_FN = 11
BASELINE_FP = 41
BASELINE_COST = 2 * BASELINE_FN + BASELINE_FP  # 63

# Experiment parameters
CALIBRATION_SEED = 42
CALIBRATION_FRACTION = 0.7
THRESHOLD_SWEEP_START = 0.05
THRESHOLD_SWEEP_END = 0.95
THRESHOLD_SWEEP_STEP = 0.01


def load_data() -> pd.DataFrame:
    """Load OOF predictions from CSV."""
    logger.info(f"Loading data from: {MERGED_OOF_FILE}")
    
    if not MERGED_OOF_FILE.exists():
        raise FileNotFoundError(f"OOF predictions file not found: {MERGED_OOF_FILE}")
    
    df = pd.read_csv(MERGED_OOF_FILE)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Features: {FEATURE_COLUMNS}")
    logger.info(f"Target distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
    
    return df


def apply_platt_calibration(
    meta_learner: object,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_thr: np.ndarray
) -> Tuple[Dict, np.ndarray]:
    """
    Apply Platt calibration to meta-learner probabilities.
    
    Handles both models with predict_proba (LogisticRegression, XGBoost) and
    models with decision_function (LinearSVC).
    
    Returns:
        calibrator dict, calibrated probabilities for threshold set
    """
    from sklearn.linear_model import LogisticRegression as PlattScaling
    
    # Get uncalibrated scores/probabilities on calibration set
    if hasattr(meta_learner, 'predict_proba'):
        y_scores_cal = meta_learner.predict_proba(X_cal)[:, 1]
    elif hasattr(meta_learner, 'decision_function'):
        # For LinearSVC, use decision_function and convert to probability-like scores
        decision_scores = meta_learner.decision_function(X_cal)
        # Convert to [0, 1] range using sigmoid
        y_scores_cal = 1 / (1 + np.exp(-decision_scores))
    else:
        raise ValueError("Model must have either predict_proba or decision_function")
    
    # Clip to avoid log(0) and log(1)
    y_scores_cal_clipped = np.clip(y_scores_cal, 1e-7, 1 - 1e-7)
    log_odds = np.log(y_scores_cal_clipped / (1 - y_scores_cal_clipped))
    
    # Fit Platt scaling
    platt_model = PlattScaling()
    platt_model.fit(log_odds.reshape(-1, 1), y_cal)
    
    # Apply to threshold set
    if hasattr(meta_learner, 'predict_proba'):
        y_scores_thr = meta_learner.predict_proba(X_thr)[:, 1]
    elif hasattr(meta_learner, 'decision_function'):
        decision_scores = meta_learner.decision_function(X_thr)
        y_scores_thr = 1 / (1 + np.exp(-decision_scores))
    else:
        raise ValueError("Model must have either predict_proba or decision_function")
    
    y_scores_thr_clipped = np.clip(y_scores_thr, 1e-7, 1 - 1e-7)
    log_odds_thr = np.log(y_scores_thr_clipped / (1 - y_scores_thr_clipped))
    y_proba_thr_cal = platt_model.predict_proba(log_odds_thr.reshape(-1, 1))[:, 1]
    
    calibrator = {'type': 'platt', 'model': platt_model}
    
    return calibrator, y_proba_thr_cal


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


def evaluate_threshold(
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
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
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
        'cost': float(2 * fn + fp),
        'lgg_precision': float(report['0']['precision']),
        'lgg_recall': float(report['0']['recall']),
        'lgg_f1': float(report['0']['f1-score']),
        'hgg_precision': float(report['1']['precision']),
        'hgg_recall': float(report['1']['recall']),
        'hgg_f1': float(report['1']['f1-score'])
    }


def train_baseline_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Train baseline LogisticRegression (reproduction)."""
    logger.info("Training baseline LogisticRegression...")
    
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    
    model = LogisticRegression(
        C=1.0,
        class_weight=class_weights,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X, y)
    
    logger.info("✓ Baseline LogisticRegression trained")
    return model


def train_tuned_lr(X: np.ndarray, y: np.ndarray) -> List[Tuple[str, LogisticRegression]]:
    """Train tuned LogisticRegression models with grid search."""
    logger.info("Training tuned LogisticRegression models...")
    
    param_grid = {
        'C': [0.1, 1, 10],
        'class_weight': [None, 'balanced']
    }
    
    base_model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid_search.fit(X, y)
    
    results = []
    for params in grid_search.cv_results_['params']:
        model = LogisticRegression(
            C=params['C'],
            class_weight=params['class_weight'],
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        model.fit(X, y)
        
        c_str = str(params['C'])
        cw_str = 'balanced' if params['class_weight'] == 'balanced' else 'none'
        name = f"LogisticRegression_C{c_str}_{cw_str}"
        
        results.append((name, model))
        logger.info(f"  ✓ {name} trained")
    
    return results


def train_linearsvc(X: np.ndarray, y: np.ndarray) -> LinearSVC:
    """Train LinearSVC (will use decision_function for calibration)."""
    logger.info("Training LinearSVC...")
    
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    
    model = LinearSVC(
        C=1.0,
        class_weight=class_weights,
        max_iter=10000,
        random_state=42
    )
    model.fit(X, y)
    
    logger.info("✓ LinearSVC trained")
    return model


def train_xgboost_models(X: np.ndarray, y: np.ndarray) -> List[Tuple[str, object]]:
    """Train XGBoost models with small conservative grid."""
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available, skipping XGBoost models")
        return []
    
    logger.info("Training XGBoost models...")
    
    param_grid = {
        'max_depth': [3, 4],
        'learning_rate': [0.1],
        'n_estimators': [50, 100]
    }
    
    results = []
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for n_estimators in param_grid['n_estimators']:
                model = xgb.XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
                model.fit(X, y)
                
                name = f"XGBoost_depth{max_depth}_lr{learning_rate}_n{n_estimators}"
                results.append((name, model))
                logger.info(f"  ✓ {name} trained")
    
    return results


def evaluate_model(
    model: object,
    model_name: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_thr: np.ndarray,
    y_thr: np.ndarray
) -> Dict:
    """
    Evaluate a meta-learner model following the baseline protocol.
    
    Returns:
        Dictionary with model name, selected threshold, and full OOF evaluation metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*80}")
    
    # Apply Platt calibration
    calibrator, y_proba_thr_cal = apply_platt_calibration(
        model, X_cal, y_cal, X_thr
    )
    
    # Threshold sweep on threshold selection set
    sweep_results = threshold_sweep(
        y_thr, y_proba_thr_cal,
        THRESHOLD_SWEEP_START, THRESHOLD_SWEEP_END, THRESHOLD_SWEEP_STEP
    )
    
    # Select optimal threshold
    optimal_thr_result = select_optimal_threshold(sweep_results)
    selected_threshold = optimal_thr_result['threshold']
    
    logger.info(f"Selected threshold: {selected_threshold:.4f} (cost={optimal_thr_result['cost']:.1f}, "
                f"FN={optimal_thr_result['fn']}, FP={optimal_thr_result['fp']})")
    
    # Apply calibration to full OOF set
    if hasattr(model, 'predict_proba'):
        y_proba_full_uncal = model.predict_proba(X_full)[:, 1]
    elif hasattr(model, 'decision_function'):
        decision_scores = model.decision_function(X_full)
        y_proba_full_uncal = 1 / (1 + np.exp(-decision_scores))
    else:
        raise ValueError("Model must have either predict_proba or decision_function")
    
    y_proba_full_uncal_clipped = np.clip(y_proba_full_uncal, 1e-7, 1 - 1e-7)
    log_odds_full = np.log(y_proba_full_uncal_clipped / (1 - y_proba_full_uncal_clipped))
    y_proba_full_cal = calibrator['model'].predict_proba(log_odds_full.reshape(-1, 1))[:, 1]
    
    # Evaluate on full OOF set
    full_eval = evaluate_threshold(y_full, y_proba_full_cal, selected_threshold)
    
    logger.info(f"Full OOF evaluation: FN={full_eval['fn']}, FP={full_eval['fp']}, "
                f"Cost={full_eval['cost']:.1f}, Recall={full_eval['recall']:.4f}")
    
    return {
        'model_name': model_name,
        'model_type': model_name.split('_')[0],
        'selected_threshold': selected_threshold,
        **full_eval
    }


def main():
    """Main experiment function."""
    logger.info("="*80)
    logger.info("META-LEARNER V2 EXPERIMENT")
    logger.info("="*80)
    
    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data()
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # Split for calibration/threshold selection (same protocol as baseline)
    X_cal, X_thr, y_cal, y_thr = train_test_split(
        X, y, test_size=1-CALIBRATION_FRACTION, random_state=CALIBRATION_SEED, stratify=y
    )
    logger.info(f"Calibration set: {len(X_cal)} samples")
    logger.info(f"Threshold selection set: {len(X_thr)} samples")
    
    # Baseline metrics (from known results)
    baseline_result = {
        'model_name': 'Baseline_LogisticRegression',
        'model_type': 'LogisticRegression',
        'selected_threshold': 0.35,
        'tn': 34,
        'fp': 41,
        'fn': 11,
        'tp': 199,
        'precision': 0.8292,
        'recall': 0.9476,
        'f1': 0.8844,
        'accuracy': 0.8175,
        'specificity': 0.4533,
        'cost': BASELINE_COST,
        'lgg_precision': 0.7556,
        'lgg_recall': 0.4533,
        'lgg_f1': 0.5667,
        'hgg_precision': 0.8292,
        'hgg_recall': 0.9476,
        'hgg_f1': 0.8844
    }
    
    all_results = [baseline_result]
    
    # Train and evaluate baseline reproduction (sanity check)
    logger.info("\n" + "="*80)
    logger.info("BASELINE REPRODUCTION (Sanity Check)")
    logger.info("="*80)
    try:
        baseline_model = train_baseline_lr(X, y)
        baseline_repro = evaluate_model(
            baseline_model, 'Baseline_Reproduction',
            X, y, X_cal, y_cal, X_thr, y_thr
        )
        all_results.append(baseline_repro)
        
        # Save baseline reproduction model
        joblib.dump(baseline_model, MODELS_DIR / 'baseline_reproduction.joblib')
    except Exception as e:
        logger.error(f"Failed to train baseline reproduction: {e}")
    
    # Train and evaluate tuned LogisticRegression
    logger.info("\n" + "="*80)
    logger.info("TUNED LOGISTIC REGRESSION")
    logger.info("="*80)
    try:
        tuned_lr_models = train_tuned_lr(X, y)
        for name, model in tuned_lr_models:
            try:
                result = evaluate_model(
                    model, name, X, y, X_cal, y_cal, X_thr, y_thr
                )
                all_results.append(result)
                
                # Save model
                safe_name = name.replace('/', '_').replace('\\', '_')
                joblib.dump(model, MODELS_DIR / f'{safe_name}.joblib')
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
    except Exception as e:
        logger.error(f"Failed to train tuned LogisticRegression: {e}")
    
    # Train and evaluate LinearSVC
    logger.info("\n" + "="*80)
    logger.info("LINEAR SVC")
    logger.info("="*80)
    try:
        svc_model = train_linearsvc(X, y)
        svc_result = evaluate_model(
            svc_model, 'LinearSVC', X, y, X_cal, y_cal, X_thr, y_thr
        )
        all_results.append(svc_result)
        
        # Save model
        joblib.dump(svc_model, MODELS_DIR / 'linearsvc.joblib')
    except Exception as e:
        logger.error(f"Failed to train/evaluate LinearSVC: {e}")
    
    # Train and evaluate XGBoost
    if XGBOOST_AVAILABLE:
        logger.info("\n" + "="*80)
        logger.info("XGBOOST")
        logger.info("="*80)
        try:
            xgb_models = train_xgboost_models(X, y)
            for name, model in xgb_models:
                try:
                    result = evaluate_model(
                        model, name, X, y, X_cal, y_cal, X_thr, y_thr
                    )
                    all_results.append(result)
                    
                    # Save model
                    safe_name = name.replace('/', '_').replace('\\', '_')
                    joblib.dump(model, MODELS_DIR / f'{safe_name}.joblib')
                except Exception as e:
                    logger.error(f"Failed to evaluate {name}: {e}")
        except Exception as e:
            logger.error(f"Failed to train XGBoost: {e}")
    else:
        logger.warning("Skipping XGBoost (not available)")
    
    # Find best candidate (excluding baseline)
    candidates = [r for r in all_results if r['model_name'] != 'Baseline_LogisticRegression']
    
    if not candidates:
        logger.warning("No candidates evaluated successfully")
        best_candidate = None
    else:
        # Sort by cost, then by FN (lower is better)
        candidates_sorted = sorted(candidates, key=lambda x: (x['cost'], x['fn']))
        best_candidate = candidates_sorted[0]
    
    # Determine recommendation
    if best_candidate is None:
        recommendation = "keep_baseline"
        reason = "No candidates evaluated successfully"
    elif best_candidate['fn'] > BASELINE_FN + 2:
        recommendation = "keep_baseline"
        reason = f"Best candidate increases FN by {best_candidate['fn'] - BASELINE_FN} cases (unacceptable)"
    elif best_candidate['cost'] >= BASELINE_COST:
        recommendation = "keep_baseline"
        reason = f"Best candidate cost ({best_candidate['cost']:.1f}) >= baseline cost ({BASELINE_COST:.1f})"
    else:
        recommendation = "adopt_v2"
        reason = f"Best candidate reduces cost from {BASELINE_COST:.1f} to {best_candidate['cost']:.1f} " \
                 f"(FN: {BASELINE_FN} → {best_candidate['fn']}, FP: {BASELINE_FP} → {best_candidate['fp']})"
    
    # Prepare output
    output_data = {
        'experiment_timestamp': datetime.now().isoformat(),
        'baseline': baseline_result,
        'candidates': candidates,
        'best_candidate': best_candidate,
        'recommendation': recommendation,
        'recommendation_reason': reason,
        'baseline_metrics': {
            'fn': BASELINE_FN,
            'fp': BASELINE_FP,
            'cost': BASELINE_COST
        }
    }
    
    # Save JSON
    json_path = RESULTS_DIR / 'meta_learner_v2_comparison.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\n✓ Results saved to: {json_path}")
    
    # Generate markdown report
    generate_report(output_data, RESULTS_DIR / 'REPORT.md')
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Recommendation: {recommendation.upper()}")
    logger.info(f"Reason: {reason}")


def generate_report(comparison_data: Dict, output_path: Path):
    """Generate markdown report."""
    logger.info("Generating markdown report...")
    
    baseline = comparison_data['baseline']
    candidates = comparison_data['candidates']
    best = comparison_data.get('best_candidate')
    recommendation = comparison_data['recommendation']
    reason = comparison_data['recommendation_reason']
    
    report_lines = [
        "# Meta-Learner V2 Experiment Report",
        "",
        f"**Experiment Date**: {comparison_data['experiment_timestamp']}",
        "",
        "## Executive Summary",
        "",
        f"**Recommendation**: **{recommendation.upper().replace('_', ' ')}**",
        "",
        f"**Reason**: {reason}",
        "",
        "---",
        "",
        "## Baseline Performance",
        "",
        f"- **Model**: {baseline['model_name']}",
        f"- **Threshold**: {baseline['selected_threshold']:.4f}",
        f"- **FN**: {baseline['fn']}",
        f"- **FP**: {baseline['fp']}",
        f"- **Cost** (2×FN + FP): {baseline['cost']:.1f}",
        f"- **Recall**: {baseline['recall']:.4f}",
        f"- **Precision**: {baseline['precision']:.4f}",
        f"- **F1**: {baseline['f1']:.4f}",
        "",
        "---",
        "",
        "## Candidate Models Comparison",
        "",
        "| Model | Threshold | FN | FP | Cost | Recall | Precision | F1 |",
        "|-------|-----------|----|----|------|--------|-----------|----|"
    ]
    
    # Add baseline row
    report_lines.append(
        f"| **{baseline['model_name']}** (baseline) | {baseline['selected_threshold']:.4f} | "
        f"{baseline['fn']} | {baseline['fp']} | **{baseline['cost']:.1f}** | "
        f"{baseline['recall']:.4f} | {baseline['precision']:.4f} | {baseline['f1']:.4f} |"
    )
    
    # Add candidate rows
    for cand in sorted(candidates, key=lambda x: x['cost']):
        marker = " ⭐ **BEST**" if cand == best else ""
        report_lines.append(
            f"| {cand['model_name']}{marker} | {cand['selected_threshold']:.4f} | "
            f"{cand['fn']} | {cand['fp']} | {cand['cost']:.1f} | "
            f"{cand['recall']:.4f} | {cand['precision']:.4f} | {cand['f1']:.4f} |"
        )
    
    report_lines.extend([
        "",
        "---",
        "",
        "## Best Candidate",
        ""
    ])
    
    if best:
        report_lines.extend([
            f"- **Model**: {best['model_name']}",
            f"- **Threshold**: {best['selected_threshold']:.4f}",
            f"- **FN**: {best['fn']} ({best['fn'] - baseline['fn']:+d} vs baseline)",
            f"- **FP**: {best['fp']} ({best['fp'] - baseline['fp']:+d} vs baseline)",
            f"- **Cost**: {best['cost']:.1f} ({best['cost'] - baseline['cost']:+.1f} vs baseline)",
            f"- **Recall**: {best['recall']:.4f} ({best['recall'] - baseline['recall']:+.4f} vs baseline)",
            f"- **Precision**: {best['precision']:.4f} ({best['precision'] - baseline['precision']:+.4f} vs baseline)",
            f"- **F1**: {best['f1']:.4f} ({best['f1'] - baseline['f1']:+.4f} vs baseline)",
            ""
        ])
    else:
        report_lines.append("No candidate models evaluated successfully.")
        report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "## Decision Criteria",
        "",
        "A candidate meta-learner is recommended if:",
        "",
        "1. **Cost reduction**: Lower cost (2×FN + FP) than baseline",
        "2. **FN constraint**: FN ≤ baseline + 2 (medical priority)",
        "3. **FP reduction**: Significant FP reduction if FN is similar",
        "",
        "---",
        "",
        "## Final Decision",
        "",
        f"**{recommendation.upper().replace('_', ' ')}**",
        "",
        f"{reason}",
        ""
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"✓ Report saved to: {output_path}")


if __name__ == '__main__':
    main()

