"""
Train Logistic Regression Meta-Learner on Merged OOF Predictions

This script trains a Logistic Regression meta-learner to combine predictions
from three base models (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D) using
their out-of-fold (OOF) predictions.

Usage:
    python scripts/ensemble/train_meta_learner.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple
import joblib
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
ENSEMBLE_DIR = Path('ensemble')
MODELS_DIR = ENSEMBLE_DIR / 'models'
RESULTS_DIR = ENSEMBLE_DIR / 'results'
METRICS_FILE = RESULTS_DIR / 'meta_learner_metrics.json'
MODEL_FILE = MODELS_DIR / 'meta_learner_logistic_regression.joblib'

# Feature columns (base model predictions)
FEATURE_COLUMNS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
TARGET_COLUMN = 'label'


def load_data() -> pd.DataFrame:
    """Load merged OOF predictions."""
    if not MERGED_OOF_FILE.exists():
        raise FileNotFoundError(f"Merged OOF file not found: {MERGEOOF_FILE}")
    
    logger.info(f"Loading merged OOF predictions from: {MERGED_OOF_FILE}")
    df = pd.read_csv(MERGED_OOF_FILE)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features (X) and target (y) for training.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
    """
    logger.info("Preparing data for training...")
    
    # Extract features
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    X = df[FEATURE_COLUMNS].values
    logger.info(f"Features (X): shape {X.shape}, columns {FEATURE_COLUMNS}")
    
    # Extract target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")
    
    y = df[TARGET_COLUMN].values
    logger.info(f"Target (y): shape {y.shape}, unique values {np.unique(y)}")
    
    # Verify no missing values
    if np.isnan(X).any():
        raise ValueError("Found NaN values in features")
    if np.isnan(y).any():
        raise ValueError("Found NaN values in target")
    
    # Verify class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    return X, y


def train_meta_learner(X: np.ndarray, y: np.ndarray, use_class_weights: bool = True) -> LogisticRegression:
    """
    Train Logistic Regression meta-learner.
    
    Args:
        X: Feature matrix
        y: Target vector
        use_class_weights: Whether to use class weights to handle imbalance
    
    Returns:
        Trained LogisticRegression model
    """
    logger.info("\n" + "="*80)
    logger.info("Training Logistic Regression Meta-Learner")
    logger.info("="*80)
    
    # Compute class weights if requested
    class_weights = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        logger.info(f"Using balanced class weights: {class_weights}")
    else:
        logger.info("Using uniform class weights (no class balancing)")
    
    # Create and train model
    # Using L2 regularization with moderate strength (C=1.0 is default)
    # With 285 samples and 3 features, risk of overfitting is low
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,  # Sufficient for convergence
        solver='lbfgs',  # Good for small datasets
        class_weight=class_weights if use_class_weights else None,
        C=1.0,  # Default regularization strength
        penalty='l2'  # L2 regularization (default)
    )
    
    logger.info(f"Model configuration:")
    logger.info(f"  Solver: {model.solver}")
    logger.info(f"  Regularization (C): {model.C}")
    logger.info(f"  Penalty: {model.penalty}")
    logger.info(f"  Class weights: {class_weights if use_class_weights else 'None'}")
    logger.info(f"  Max iterations: {model.max_iter}")
    
    logger.info("\nTraining model...")
    model.fit(X, y)
    logger.info("✓ Training completed")
    
    # Log learned coefficients
    logger.info("\nLearned coefficients:")
    for i, feature_name in enumerate(FEATURE_COLUMNS):
        logger.info(f"  {feature_name}: {model.coef_[0][i]:.6f}")
    logger.info(f"  Intercept: {model.intercept_[0]:.6f}")
    
    # Interpret coefficients (higher absolute value = more important)
    abs_coefs = np.abs(model.coef_[0])
    importance_order = np.argsort(abs_coefs)[::-1]
    logger.info("\nFeature importance (by absolute coefficient):")
    for rank, idx in enumerate(importance_order, 1):
        logger.info(f"  {rank}. {FEATURE_COLUMNS[idx]}: {abs_coefs[idx]:.6f} (coef: {model.coef_[0][idx]:.6f})")
    
    return model


def evaluate_model(model: LogisticRegression, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Evaluate model performance at a given threshold.
    
    Args:
        model: Trained LogisticRegression model
        X: Feature matrix
        y: True labels
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("\n" + "="*80)
    logger.info(f"Evaluating Meta-Learner (Threshold = {threshold:.2f})")
    logger.info("="*80)
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of positive class (HGG)
    
    # Apply threshold to get predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Extract confusion matrix components
    tn, fp, fn, tp = cm.ravel()
    
    # Classification report
    class_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    
    # Log results
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.6f} ({accuracy*100:.2f}%)")
    logger.info(f"  Precision: {precision:.6f}")
    logger.info(f"  Recall:    {recall:.6f}")
    logger.info(f"  F1-Score:  {f1:.6f}")
    logger.info(f"  AUC-ROC:   {auc:.6f}")
    
    logger.info(f"\nConfusion Matrix (Threshold = {threshold:.2f}):")
    logger.info(f"                Predicted")
    logger.info(f"                LGG  HGG")
    logger.info(f"  Actual LGG    {tn:4d}  {fp:4d}")
    logger.info(f"        HGG     {fn:4d}  {tp:4d}")
    logger.info(f"\n  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    logger.info(f"\nPer-Class Metrics:")
    logger.info(f"  LGG (class 0):")
    logger.info(f"    Precision: {class_report['0']['precision']:.6f}")
    logger.info(f"    Recall:    {class_report['0']['recall']:.6f}")
    logger.info(f"    F1-Score:  {class_report['0']['f1-score']:.6f}")
    logger.info(f"    Support:   {class_report['0']['support']:.0f}")
    logger.info(f"  HGG (class 1):")
    logger.info(f"    Precision: {class_report['1']['precision']:.6f}")
    logger.info(f"    Recall:    {class_report['1']['recall']:.6f}")
    logger.info(f"    F1-Score:  {class_report['1']['f1-score']:.6f}")
    logger.info(f"    Support:   {class_report['1']['support']:.0f}")
    
    # Extract confusion matrix components
    tn, fp, fn, tp = cm.ravel()
    
    # Compile metrics dictionary
    metrics = {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'classification_report': class_report,
        'model_coefficients': {
            FEATURE_COLUMNS[i]: float(model.coef_[0][i]) for i in range(len(FEATURE_COLUMNS))
        },
        'model_intercept': float(model.intercept_[0]),
        'feature_importance': {
            FEATURE_COLUMNS[i]: float(np.abs(model.coef_[0][i])) for i in range(len(FEATURE_COLUMNS))
        }
    }
    
    return metrics


def save_model(model: LogisticRegression, metrics: Dict, save_model_file: bool = True):
    """Save trained model and metrics."""
    logger.info("\n" + "="*80)
    logger.info("Saving Model and Metrics")
    logger.info("="*80)
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model (only if requested, e.g., during training)
    if save_model_file:
        logger.info(f"Saving model to: {MODEL_FILE}")
        joblib.dump(model, MODEL_FILE)
        logger.info("✓ Model saved")
    
    # Prepare metrics for JSON serialization
    metrics_serializable = metrics.copy()
    # Convert numpy types to native Python types (already done, but ensure)
    for key, value in metrics_serializable.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value) if isinstance(value, np.floating) else int(value)
    
    # Add metadata
    threshold = metrics.get('threshold', 0.5)
    metrics_serializable['metadata'] = {
        'model_type': 'LogisticRegression',
        'evaluation_date': datetime.now().isoformat(),
        'threshold': float(threshold),
        'dataset': 'OOF predictions (validation data from cross-validation)',
        'features': FEATURE_COLUMNS,
        'target': TARGET_COLUMN,
        'n_samples': int(len(pd.read_csv(MERGED_OOF_FILE))),
        'class_distribution': {
            'LGG': int(metrics['classification_report']['0']['support']),
            'HGG': int(metrics['classification_report']['1']['support'])
        }
    }
    
    # Save metrics with threshold in filename
    threshold_str = f"{threshold:.2f}".replace('.', '_')
    metrics_file = RESULTS_DIR / f'eval_threshold_{threshold_str}.json'
    logger.info(f"Saving metrics to: {metrics_file}")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    logger.info("✓ Metrics saved")
    
    return metrics_file


def main():
    """Main training/evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train or evaluate Logistic Regression Meta-Learner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.22,
        help='Classification threshold for evaluation (default: 0.22)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and only evaluate with existing model (if available)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Logistic Regression Meta-Learner Training/Evaluation")
    logger.info("=" * 80)
    logger.info(f"Threshold: {args.threshold:.2f}")
    
    try:
        # Step 1: Load data
        df = load_data()
        
        # Step 2: Prepare data
        X, y = prepare_data(df)
        
        # Step 3: Train or load model
        if args.skip_training and MODEL_FILE.exists():
            logger.info(f"\nLoading existing model from: {MODEL_FILE}")
            model = joblib.load(MODEL_FILE)
            logger.info("✓ Model loaded")
        else:
            # Step 3: Train model
            # Using class weights to handle class imbalance (75 LGG vs 210 HGG)
            model = train_meta_learner(X, y, use_class_weights=True)
        
        # Step 4: Evaluate model at specified threshold
        metrics = evaluate_model(model, X, y, threshold=args.threshold)
        
        # Step 5: Save model and metrics
        save_model_file = not args.skip_training
        metrics_file = save_model(model, metrics, save_model_file=save_model_file)
        
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Complete")
        logger.info("=" * 80)
        if save_model_file:
            logger.info(f"Model saved to: {MODEL_FILE}")
        logger.info(f"Metrics saved to: {metrics_file}")
        logger.info(f"\nPerformance at Threshold {args.threshold:.2f}:")
        logger.info(f"  TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TP: {metrics['tp']}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
        
    except Exception as e:
        logger.error(f"Processing failed with error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

