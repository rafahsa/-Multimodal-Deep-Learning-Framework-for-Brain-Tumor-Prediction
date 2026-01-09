"""
Prepare Out-of-Fold (OOF) Predictions for Ensemble Stacking

This script aggregates validation predictions from multiple base models across
5-fold cross-validation to create OOF prediction tables for ensemble meta-learner training.

Usage:
    python scripts/ensemble/prepare_oof_predictions.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model name to results directory mapping
MODEL_CONFIGS = {
    'ResNet50-3D': {
        'results_dir': Path('results/ResNet50-3D/runs'),
        'output_name': 'resnet50_3d'
    },
    'SwinUNETR-3D': {
        'results_dir': Path('results/SwinUNETR-3D/runs'),
        'output_name': 'swinunetr_3d'
    },
    'DualStreamMIL-3D': {
        'results_dir': Path('results/DualStreamMIL-3D/runs'),
        'output_name': 'dualstream_mil_3d'
    }
}

SPLITS_DIR = Path('splits')
ENSEMBLE_DIR = Path('ensemble/oof_predictions')
NUM_FOLDS = 5


def find_latest_run(model_results_dir: Path, fold: int) -> Optional[Path]:
    """
    Find the most recent run directory for a given fold.
    
    Args:
        model_results_dir: Base results directory for the model (e.g., results/ResNet50-3D/runs)
        fold: Fold number (0-4)
    
    Returns:
        Path to the latest run directory, or None if no runs found
    """
    fold_dir = model_results_dir / f'fold_{fold}'
    
    if not fold_dir.exists():
        logger.warning(f"Fold directory not found: {fold_dir}")
        return None
    
    # List all run directories (matching pattern run_*)
    run_dirs = [d for d in fold_dir.iterdir() 
                if d.is_dir() and d.name.startswith('run_')]
    
    if not run_dirs:
        logger.warning(f"No run directories found in {fold_dir}")
        return None
    
    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_run = run_dirs[0]
    
    logger.info(f"Selected latest run for fold {fold}: {latest_run.name}")
    return latest_run


def verify_run_has_predictions(run_dir: Path) -> bool:
    """
    Verify that a run directory contains required prediction files.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        True if all required files exist, False otherwise
    """
    predictions_dir = run_dir / 'predictions'
    val_probs_file = predictions_dir / 'val_probs.npy'
    
    if not predictions_dir.exists():
        logger.warning(f"Predictions directory not found: {predictions_dir}")
        return False
    
    if not val_probs_file.exists():
        logger.warning(f"val_probs.npy not found: {val_probs_file}")
        return False
    
    return True


def load_oof_predictions_from_run(
    run_dir: Path, 
    fold: int, 
    val_split_csv: Path
) -> pd.DataFrame:
    """
    Load OOF predictions from a single run and match with patient IDs.
    
    Args:
        run_dir: Path to the run directory
        fold: Fold number (for tracking)
        val_split_csv: Path to validation split CSV file
    
    Returns:
        DataFrame with columns: patient_id, fold, hgg_prob, label
    """
    # Load validation predictions
    predictions_dir = run_dir / 'predictions'
    val_probs = np.load(predictions_dir / 'val_probs.npy')
    
    # Extract HGG probabilities (second column, index 1)
    # val_probs shape: (n_samples, 2) where columns are [LGG_prob, HGG_prob]
    if val_probs.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in val_probs, got {val_probs.shape[1]}")
    
    hgg_probs = val_probs[:, 1]
    
    # Load validation split CSV to get patient IDs
    val_split_df = pd.read_csv(val_split_csv)
    
    # Verify number of predictions matches number of patients
    if len(hgg_probs) != len(val_split_df):
        raise ValueError(
            f"Mismatch: {len(hgg_probs)} predictions but {len(val_split_df)} patients "
            f"in {val_split_csv.name}"
        )
    
    # Create DataFrame
    # ASSUMPTION: Predictions are in the same order as patients in the CSV file
    result_df = pd.DataFrame({
        'patient_id': val_split_df['patient_id'].values,
        'fold': fold,
        'hgg_prob': hgg_probs,
        'label': val_split_df['class_label'].values
    })
    
    return result_df


def aggregate_oof_predictions(model_name: str) -> Optional[pd.DataFrame]:
    """
    Aggregate OOF predictions across all folds for a single model.
    
    Args:
        model_name: Name of the model (key in MODEL_CONFIGS)
    
    Returns:
        DataFrame with all OOF predictions for the model, or None if failed
    """
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}")
        return None
    
    config = MODEL_CONFIGS[model_name]
    results_dir = config['results_dir']
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return None
    
    all_fold_dfs = []
    
    for fold in range(NUM_FOLDS):
        # Find latest run for this fold
        latest_run = find_latest_run(results_dir, fold)
        if latest_run is None:
            logger.error(f"Could not find run for {model_name} fold {fold}")
            return None
        
        # Verify required files exist
        if not verify_run_has_predictions(latest_run):
            logger.error(f"Missing prediction files in {latest_run}")
            return None
        
        # Load validation split CSV
        val_split_csv = SPLITS_DIR / f'fold_{fold}_val.csv'
        if not val_split_csv.exists():
            logger.error(f"Validation split file not found: {val_split_csv}")
            return None
        
        # Load OOF predictions for this fold
        try:
            fold_df = load_oof_predictions_from_run(latest_run, fold, val_split_csv)
            all_fold_dfs.append(fold_df)
            logger.info(f"Loaded {len(fold_df)} predictions for {model_name} fold {fold}")
        except Exception as e:
            logger.error(f"Error loading predictions for {model_name} fold {fold}: {e}")
            return None
    
    # Concatenate all folds
    combined_df = pd.concat(all_fold_dfs, ignore_index=True)
    
    # Sort by patient_id for consistency
    combined_df = combined_df.sort_values('patient_id').reset_index(drop=True)
    
    return combined_df


def verify_oof_predictions(df: pd.DataFrame, model_name: str) -> bool:
    """
    Perform verification checks on OOF predictions.
    
    Args:
        df: DataFrame with OOF predictions
        model_name: Name of the model (for logging)
    
    Returns:
        True if all checks pass, False otherwise
    """
    logger.info(f"Verifying OOF predictions for {model_name}...")
    
    # Check 1: Uniqueness - each patient should appear exactly once
    duplicate_patients = df[df.duplicated(subset=['patient_id'], keep=False)]
    if len(duplicate_patients) > 0:
        logger.error(f"Found {len(duplicate_patients)} duplicate patient IDs!")
        logger.error(f"Duplicate patients: {duplicate_patients['patient_id'].unique()}")
        return False
    logger.info("✓ Uniqueness check passed")
    
    # Check 2: Completeness - should have predictions for all folds
    fold_counts = df['fold'].value_counts().sort_index()
    expected_folds = set(range(NUM_FOLDS))
    actual_folds = set(fold_counts.index)
    if expected_folds != actual_folds:
        logger.error(f"Missing folds. Expected {expected_folds}, got {actual_folds}")
        return False
    logger.info(f"✓ Completeness check passed (folds: {sorted(actual_folds)})")
    
    # Check 3: Probability range
    invalid_probs = df[(df['hgg_prob'] < 0) | (df['hgg_prob'] > 1)]
    if len(invalid_probs) > 0:
        logger.error(f"Found {len(invalid_probs)} probabilities outside [0, 1]")
        return False
    logger.info("✓ Probability range check passed")
    
    # Check 4: Label values
    invalid_labels = df[~df['label'].isin([0, 1])]
    if len(invalid_labels) > 0:
        logger.error(f"Found {len(invalid_labels)} invalid labels (not 0 or 1)")
        return False
    logger.info("✓ Label values check passed")
    
    # Check 5: Total number of predictions (should match total validation set size)
    # Load all validation splits to get expected total
    total_val_patients = 0
    for fold in range(NUM_FOLDS):
        val_split_csv = SPLITS_DIR / f'fold_{fold}_val.csv'
        if val_split_csv.exists():
            val_df = pd.read_csv(val_split_csv)
            total_val_patients += len(val_df)
    
    if len(df) != total_val_patients:
        logger.warning(
            f"Total predictions ({len(df)}) does not match total validation patients "
            f"({total_val_patients}). This may be expected if some folds have different sizes."
        )
    else:
        logger.info(f"✓ Total predictions check passed ({len(df)} patients)")
    
    logger.info(f"All verification checks passed for {model_name}")
    return True


def main():
    """Main function to prepare OOF predictions for all models."""
    logger.info("=" * 80)
    logger.info("Preparing Out-of-Fold Predictions for Ensemble Stacking")
    logger.info("=" * 80)
    
    # Ensure output directory exists
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each model
    results_summary = {}
    
    for model_name in MODEL_CONFIGS.keys():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'=' * 80}")
        
        # Aggregate OOF predictions
        oof_df = aggregate_oof_predictions(model_name)
        
        if oof_df is None:
            logger.error(f"Failed to aggregate OOF predictions for {model_name}")
            continue
        
        # Verify predictions
        if not verify_oof_predictions(oof_df, model_name):
            logger.error(f"Verification failed for {model_name}. Skipping.")
            continue
        
        # Save to CSV
        output_name = MODEL_CONFIGS[model_name]['output_name']
        output_file = ENSEMBLE_DIR / f'{output_name}_oof.csv'
        oof_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved OOF predictions to: {output_file}")
        logger.info(f"Shape: {oof_df.shape}")
        logger.info(f"Columns: {list(oof_df.columns)}")
        
        # Store summary
        results_summary[model_name] = {
            'output_file': str(output_file),
            'num_predictions': len(oof_df),
            'folds_covered': sorted(oof_df['fold'].unique().tolist())
        }
    
    # Print final summary
    logger.info(f"\n{'=' * 80}")
    logger.info("Summary")
    logger.info(f"{'=' * 80}")
    for model_name, summary in results_summary.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Output: {summary['output_file']}")
        logger.info(f"  Predictions: {summary['num_predictions']}")
        logger.info(f"  Folds: {summary['folds_covered']}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("OOF Prediction Preparation Complete")
    logger.info(f"{'=' * 80}")


if __name__ == '__main__':
    main()

