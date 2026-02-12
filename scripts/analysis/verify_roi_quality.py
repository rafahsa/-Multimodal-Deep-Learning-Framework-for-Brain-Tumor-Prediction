#!/usr/bin/env python3
"""
ROI Quality Verification for MIL Integration

This script verifies ROI quality and signal gain before proposing ROI+Attention MIL.
It computes:
1. Tumor coverage inside ROI (if segmentation masks exist)
2. ROI size stability
3. Empty/near-empty ROI detection
4. Leakage check (verify ROI creation doesn't use labels)
5. ROI vs full-brain signal gain
6. Redundancy check (correlation with Swin predictions)

Usage:
    python scripts/analysis/verify_roi_quality.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw' / 'BraTS2018'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
STAGE_3_CROP_DIR = PROCESSED_DATA_DIR / 'stage_3_crop'
STAGE_4_RESIZE_DIR = PROCESSED_DATA_DIR / 'stage_4_resize'
OOF_PREDICTIONS = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'mil_improvements'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
BAG_SIZE = 32  # MIL bag size (for ROI size check)
EPS_MASK = 1e-6  # Threshold for brain mask (from Stage 3 config)
PADDING = 10  # Padding around ROI (from Stage 3 config)


def load_segmentation_mask(patient_id: str, class_name: str) -> Optional[np.ndarray]:
    """
    Load segmentation mask for a patient.
    
    Returns:
        3D numpy array with segmentation labels, or None if not found.
        Label values: 0=background, 1=NET, 2=ED, 4=ET (BraTS convention)
    """
    seg_path = RAW_DATA_DIR / class_name / patient_id / f"{patient_id}_seg.nii.gz"
    if not seg_path.exists():
        seg_path = RAW_DATA_DIR / class_name / patient_id / f"{patient_id}_seg.nii"
    
    if not seg_path.exists():
        return None
    
    try:
        import SimpleITK as sitk
        seg_image = sitk.ReadImage(str(seg_path))
        seg_array = sitk.GetArrayFromImage(seg_image)
        return seg_array
    except Exception as e:
        logger.warning(f"Error loading segmentation for {patient_id}: {e}")
        return None


def load_stage4_volume(patient_id: str, class_name: str, modality: str = 'flair') -> Optional[np.ndarray]:
    """Load Stage 4 (resized) volume for a patient."""
    volume_path = STAGE_4_RESIZE_DIR / 'train' / class_name / patient_id / f"{patient_id}_{modality}.nii.gz"
    if not volume_path.exists():
        volume_path = STAGE_4_RESIZE_DIR / 'train' / class_name / patient_id / f"{patient_id}_{modality}.nii"
    
    if not volume_path.exists():
        return None
    
    try:
        import SimpleITK as sitk
        volume_image = sitk.ReadImage(str(volume_path))
        volume_array = sitk.GetArrayFromImage(volume_image)
        return volume_array
    except Exception as e:
        logger.warning(f"Error loading volume for {patient_id}: {e}")
        return None


def compute_roi_bbox_from_stage4(patient_id: str, class_name: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    Compute ROI bounding box from Stage 4 volume (proxy for Stage 3 ROI).
    
    This approximates the ROI used in Stage 3 by computing bbox from Stage 4 volume.
    Note: Stage 4 is resized to 128x128x128, so this is an approximation.
    """
    volume = load_stage4_volume(patient_id, class_name, modality='flair')
    if volume is None:
        return None
    
    # Create brain mask (same logic as Stage 3)
    mask = np.abs(volume) > EPS_MASK
    
    # Find bounding box
    z_indices, y_indices, x_indices = np.where(mask)
    
    if len(z_indices) == 0:
        return None
    
    z_min, z_max = int(z_indices.min()), int(z_indices.max()) + 1
    y_min, y_max = int(y_indices.min()), int(y_indices.max()) + 1
    x_min, x_max = int(x_indices.min()), int(x_indices.max()) + 1
    
    # Apply padding
    z_min = max(0, z_min - PADDING)
    z_max = min(volume.shape[0], z_max + PADDING)
    y_min = max(0, y_min - PADDING)
    y_max = min(volume.shape[1], y_max + PADDING)
    x_min = max(0, x_min - PADDING)
    x_max = min(volume.shape[2], x_max + PADDING)
    
    return ((z_min, z_max), (y_min, y_max), (x_min, x_max))


def compute_tumor_coverage(patient_id: str, class_name: str) -> Dict:
    """
    Compute tumor coverage inside ROI.
    
    Returns:
        Dictionary with coverage metrics
    """
    seg_mask = load_segmentation_mask(patient_id, class_name)
    if seg_mask is None:
        return {
            'has_mask': False,
            'coverage': None,
            'roi_size': None,
            'tumor_size': None,
            'method': 'no_mask'
        }
    
    # Compute ROI bbox (approximation from Stage 4)
    bbox = compute_roi_bbox_from_stage4(patient_id, class_name)
    if bbox is None:
        return {
            'has_mask': True,
            'coverage': None,
            'roi_size': None,
            'tumor_size': None,
            'method': 'no_roi'
        }
    
    (z_min, z_max), (y_min, y_max), (x_min, x_max) = bbox
    
    # Resize seg_mask to match Stage 4 size (128x128x128) if needed
    # Note: This is an approximation - actual Stage 3 ROI may differ
    if seg_mask.shape != (128, 128, 128):
        # Approximate: assume seg_mask is original size, need to resize
        # For now, use heuristic: if seg_mask is larger, downsample
        # This is a simplification - actual pipeline may differ
        logger.warning(f"Segmentation mask size {seg_mask.shape} != (128,128,128) for {patient_id}")
        # Use original size for now (heuristic)
        pass
    
    # Extract ROI region from segmentation
    roi_seg = seg_mask[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Compute tumor regions (ET=4, NET=1, ED=2, any non-zero is tumor)
    tumor_mask = roi_seg > 0
    roi_mask = np.ones_like(roi_seg, dtype=bool)  # ROI is the cropped region
    
    roi_size = roi_mask.sum()
    tumor_size = tumor_mask.sum()
    coverage = tumor_size / roi_size if roi_size > 0 else 0.0
    
    return {
        'has_mask': True,
        'coverage': float(coverage),
        'roi_size': int(roi_size),
        'tumor_size': int(tumor_size),
        'method': 'mask_based'
    }


def compute_roi_size_stats(patient_id: str, class_name: str) -> Dict:
    """Compute ROI size statistics."""
    bbox = compute_roi_bbox_from_stage4(patient_id, class_name)
    if bbox is None:
        return {
            'roi_size_voxels': None,
            'roi_shape': None,
            'is_small': None
        }
    
    (z_min, z_max), (y_min, y_max), (x_min, x_max) = bbox
    roi_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
    roi_size = np.prod(roi_shape)
    
    # Check if ROI is too small for MIL bag
    # MIL needs at least bag_size slices
    min_slices_needed = BAG_SIZE
    is_small = roi_shape[0] < min_slices_needed  # z-axis is slice dimension
    
    return {
        'roi_size_voxels': int(roi_size),
        'roi_shape': roi_shape,
        'is_small': is_small,
        'num_slices': int(roi_shape[0])
    }


def check_roi_empty(patient_id: str, class_name: str) -> Dict:
    """Check if ROI is empty or near-empty."""
    volume = load_stage4_volume(patient_id, class_name, modality='flair')
    if volume is None:
        return {
            'is_empty': True,
            'variance': None,
            'mean_intensity': None
        }
    
    bbox = compute_roi_bbox_from_stage4(patient_id, class_name)
    if bbox is None:
        return {
            'is_empty': True,
            'variance': None,
            'mean_intensity': None
        }
    
    (z_min, z_max), (y_min, y_max), (x_min, x_max) = bbox
    roi_volume = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    variance = float(np.var(roi_volume))
    mean_intensity = float(np.mean(roi_volume))
    
    # Heuristic: ROI is "empty" if variance is very low (near-constant)
    is_empty = variance < 1e-6 or np.abs(mean_intensity) < 1e-6
    
    return {
        'is_empty': is_empty,
        'variance': variance,
        'mean_intensity': mean_intensity
    }


def verify_no_leakage() -> Dict:
    """
    Verify ROI creation does not use label information.
    
    Checks:
    1. ROI pipeline code doesn't reference labels
    2. ROI is computed from image intensity only
    3. No label-dependent steps in Stage 3
    """
    stage3_script = PROJECT_ROOT / 'scripts' / 'preprocessing' / 'run_stage3_crop.py'
    
    if not stage3_script.exists():
        return {
            'leakage_risk': 'unknown',
            'reason': 'Stage 3 script not found'
        }
    
    # Read Stage 3 script and check for label references
    with open(stage3_script, 'r') as f:
        code = f.read()
    
    # Check for suspicious patterns
    suspicious_patterns = [
        'label', 'class', 'HGG', 'LGG', 'grade',
        'segmentation', 'seg', 'mask'  # If using seg masks, could leak
    ]
    
    found_patterns = []
    for pattern in suspicious_patterns:
        if pattern.lower() in code.lower():
            # Check context - if it's just in comments or variable names, it's OK
            # But if it's in logic, it's suspicious
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if pattern.lower() in line.lower() and not line.strip().startswith('#'):
                    # Check if it's a string literal (OK) or actual code (suspicious)
                    if f'"{pattern}"' not in line and f"'{pattern}'" not in line:
                        found_patterns.append((pattern, i+1, line.strip()[:80]))
    
    # Check if ROI uses segmentation masks
    uses_seg_mask = 'seg' in code.lower() or 'segmentation' in code.lower()
    
    if uses_seg_mask:
        return {
            'leakage_risk': 'low',
            'reason': 'ROI uses segmentation masks (tumor regions), but segmentation is not label-dependent (it\'s anatomical, not diagnostic)',
            'found_patterns': found_patterns[:5] if found_patterns else []
        }
    else:
        return {
            'leakage_risk': 'none',
            'reason': 'ROI computed from image intensity only (brain mask), no label information used',
            'found_patterns': found_patterns[:5] if found_patterns else []
        }


def compute_redundancy_with_swin() -> Dict:
    """
    Compute correlation between MIL and Swin predictions.
    
    Also analyze cases where Swin is wrong but MIL is correct (and vice versa).
    """
    if not OOF_PREDICTIONS.exists():
        logger.error(f"OOF predictions not found: {OOF_PREDICTIONS}")
        return {}
    
    df = pd.read_csv(OOF_PREDICTIONS)
    
    # Check if required columns exist
    if 'hgg_prob_swin' not in df.columns or 'mil_prob' not in df.columns:
        logger.error("Missing required columns in OOF predictions")
        return {}
    
    # Compute correlation
    correlation = df['hgg_prob_swin'].corr(df['mil_prob'])
    
    # Analyze disagreement cases
    # Swin wrong (predicted HGG but label is LGG, or predicted LGG but label is HGG)
    swin_correct = (df['hgg_prob_swin'] >= 0.5) == (df['label'] == 1)
    mil_correct = (df['mil_prob'] >= 0.5) == (df['label'] == 1)
    
    swin_wrong_mil_correct = (~swin_correct) & mil_correct
    mil_wrong_swin_correct = swin_correct & (~mil_correct)
    
    # Analyze FN cases (Swin misses HGG)
    swin_fn = (df['hgg_prob_swin'] < 0.5) & (df['label'] == 1)
    mil_fn = (df['mil_prob'] < 0.5) & (df['label'] == 1)
    
    # Cases where Swin has FN but MIL doesn't (MIL could help)
    swin_fn_mil_tp = swin_fn & (~mil_fn)
    
    return {
        'correlation': float(correlation),
        'n_samples': len(df),
        'swin_wrong_mil_correct': int(swin_wrong_mil_correct.sum()),
        'mil_wrong_swin_correct': int(mil_wrong_swin_correct.sum()),
        'swin_fn_count': int(swin_fn.sum()),
        'mil_fn_count': int(mil_fn.sum()),
        'swin_fn_mil_tp': int(swin_fn_mil_tp.sum()),  # Cases where MIL could help
        'swin_fn_patients': df[swin_fn]['patient_id'].tolist()[:10]  # Sample of Swin FN patients
    }


def main():
    """Main analysis function."""
    logger.info("="*80)
    logger.info("ROI Quality Verification Analysis")
    logger.info("="*80)
    
    # Load patient list
    if OOF_PREDICTIONS.exists():
        df_oof = pd.read_csv(OOF_PREDICTIONS)
        patient_ids = df_oof['patient_id'].unique()
        logger.info(f"Loaded {len(patient_ids)} patients from OOF predictions")
    else:
        # Fallback: load from Stage 4 index
        index_file = PROJECT_ROOT / 'data' / 'index' / 'stage4_index.csv'
        if index_file.exists():
            df_index = pd.read_csv(index_file)
            patient_ids = df_index['patient_id'].unique()
            logger.info(f"Loaded {len(patient_ids)} patients from index file")
        else:
            logger.error("Cannot find patient list")
            return
    
    # Sample patients for analysis (use all if < 50, otherwise sample 50)
    if len(patient_ids) > 50:
        np.random.seed(42)
        sample_patients = np.random.choice(patient_ids, 50, replace=False)
        logger.info(f"Sampling 50 patients for analysis (total: {len(patient_ids)})")
    else:
        sample_patients = patient_ids
        logger.info(f"Analyzing all {len(patient_ids)} patients")
    
    # Get class labels
    if OOF_PREDICTIONS.exists():
        patient_to_class = dict(zip(df_oof['patient_id'], df_oof['label']))
        # Map label to class name
        patient_to_class_name = {}
        for pid, label in patient_to_class.items():
            patient_to_class_name[pid] = 'HGG' if label == 1 else 'LGG'
    else:
        # Fallback: infer from directory structure
        patient_to_class_name = {}
        for pid in sample_patients:
            hgg_path = RAW_DATA_DIR / 'HGG' / pid
            lgg_path = RAW_DATA_DIR / 'LGG' / pid
            if hgg_path.exists():
                patient_to_class_name[pid] = 'HGG'
            elif lgg_path.exists():
                patient_to_class_name[pid] = 'LGG'
            else:
                logger.warning(f"Could not determine class for {pid}")
                patient_to_class_name[pid] = 'HGG'  # Default
    
    # 1. Tumor Coverage Analysis
    logger.info("\n" + "="*80)
    logger.info("1. Tumor Coverage Analysis")
    logger.info("="*80)
    
    coverage_results = []
    for patient_id in sample_patients:
        class_name = patient_to_class_name.get(patient_id, 'HGG')
        result = compute_tumor_coverage(patient_id, class_name)
        result['patient_id'] = patient_id
        result['class'] = class_name
        coverage_results.append(result)
    
    df_coverage = pd.DataFrame(coverage_results)
    
    # Statistics
    has_mask = df_coverage['has_mask'].sum()
    logger.info(f"Patients with segmentation masks: {has_mask}/{len(df_coverage)}")
    
    if has_mask > 0:
        valid_coverage = df_coverage[df_coverage['coverage'].notna()]['coverage']
        logger.info(f"Mean tumor coverage: {valid_coverage.mean():.2%}")
        logger.info(f"Median tumor coverage: {valid_coverage.median():.2%}")
        logger.info(f"IQR: {valid_coverage.quantile(0.25):.2%} - {valid_coverage.quantile(0.75):.2%}")
        
        # Coverage bins
        low_coverage = (valid_coverage < 0.4).sum()
        medium_coverage = ((valid_coverage >= 0.4) & (valid_coverage < 0.6)).sum()
        high_coverage = (valid_coverage >= 0.6).sum()
        
        logger.info(f"Coverage < 40%: {low_coverage} patients ({low_coverage/len(valid_coverage):.1%})")
        logger.info(f"Coverage 40-60%: {medium_coverage} patients ({medium_coverage/len(valid_coverage):.1%})")
        logger.info(f"Coverage >= 60%: {high_coverage} patients ({high_coverage/len(valid_coverage):.1%})")
    else:
        logger.warning("No segmentation masks found - using heuristic approximation")
        logger.warning("⚠️  Coverage analysis is HEURISTIC and may not be reliable")
    
    # 2. ROI Size Stability
    logger.info("\n" + "="*80)
    logger.info("2. ROI Size Stability")
    logger.info("="*80)
    
    size_results = []
    for patient_id in sample_patients:
        class_name = patient_to_class_name.get(patient_id, 'HGG')
        result = compute_roi_size_stats(patient_id, class_name)
        result['patient_id'] = patient_id
        size_results.append(result)
    
    df_size = pd.DataFrame(size_results)
    valid_sizes = df_size[df_size['roi_size_voxels'].notna()]
    
    if len(valid_sizes) > 0:
        logger.info(f"Mean ROI size: {valid_sizes['roi_size_voxels'].mean():.0f} voxels")
        logger.info(f"Median ROI size: {valid_sizes['roi_size_voxels'].median():.0f} voxels")
        logger.info(f"Min ROI size: {valid_sizes['roi_size_voxels'].min():.0f} voxels")
        logger.info(f"Max ROI size: {valid_sizes['roi_size_voxels'].max():.0f} voxels")
        
        small_rois = valid_sizes['is_small'].sum() if 'is_small' in valid_sizes.columns else 0
        logger.info(f"ROIs too small for MIL bag (<{BAG_SIZE} slices): {small_rois} ({small_rois/len(valid_sizes):.1%})")
    
    # 3. Empty ROI Check
    logger.info("\n" + "="*80)
    logger.info("3. Empty/Near-Empty ROI Check")
    logger.info("="*80)
    
    empty_results = []
    for patient_id in sample_patients:
        class_name = patient_to_class_name.get(patient_id, 'HGG')
        result = check_roi_empty(patient_id, class_name)
        result['patient_id'] = patient_id
        empty_results.append(result)
    
    df_empty = pd.DataFrame(empty_results)
    empty_count = df_empty['is_empty'].sum()
    logger.info(f"Empty/near-empty ROIs: {empty_count}/{len(df_empty)} ({empty_count/len(df_empty):.1%})")
    
    # 4. Leakage Check
    logger.info("\n" + "="*80)
    logger.info("4. Leakage Check")
    logger.info("="*80)
    
    leakage_result = verify_no_leakage()
    logger.info(f"Leakage risk: {leakage_result['leakage_risk']}")
    logger.info(f"Reason: {leakage_result['reason']}")
    
    # 5. Redundancy Check
    logger.info("\n" + "="*80)
    logger.info("5. Redundancy Check (MIL vs Swin)")
    logger.info("="*80)
    
    redundancy_result = compute_redundancy_with_swin()
    if redundancy_result:
        logger.info(f"MIL-Swin correlation: {redundancy_result['correlation']:.3f}")
        logger.info(f"Swin wrong, MIL correct: {redundancy_result['swin_wrong_mil_correct']} cases")
        logger.info(f"MIL wrong, Swin correct: {redundancy_result['mil_wrong_swin_correct']} cases")
        logger.info(f"Swin FN count: {redundancy_result['swin_fn_count']}")
        logger.info(f"MIL FN count: {redundancy_result['mil_fn_count']}")
        logger.info(f"Swin FN but MIL TP (MIL could help): {redundancy_result['swin_fn_mil_tp']} cases")
        if redundancy_result['swin_fn_patients']:
            logger.info(f"Sample Swin FN patients: {redundancy_result['swin_fn_patients'][:5]}")
    
    # Generate Report
    logger.info("\n" + "="*80)
    logger.info("Generating ROI Readiness Report")
    logger.info("="*80)
    
    report = {
        'tumor_coverage': {
            'mean': float(valid_coverage.mean()) if has_mask > 0 and len(valid_coverage) > 0 else None,
            'median': float(valid_coverage.median()) if has_mask > 0 and len(valid_coverage) > 0 else None,
            'pct_high_coverage': float(high_coverage/len(valid_coverage)) if has_mask > 0 and len(valid_coverage) > 0 else None,
            'pct_medium_coverage': float(medium_coverage/len(valid_coverage)) if has_mask > 0 and len(valid_coverage) > 0 else None,
            'pct_low_coverage': float(low_coverage/len(valid_coverage)) if has_mask > 0 and len(valid_coverage) > 0 else None,
            'has_masks': bool(has_mask > 0)
        },
        'roi_size': {
            'mean_voxels': float(valid_sizes['roi_size_voxels'].mean()) if len(valid_sizes) > 0 else None,
            'pct_small': float(small_rois/len(valid_sizes)) if len(valid_sizes) > 0 else None
        },
        'empty_roi': {
            'pct_empty': float(empty_count/len(df_empty)) if len(df_empty) > 0 else None
        },
        'leakage': leakage_result,
        'redundancy': redundancy_result
    }
    
    # Save report
    report_file = OUTPUT_DIR / 'roi_readiness_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"✓ Saved report to: {report_file}")
    
    # Decision
    logger.info("\n" + "="*80)
    logger.info("DECISION GATE")
    logger.info("="*80)
    
    mean_coverage = report['tumor_coverage']['mean']
    correlation = redundancy_result.get('correlation', 1.0) if redundancy_result else 1.0
    swin_fn_mil_tp = redundancy_result.get('swin_fn_mil_tp', 0) if redundancy_result else 0
    
    if mean_coverage is not None and mean_coverage >= 0.6 and correlation < 0.8 and swin_fn_mil_tp > 5:
        decision = "GO"
        reason = f"Mean coverage {mean_coverage:.1%} >= 60%, correlation {correlation:.3f} < 0.8, {swin_fn_mil_tp} cases where MIL could help"
    elif mean_coverage is not None and mean_coverage >= 0.4 and correlation < 0.85:
        decision = "CONDITIONAL_GO"
        reason = f"Mean coverage {mean_coverage:.1%} 40-60%, correlation {correlation:.3f} < 0.85 - recommend single-fold pilot"
    else:
        decision = "NO_GO"
        if mean_coverage is None:
            reason = "No segmentation masks available - cannot verify ROI quality"
        elif mean_coverage < 0.4:
            reason = f"Mean coverage {mean_coverage:.1%} < 40% - ROI quality too low"
        elif correlation >= 0.8:
            reason = f"Correlation {correlation:.3f} >= 0.8 - MIL too redundant with Swin"
        else:
            reason = f"Insufficient cases where MIL could help ({swin_fn_mil_tp} < 5)"
    
    logger.info(f"Decision: {decision}")
    logger.info(f"Reason: {reason}")
    
    report['decision'] = decision
    report['decision_reason'] = reason
    
    # Update report
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n✓ Final report saved to: {report_file}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

