#!/usr/bin/env python3
"""
Extract Meta-Features for Swin-1 Post-Hoc Meta-Decision Layer

This script extracts lightweight features from existing data:
- hgg_prob_swin (from Swin-1)
- Prediction entropy
- Tumor volume proxy
- Intensity variance (T1ce, FLAIR)
- Texture statistics (GLCM: contrast, entropy, homogeneity)

NO DEEP LEARNING - strictly post-hoc feature extraction.
"""

import sys
from pathlib import Path

# Set project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import logging
from typing import Dict
import SimpleITK as sitk
from scipy import stats
from scipy.ndimage import label
from skimage.feature import graycomatrix, graycoprops

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OOF_FILE = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / 'stage_4_resize' / 'train'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_decision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_entropy(prob: float) -> float:
    """Compute binary entropy: -p*log(p) - (1-p)*log(1-p)."""
    if prob <= 0 or prob >= 1:
        return 0.0
    return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)


def load_volume(patient_id: str, class_name: str, modality: str) -> np.ndarray:
    """Load a single modality volume for a patient."""
    patient_dir = DATA_ROOT / class_name / patient_id
    if not patient_dir.exists():
        return None
    
    volume_path = patient_dir / f"{patient_id}_{modality}.nii.gz"
    if not volume_path.exists():
        volume_path = patient_dir / f"{patient_id}_{modality}.nii"
    
    if not volume_path.exists():
        return None
    
    try:
        volume = sitk.ReadImage(str(volume_path))
        volume_array = sitk.GetArrayFromImage(volume).astype(np.float32)
        return volume_array
    except Exception as e:
        logger.warning(f"Error loading {volume_path}: {e}")
        return None


def compute_tumor_volume_proxy(volume: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.0) -> float:
    """
    Compute tumor volume proxy using intensity-based segmentation.
    
    Uses high-intensity regions (likely tumor) as proxy for volume.
    """
    # Remove background
    brain_mask = volume > np.percentile(volume, percentile_low)
    brain_values = volume[brain_mask]
    
    if len(brain_values) == 0:
        return 0.0
    
    # Use high-intensity regions (top 10% of brain values) as tumor proxy
    threshold = np.percentile(brain_values, 90)
    tumor_mask = volume > threshold
    
    # Count voxels in tumor-like regions
    tumor_voxels = np.sum(tumor_mask)
    total_voxels = volume.size
    
    # Volume proxy: fraction of high-intensity voxels
    volume_proxy = float(tumor_voxels / total_voxels) if total_voxels > 0 else 0.0
    
    return volume_proxy


def compute_intensity_variance(volume: np.ndarray, percentile_low: float = 1.0) -> float:
    """Compute intensity variance inside brain region."""
    brain_mask = volume > np.percentile(volume, percentile_low)
    brain_values = volume[brain_mask]
    
    if len(brain_values) == 0:
        return 0.0
    
    return float(np.var(brain_values))


def compute_glcm_features(volume: np.ndarray, distances=[1], angles=[0]) -> Dict:
    """
    Compute GLCM texture features: contrast, entropy, homogeneity.
    
    Uses 2D slices (axial) and averages across slices.
    """
    # Normalize volume to 0-255 for GLCM
    volume_normalized = volume.copy()
    if volume_normalized.max() > volume_normalized.min():
        volume_normalized = ((volume_normalized - volume_normalized.min()) / 
                           (volume_normalized.max() - volume_normalized.min()) * 255).astype(np.uint8)
    else:
        volume_normalized = volume_normalized.astype(np.uint8)
    
    # Compute GLCM for each axial slice and average
    all_contrast = []
    all_entropy = []
    all_homogeneity = []
    
    for z in range(volume_normalized.shape[0]):
        slice_2d = volume_normalized[z, :, :]
        
        # Skip slices with no variation
        if slice_2d.max() == slice_2d.min():
            continue
        
        try:
            glcm = graycomatrix(slice_2d, distances=distances, angles=angles, 
                              levels=256, symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            entropy_glcm = -np.sum(glcm * np.log(glcm + 1e-10))
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            
            all_contrast.append(contrast)
            all_entropy.append(entropy_glcm)
            all_homogeneity.append(homogeneity)
        except Exception as e:
            logger.debug(f"GLCM computation failed for slice {z}: {e}")
            continue
    
    # Average across slices
    return {
        'glcm_contrast': float(np.mean(all_contrast)) if all_contrast else 0.0,
        'glcm_entropy': float(np.mean(all_entropy)) if all_entropy else 0.0,
        'glcm_homogeneity': float(np.mean(all_homogeneity)) if all_homogeneity else 0.0
    }


def extract_features_for_patient(patient_id: str, label: int, hgg_prob_swin: float) -> Dict:
    """Extract all meta-features for a single patient."""
    # Determine class
    class_name = 'HGG' if label == 1 else 'LGG'
    
    # Initialize feature dict
    features = {
        'patient_id': patient_id,
        'label': label,
        'hgg_prob_swin': hgg_prob_swin,
        'prediction_entropy': compute_entropy(hgg_prob_swin)
    }
    
    # Extract features from T1ce and FLAIR (most informative for tumors)
    for modality in ['t1ce', 'flair']:
        volume = load_volume(patient_id, class_name, modality)
        
        if volume is not None:
            # Tumor volume proxy
            volume_proxy = compute_tumor_volume_proxy(volume)
            features[f'{modality}_volume_proxy'] = volume_proxy
            
            # Intensity variance
            intensity_var = compute_intensity_variance(volume)
            features[f'{modality}_intensity_variance'] = intensity_var
            
            # GLCM texture features
            glcm_features = compute_glcm_features(volume)
            features[f'{modality}_glcm_contrast'] = glcm_features['glcm_contrast']
            features[f'{modality}_glcm_entropy'] = glcm_features['glcm_entropy']
            features[f'{modality}_glcm_homogeneity'] = glcm_features['glcm_homogeneity']
        else:
            # Fill with zeros if volume not found
            features[f'{modality}_volume_proxy'] = 0.0
            features[f'{modality}_intensity_variance'] = 0.0
            features[f'{modality}_glcm_contrast'] = 0.0
            features[f'{modality}_glcm_entropy'] = 0.0
            features[f'{modality}_glcm_homogeneity'] = 0.0
    
    return features


def main():
    logger.info("="*80)
    logger.info("META-FEATURE EXTRACTION FOR SWIN-1 POST-HOC META-DECISION")
    logger.info("="*80)
    
    # Load OOF predictions
    logger.info(f"\nLoading OOF predictions from: {OOF_FILE}")
    df = pd.read_csv(OOF_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    # Verify required columns
    required_cols = ['patient_id', 'label', 'hgg_prob_swin']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract features
    logger.info("\nExtracting meta-features for all patients...")
    logger.info("This may take a few minutes...")
    
    features_list = []
    
    for idx, row in df.iterrows():
        patient_id = row['patient_id']
        label = row['label']
        hgg_prob_swin = row['hgg_prob_swin']
        
        features = extract_features_for_patient(patient_id, label, hgg_prob_swin)
        features_list.append(features)
        
        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} patients...")
    
    features_df = pd.DataFrame(features_list)
    
    # Save features
    features_file = OUTPUT_DIR / 'meta_features.csv'
    features_df.to_csv(features_file, index=False)
    logger.info(f"\n✓ Saved features to: {features_file}")
    logger.info(f"Features extracted: {len(features_df.columns)} columns")
    logger.info(f"Feature columns: {list(features_df.columns)}")
    
    logger.info("\n" + "="*80)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

