"""
Step 4: Non-DL Feature Extraction

Extracts patient-level features that don't require tumor masks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict
import SimpleITK as sitk
from scipy import stats
# GLCM features removed for simplicity - can be added if needed

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / 'stage_4_resize' / 'train'


def extract_non_dl_features(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Extract non-DL features for each patient."""
    logger.info("Extracting non-DL features...")
    
    features_list = []
    
    for idx, row in df.iterrows():
        patient_id = row['patient_id']
        fold = row['fold']
        label = row['label']
        
        # Determine class
        class_name = 'HGG' if label == 1 else 'LGG'
        patient_dir = DATA_ROOT / class_name / patient_id
        
        if not patient_dir.exists():
            logger.warning(f"Patient directory not found: {patient_dir}")
            features_list.append({
                'patient_id': patient_id,
                'fold': fold,
                'label': label
            })
            continue
        
        # Load volumes
        modalities = ['t1', 't1ce', 't2', 'flair']
        patient_features = {
            'patient_id': patient_id,
            'fold': fold,
            'label': label
        }
        
        for mod in modalities:
            volume_path = patient_dir / f"{patient_id}_{mod}.nii.gz"
            if not volume_path.exists():
                volume_path = patient_dir / f"{patient_id}_{mod}.nii"
            
            if not volume_path.exists():
                continue
            
            try:
                volume = sitk.ReadImage(str(volume_path))
                volume_array = sitk.GetArrayFromImage(volume).astype(np.float32)
                
                # Remove background (zero values)
                brain_mask = volume_array > 1e-6
                brain_values = volume_array[brain_mask]
                
                if len(brain_values) == 0:
                    continue
                
                # Intensity statistics
                patient_features[f'{mod}_mean'] = float(np.mean(brain_values))
                patient_features[f'{mod}_std'] = float(np.std(brain_values))
                patient_features[f'{mod}_skew'] = float(stats.skew(brain_values))
                patient_features[f'{mod}_kurtosis'] = float(stats.kurtosis(brain_values))
                patient_features[f'{mod}_p1'] = float(np.percentile(brain_values, 1))
                patient_features[f'{mod}_p5'] = float(np.percentile(brain_values, 5))
                patient_features[f'{mod}_p50'] = float(np.percentile(brain_values, 50))
                patient_features[f'{mod}_p95'] = float(np.percentile(brain_values, 95))
                patient_features[f'{mod}_p99'] = float(np.percentile(brain_values, 99))
                
                # Global entropy
                hist, _ = np.histogram(brain_values, bins=256)
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                patient_features[f'{mod}_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
                
                # Gradient energy (simplified)
                grad = np.gradient(volume_array)
                grad_magnitude = np.sqrt(sum(g**2 for g in grad))
                patient_features[f'{mod}_gradient_energy'] = float(np.mean(grad_magnitude[brain_mask]))
                
            except Exception as e:
                logger.warning(f"Error processing {mod} for {patient_id}: {e}")
        
        features_list.append(patient_features)
    
    # Create DataFrame
    df_features = pd.DataFrame(features_list)
    
    # Merge with original predictions
    df_merged = df.merge(df_features, on=['patient_id', 'fold', 'label'], how='left')
    
    # Save
    output_file = output_dir / 'non_dl_features.csv'
    df_features.to_csv(output_file, index=False)
    logger.info(f"✓ Saved non-DL features to: {output_file}")
    
    merged_file = output_dir / 'oof_predictions_with_features.csv'
    df_merged.to_csv(merged_file, index=False)
    logger.info(f"✓ Saved merged predictions with features to: {merged_file}")
    
    logger.info(f"\nExtracted {len(df_features.columns) - 3} features per patient")
    
    return df_merged

