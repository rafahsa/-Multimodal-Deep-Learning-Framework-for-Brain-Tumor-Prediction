#!/usr/bin/env python3
"""
Add Segmentation Paths to Fold CSV Files

This script adds a 'path_seg' column to existing fold CSV files pointing to
segmentation mask files, without changing the patient assignments.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Go up 1 level from scripts/ to project root
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / 'splits'
DATA_ROOT = PROJECT_ROOT / 'data' / 'raw' / 'BraTS2018'


def add_seg_paths_to_fold(fold_file: Path, output_file: Path):
    """Add segmentation paths to a fold CSV file."""
    df = pd.read_csv(fold_file)
    
    # Add path_seg column
    seg_paths = []
    for _, row in df.iterrows():
        patient_id = row['patient_id']
        class_name = row['class']
        
        # Construct segmentation path: <class>/<patient_id>/<patient_id>_seg.nii.gz
        seg_path = f"{class_name}/{patient_id}/{patient_id}_seg.nii.gz"
        
        # Check if file exists (try both .nii.gz and .nii)
        full_path_gz = DATA_ROOT / seg_path
        full_path_nii = DATA_ROOT / seg_path.replace('.nii.gz', '.nii')
        
        if full_path_gz.exists():
            seg_paths.append(seg_path)
        elif full_path_nii.exists():
            seg_paths.append(seg_path.replace('.nii.gz', '.nii'))
        else:
            # File doesn't exist, but we'll still add the path (will fail during training if missing)
            seg_paths.append(seg_path)
            print(f"Warning: Segmentation file not found for {patient_id}: {full_path_gz}")
    
    df['path_seg'] = seg_paths
    
    # Save to output file
    df.to_csv(output_file, index=False)
    print(f"✓ Added segmentation paths to {output_file.name}: {len(df)} patients")


def main():
    print("="*80)
    print("ADDING SEGMENTATION PATHS TO FOLD CSV FILES")
    print("="*80)
    
    # Process all fold files
    for fold in range(5):
        for split_type in ['train', 'val']:
            input_file = SPLITS_DIR / f'fold_{fold}_{split_type}.csv'
            output_file = SPLITS_DIR / f'fold_{fold}_{split_type}_with_seg.csv'
            
            if input_file.exists():
                add_seg_paths_to_fold(input_file, output_file)
            else:
                print(f"Warning: {input_file} not found, skipping")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nNew CSV files saved with '_with_seg' suffix in: {SPLITS_DIR}")


if __name__ == '__main__':
    main()
