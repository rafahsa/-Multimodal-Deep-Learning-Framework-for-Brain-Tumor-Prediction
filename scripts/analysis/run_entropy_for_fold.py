#!/usr/bin/env python3
"""
Generate Entropy JSON files for a specific K-fold split

This script generates entropy-based slice informativeness analysis for patients
in a specific fold's train and validation sets. It only computes entropy for
missing patients (resumable).

Usage:
    python scripts/analysis/run_entropy_for_fold.py --fold 0 --modality flair --axis axial --top-k 16
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import SimpleITK as sitk
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.entropy_analysis import compute_slice_entropy, select_top_k_slices


def load_patient_ids_from_split(split_csv: Path) -> Set[str]:
    """
    Load patient IDs from a split CSV file.
    
    Args:
        split_csv: Path to split CSV file
        
    Returns:
        Set of patient IDs
    """
    patient_ids = set()
    
    if not split_csv.exists():
        raise FileNotFoundError(f"Split file not found: {split_csv}")
    
    with open(split_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_ids.add(row['patient_id'])
    
    return patient_ids


def get_entropy_json_path(patient_id: str, entropy_dir: Path) -> Path:
    """
    Get the expected entropy JSON file path for a patient.
    
    Naming convention: <patient_id>_entropy.json
    This matches what MILDataset expects: entropy_dir / f"{patient_id}_entropy.json"
    
    Args:
        patient_id: Patient ID
        entropy_dir: Directory containing entropy JSON files
        
    Returns:
        Path to entropy JSON file
    """
    return entropy_dir / f"{patient_id}_entropy.json"


def load_volume(volume_path: Path) -> torch.Tensor:
    """
    Load NIfTI volume and convert to torch tensor.
    
    Args:
        volume_path: Path to NIfTI file
        
    Returns:
        Torch tensor of shape (D, H, W)
    """
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume file not found: {volume_path}")
    
    image = sitk.ReadImage(str(volume_path))
    array = sitk.GetArrayFromImage(image)  # Shape: (D, H, W) in SimpleITK
    
    # Convert to torch tensor
    tensor = torch.from_numpy(array).float()
    
    return tensor


def compute_entropy_for_patient(
    patient_id: str,
    data_root: Path,
    entropy_dir: Path,
    modality: str,
    axis: str,
    top_k: int,
    use_cuda: bool = False,
    num_bins: int = 256
) -> Tuple[bool, str]:
    """
    Compute entropy for a single patient and save JSON.
    
    Args:
        patient_id: Patient ID
        data_root: Root directory containing Stage 4 data
        entropy_dir: Directory to save entropy JSON files
        modality: Modality to use ('flair', 't1', 't1ce', 't2')
        axis: Slice axis ('axial', 'coronal', 'sagittal')
        top_k: Number of top slices to select
        use_cuda: Whether to use GPU
        num_bins: Number of bins for histogram
        
    Returns:
        (success: bool, error_message: str)
    """
    try:
        # Check if entropy file already exists
        entropy_file = get_entropy_json_path(patient_id, entropy_dir)
        if entropy_file.exists():
            return True, "already exists"
        
        # Find volume file
        volume_path = None
        for class_name in ['HGG', 'LGG']:
            candidate = data_root / class_name / patient_id / f"{patient_id}_{modality}.nii.gz"
            if candidate.exists():
                volume_path = candidate
                break
            candidate = data_root / class_name / patient_id / f"{patient_id}_{modality}.nii"
            if candidate.exists():
                volume_path = candidate
                break
        
        if volume_path is None:
            return False, f"Volume file not found for {patient_id}"
        
        # Load volume
        volume = load_volume(volume_path)
        
        # Move to GPU if requested
        if use_cuda and torch.cuda.is_available():
            volume = volume.cuda()
        
        # Compute entropy per slice
        entropy_scores = compute_slice_entropy(
            volume=volume,
            axis=axis,
            num_bins=num_bins,
            normalize=False  # Keep raw entropy values
        )
        
        # Select top-k slices
        top_k_indices = select_top_k_slices(entropy_scores, k=top_k)
        
        # Create result dictionary
        result = {
            "patient_id": patient_id,
            "axis": axis,
            "modality": modality,
            "entropy_per_slice": entropy_scores,
            "top_k": top_k,
            "top_k_slices": top_k_indices,
            "num_slices": len(entropy_scores)
        }
        
        # Save JSON file
        entropy_dir.mkdir(parents=True, exist_ok=True)
        with open(entropy_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return True, "computed"
        
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Generate entropy JSON files for a K-fold split"
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help='Fold number (default: 0)'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='flair',
        choices=['t1', 't1ce', 't2', 'flair'],
        help='Modality to analyze (default: flair)'
    )
    parser.add_argument(
        '--axis',
        type=str,
        default='axial',
        choices=['axial', 'coronal', 'sagittal'],
        help='Slice axis (default: axial)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=16,
        help='Number of top slices to select (default: 16)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/processed/stage_4_resize/train',
        help='Root directory for Stage 4 data'
    )
    parser.add_argument(
        '--splits-dir',
        type=str,
        default='splits',
        help='Directory containing split CSV files'
    )
    parser.add_argument(
        '--entropy-dir',
        type=str,
        default='data/entropy',
        help='Directory to save entropy JSON files'
    )
    parser.add_argument(
        '--use-cuda',
        action='store_true',
        help='Use GPU if available'
    )
    parser.add_argument(
        '--num-bins',
        type=int,
        default=256,
        help='Number of bins for histogram (default: 256)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    splits_dir = project_root / args.splits_dir
    data_root = project_root / args.data_root
    entropy_dir = project_root / args.entropy_dir
    
    print("=" * 70)
    print("Entropy Generation for K-Fold Split")
    print("=" * 70)
    print(f"Fold: {args.fold}")
    print(f"Modality: {args.modality}")
    print(f"Axis: {args.axis}")
    print(f"Top-k: {args.top_k}")
    print(f"Data root: {data_root}")
    print(f"Entropy directory: {entropy_dir}")
    print(f"Use CUDA: {args.use_cuda and torch.cuda.is_available()}")
    print("=" * 70)
    
    # Load patient IDs from train and validation splits
    train_csv = splits_dir / f'fold_{args.fold}_train.csv'
    val_csv = splits_dir / f'fold_{args.fold}_val.csv'
    
    print(f"\nLoading patient IDs from splits...")
    train_patients = load_patient_ids_from_split(train_csv)
    val_patients = load_patient_ids_from_split(val_csv)
    
    # Combine (union, in case there's overlap)
    all_patients = train_patients | val_patients
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Total unique patients: {len(all_patients)}")
    
    # Check which entropy files already exist
    existing = []
    missing = []
    
    for patient_id in sorted(all_patients):
        entropy_file = get_entropy_json_path(patient_id, entropy_dir)
        if entropy_file.exists():
            existing.append(patient_id)
        else:
            missing.append(patient_id)
    
    print(f"\nExisting entropy files: {len(existing)}")
    print(f"Missing entropy files: {len(missing)}")
    
    if len(missing) == 0:
        print("\nAll entropy files already exist. Nothing to do.")
        return
    
    # Compute entropy for missing patients
    print(f"\nComputing entropy for {len(missing)} patients...")
    print("=" * 70)
    
    computed = 0
    failed = 0
    errors = []
    
    for i, patient_id in enumerate(missing, 1):
        success, message = compute_entropy_for_patient(
            patient_id=patient_id,
            data_root=data_root,
            entropy_dir=entropy_dir,
            modality=args.modality,
            axis=args.axis,
            top_k=args.top_k,
            use_cuda=args.use_cuda,
            num_bins=args.num_bins
        )
        
        if success:
            if message == "computed":
                computed += 1
                if computed % 10 == 0:
                    print(f"  Processed {computed}/{len(missing)} patients...")
        else:
            failed += 1
            errors.append(f"{patient_id}: {message}")
            print(f"  ERROR {patient_id}: {message}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total patients in fold {args.fold}: {len(all_patients)}")
    print(f"  - Existing entropy files: {len(existing)}")
    print(f"  - Newly generated: {computed}")
    print(f"  - Failed: {failed}")
    print(f"  - Still missing: {len(missing) - computed}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    if failed > 0:
        print(f"\nWARNING: {failed} patients failed. Check errors above.")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Entropy generation completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()

