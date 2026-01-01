#!/usr/bin/env python3
"""
Generate K-Fold Cross-Validation Splits

This script creates Stratified K-Fold cross-validation splits for the BraTS2018 dataset.
Splits are patient-level to prevent data leakage.

Usage:
    python scripts/splits/make_kfold_splits.py --index data/index/stage4_index.csv --k 5 --seed 42
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


def load_index(index_path: Path) -> Tuple[List[Dict], List[str], List[int]]:
    """
    Load Stage 4 patient index.
    
    Args:
        index_path: Path to index CSV file
        
    Returns:
        Tuple of (patient_dicts, patient_ids, class_labels)
    """
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    patients = []
    patient_ids = []
    class_labels = []
    
    with open(index_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient = {
                'patient_id': row['patient_id'],
                'class': row['class'],
                'class_label': int(row['class_label']),
                'path_t1': row['path_t1'],
                'path_t1ce': row['path_t1ce'],
                'path_t2': row['path_t2'],
                'path_flair': row['path_flair']
            }
            patients.append(patient)
            patient_ids.append(row['patient_id'])
            class_labels.append(int(row['class_label']))
    
    return patients, patient_ids, class_labels


def generate_kfold_splits(
    patient_ids: List[str],
    class_labels: List[int],
    k: int = 5,
    seed: int = 42,
    shuffle: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate Stratified K-Fold splits.
    
    Args:
        patient_ids: List of patient IDs
        class_labels: List of class labels (0=LGG, 1=HGG)
        k: Number of folds
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle before splitting
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Convert to numpy arrays
    patient_ids = np.array(patient_ids)
    class_labels = np.array(class_labels)
    
    # Create StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    
    # Generate splits
    splits = []
    for train_idx, val_idx in skf.split(patient_ids, class_labels):
        splits.append((train_idx, val_idx))
    
    return splits


def verify_splits(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    patients: List[Dict],
    patient_ids: List[str],
    class_labels: List[int]
) -> bool:
    """
    Verify that splits have no patient overlap and preserve class distribution.
    
    Args:
        splits: List of (train_indices, val_indices) tuples
        patients: List of patient dictionaries
        patient_ids: List of patient IDs
        class_labels: List of class labels
        
    Returns:
        True if verification passes
    """
    print("\n" + "=" * 60)
    print("SPLIT VERIFICATION")
    print("=" * 60)
    
    all_pass = True
    patient_ids_arr = np.array(patient_ids)
    class_labels_arr = np.array(class_labels)
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_ids = set(patient_ids_arr[train_idx])
        val_ids = set(patient_ids_arr[val_idx])
        
        # Check for overlap
        overlap = train_ids & val_ids
        if overlap:
            print(f"ERROR: Fold {fold_idx} has patient overlap: {overlap}")
            all_pass = False
        else:
            print(f"✓ Fold {fold_idx}: No patient overlap")
        
        # Check class distribution
        train_labels = class_labels_arr[train_idx]
        val_labels = class_labels_arr[val_idx]
        
        train_hgg = (train_labels == 1).sum()
        train_lgg = (train_labels == 0).sum()
        val_hgg = (val_labels == 1).sum()
        val_lgg = (val_labels == 0).sum()
        
        train_total = len(train_idx)
        val_total = len(val_idx)
        
        print(f"  Train: {train_total} patients (HGG: {train_hgg}, LGG: {train_lgg})")
        print(f"  Val:   {val_total} patients (HGG: {val_hgg}, LGG: {val_lgg})")
        
        # Check class ratios
        train_ratio = train_hgg / train_total if train_total > 0 else 0
        val_ratio = val_hgg / val_total if val_total > 0 else 0
        
        print(f"  Train HGG ratio: {train_ratio:.3f}, Val HGG ratio: {val_ratio:.3f}")
    
    print("=" * 60)
    
    if all_pass:
        print("✓ All verifications passed!")
    else:
        print("✗ Some verifications failed!")
    
    return all_pass


def save_split_csv(
    patients: List[Dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_idx: int,
    output_dir: Path
):
    """
    Save train/val splits as CSV files.
    
    Args:
        patients: List of patient dictionaries
        train_idx: Training set indices
        val_idx: Validation set indices
        fold_idx: Fold index
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['patient_id', 'class', 'class_label', 'path_t1', 'path_t1ce', 'path_t2', 'path_flair']
    
    # Save train split
    train_path = output_dir / f"fold_{fold_idx}_train.csv"
    with open(train_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in train_idx:
            writer.writerow(patients[idx])
    print(f"  Saved: {train_path} ({len(train_idx)} patients)")
    
    # Save val split
    val_path = output_dir / f"fold_{fold_idx}_val.csv"
    with open(val_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in val_idx:
            writer.writerow(patients[idx])
    print(f"  Saved: {val_path} ({len(val_idx)} patients)")


def save_split_json(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    patients: List[Dict],
    patient_ids: List[str],
    k: int,
    seed: int,
    output_path: Path
):
    """
    Save all splits as JSON.
    
    Args:
        splits: List of (train_indices, val_indices) tuples
        patients: List of patient dictionaries
        patient_ids: List of patient IDs
        k: Number of folds
        seed: Random seed
        output_path: Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    patient_ids_arr = np.array(patient_ids)
    
    split_data = {
        'k': k,
        'seed': seed,
        'total_patients': len(patients),
        'folds': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_ids = patient_ids_arr[train_idx].tolist()
        val_ids = patient_ids_arr[val_idx].tolist()
        
        fold_data = {
            'fold': fold_idx,
            'train_patients': train_ids,
            'val_patients': val_ids,
            'train_count': len(train_ids),
            'val_count': len(val_ids)
        }
        split_data['folds'].append(fold_data)
    
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"\nSaved split summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate K-Fold cross-validation splits"
    )
    parser.add_argument(
        '--index',
        type=str,
        default='data/index/stage4_index.csv',
        help='Path to Stage 4 index CSV file'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of folds (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='splits',
        help='Output directory for split files (default: splits)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='splits/kfold_5fold_seed42.json',
        help='Output JSON file for split summary'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    index_path = project_root / args.index
    output_dir = project_root / args.output_dir
    output_json = project_root / args.output_json
    
    print("=" * 60)
    print("K-Fold Cross-Validation Split Generator")
    print("=" * 60)
    print(f"Index file: {index_path}")
    print(f"K-folds: {args.k}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load index
    print(f"\nLoading index from: {index_path}")
    patients, patient_ids, class_labels = load_index(index_path)
    
    # Print dataset summary
    class_counts = defaultdict(int)
    for patient in patients:
        class_counts[patient['class']] += 1
    
    print(f"\nDataset summary:")
    print(f"  Total patients: {len(patients)}")
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        print(f"  {class_name}: {count} ({count/len(patients)*100:.1f}%)")
    
    # Generate splits
    print(f"\nGenerating {args.k}-fold stratified splits...")
    splits = generate_kfold_splits(patient_ids, class_labels, k=args.k, seed=args.seed, shuffle=True)
    print(f"Generated {len(splits)} folds")
    
    # Verify splits
    verify_splits(splits, patients, patient_ids, class_labels)
    
    # Save splits
    print(f"\nSaving split files to: {output_dir}")
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {fold_idx}:")
        save_split_csv(patients, train_idx, val_idx, fold_idx, output_dir)
    
    # Save JSON summary
    save_split_json(splits, patients, patient_ids, args.k, args.seed, output_json)
    
    print("\n" + "=" * 60)
    print("Split generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
