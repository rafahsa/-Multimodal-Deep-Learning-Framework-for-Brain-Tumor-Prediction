#!/usr/bin/env python3
"""
Build Stage 4 Index

This script scans the Stage 4 output directory and creates an index file
listing all patients with their class labels and modality file paths.

Usage:
    python scripts/splits/build_stage4_index.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List


def discover_patients(data_root: Path) -> List[Dict]:
    """
    Discover all patients in Stage 4 outputs.
    
    Args:
        data_root: Path to data/processed/stage_4_resize/train/
        
    Returns:
        List of patient dictionaries with metadata
    """
    patients = []
    modalities = ['t1', 't1ce', 't2', 'flair']
    
    for class_name in ['HGG', 'LGG']:
        class_dir = data_root / class_name
        if not class_dir.exists():
            continue
        
        for patient_dir in sorted(class_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            
            patient_id = patient_dir.name
            patient_data = {
                'patient_id': patient_id,
                'class': class_name,
                'class_label': 1 if class_name == 'HGG' else 0,  # HGG=1, LGG=0
                'modalities': {}
            }
            
            # Check for all modalities
            all_modalities_present = True
            for modality in modalities:
                modality_file = patient_dir / f"{patient_id}_{modality}.nii.gz"
                if not modality_file.exists():
                    # Try .nii extension as fallback
                    modality_file = patient_dir / f"{patient_id}_{modality}.nii"
                
                if modality_file.exists():
                    # Store relative path from train directory
                    patient_data['modalities'][modality] = str(modality_file.relative_to(data_root))
                else:
                    all_modalities_present = False
                    break
            
            if all_modalities_present:
                patients.append(patient_data)
    
    return patients


def save_index_csv(patients: List[Dict], output_path: Path):
    """
    Save patient index as CSV.
    
    Args:
        patients: List of patient dictionaries
        output_path: Output CSV file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['patient_id', 'class', 'class_label', 'path_t1', 'path_t1ce', 'path_t2', 'path_flair']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for patient in patients:
            row = {
                'patient_id': patient['patient_id'],
                'class': patient['class'],
                'class_label': patient['class_label'],
                'path_t1': patient['modalities'].get('t1', ''),
                'path_t1ce': patient['modalities'].get('t1ce', ''),
                'path_t2': patient['modalities'].get('t2', ''),
                'path_flair': patient['modalities'].get('flair', '')
            }
            writer.writerow(row)
    
    print(f"Saved CSV index: {output_path} ({len(patients)} patients)")


def save_index_json(patients: List[Dict], output_path: Path):
    """
    Save patient index as JSON.
    
    Args:
        patients: List of patient dictionaries
        output_path: Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(patients, f, indent=2)
    print(f"Saved JSON index: {output_path} ({len(patients)} patients)")


def main():
    parser = argparse.ArgumentParser(description="Build Stage 4 patient index")
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/processed/stage_4_resize/train',
        help='Path to Stage 4 train directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/index/stage4_index.csv',
        help='Output index file path'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json', 'both'],
        default='csv',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    data_root = project_root / args.data_root
    output_path = project_root / args.output
    
    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        sys.exit(1)
    
    print(f"Scanning Stage 4 directory: {data_root}")
    patients = discover_patients(data_root)
    
    if not patients:
        print("Error: No patients found!")
        sys.exit(1)
    
    # Print summary
    class_counts = {}
    for patient in patients:
        class_name = patient['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nFound {len(patients)} patients:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    
    # Save index
    if args.format in ['csv', 'both']:
        save_index_csv(patients, output_path)
    
    if args.format in ['json', 'both']:
        json_path = output_path.with_suffix('.json')
        save_index_json(patients, json_path)


if __name__ == '__main__':
    main()
