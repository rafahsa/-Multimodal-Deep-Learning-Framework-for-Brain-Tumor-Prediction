#!/usr/bin/env python3
"""
Generate Full Recursive Structural Dump of Brain Tumor Project

This script generates a comprehensive structural overview including:
- Raw full directory tree
- Structured annotated architecture tree
- Workflow dependency explanation
- Final professional architecture summary
"""

import os
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_FILE = PROJECT_ROOT / 'FULL_PROJECT_STRUCTURE_DUMP.md'

def should_exclude(path_str):
    """Check if path should be excluded from tree."""
    exclude_patterns = [
        '__pycache__',
        '.git',
        '.venv',
        'data/raw',
        'data/brats2018',
    ]
    return any(pattern in path_str for pattern in exclude_patterns)

def get_all_paths(root_path):
    """Get all paths recursively."""
    all_paths = []
    for root, dirs, files in os.walk(root_path):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
        
        for name in dirs + files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, root_path)
            
            if should_exclude(rel_path):
                continue
            
            if os.path.isdir(full_path):
                all_paths.append((rel_path, 'dir'))
            elif os.path.isfile(full_path):
                # Skip very large files
                if name.endswith(('.nii.gz', '.tar.gz')):
                    continue
                all_paths.append((rel_path, 'file'))
    
    return sorted(all_paths)

def build_tree_structure(paths):
    """Build hierarchical tree structure."""
    tree = {}
    for path, item_type in paths:
        parts = Path(path).parts
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = {'type': item_type}
    return tree

def format_tree_markdown(tree, indent=0, max_depth=None, current_depth=0):
    """Format tree as markdown with annotations."""
    lines = []
    prefix = '  ' * indent
    
    if max_depth and current_depth >= max_depth:
        return lines
    
    for name, value in sorted(tree.items()):
        if isinstance(value, dict) and 'type' in value:
            # Leaf node (file or directory)
            item_type = value['type']
            icon = '📁' if item_type == 'dir' else '📄'
            lines.append(f"{prefix}{icon} `{name}`")
        else:
            # Directory node
            lines.append(f"{prefix}📁 `{name}/`")
            sub_lines = format_tree_markdown(value, indent + 1, max_depth, current_depth + 1)
            lines.extend(sub_lines)
    
    return lines

def get_annotations():
    """Get annotations for key directories."""
    return {
        'scripts/preprocessing': 'Data preprocessing pipeline scripts (Stages 1-4)',
        'scripts/training': 'Model training scripts for base models',
        'scripts/ensemble': 'Ensemble learning pipeline scripts',
        'scripts/analysis': 'Analysis, visualization, and verification scripts',
        'scripts/splits': 'Data splitting utilities (K-Fold generation)',
        'ensemble/models': 'Trained meta-learner models (Logistic Regression)',
        'ensemble/oof_predictions': 'Out-of-fold predictions from base models',
        'ensemble/results': 'Ensemble evaluation results and metrics',
        'ensemble/visualizations': 'Publication-ready visualizations',
        'ensemble/calibrators': 'Calibration models (Platt scaling)',
        'ensemble/audits': 'Audit reports and verification analyses',
        'results': 'Base model training results and outputs',
        'data/processed': 'Preprocessed data (Stages 1-4)',
        'data/entropy': 'Entropy metadata for MIL slice selection',
        'data/index': 'Patient index files',
        'splits': 'K-Fold cross-validation split definitions',
        'configs': 'YAML configuration files for preprocessing',
        'models': 'Model architecture definitions',
        'utils': 'Reusable utility modules',
        'pretrained': 'Pretrained model weights',
        'archive_minimal_runs': 'Archived minimal model artifacts',
        'logs': 'Training and preprocessing logs',
        'docs': 'Comprehensive technical documentation',
    }

def generate_document():
    """Generate the full structural dump document."""
    print("Generating full structural dump...")
    
    # Get all paths
    all_paths = get_all_paths(PROJECT_ROOT)
    print(f"Found {len(all_paths)} items")
    
    # Build tree
    tree = build_tree_structure(all_paths)
    
    # Get annotations
    annotations = get_annotations()
    
    # Generate document
    doc_lines = []
    doc_lines.append("# Brain Tumor Classification Project - Full Recursive Structural Dump\n")
    doc_lines.append("**Generated:** Full recursive directory structure with annotations\n")
    doc_lines.append("**Total Items:** " + str(len(all_paths)) + "\n")
    doc_lines.append("---\n\n")
    
    # Section A: Raw Full Directory Tree
    doc_lines.append("## Section A: Raw Full Directory Tree\n\n")
    doc_lines.append("### Complete File and Directory Listing (ALL " + str(len(all_paths)) + " items)\n\n")
    doc_lines.append("```\n")
    for path, item_type in all_paths:  # NO LIMIT - include ALL items
        icon = '📁' if item_type == 'dir' else '📄'
        doc_lines.append(f"{icon} {path}\n")
    doc_lines.append("```\n\n")
    
    # Section B: Structured Annotated Architecture Tree
    doc_lines.append("## Section B: Structured Annotated Architecture Tree\n\n")
    doc_lines.append("### Key Directories with Annotations\n\n")
    
    for key_path, annotation in sorted(annotations.items()):
        doc_lines.append(f"#### `{key_path}/`\n")
        doc_lines.append(f"**Purpose:** {annotation}\n\n")
    
    doc_lines.append("### Full Directory Tree (Complete Structure)\n\n")
    doc_lines.append("```\n")
    tree_lines = format_tree_markdown(tree, max_depth=None)  # NO DEPTH LIMIT
    doc_lines.extend(tree_lines)  # Include ALL tree lines
    doc_lines.append("```\n\n")
    
    # Section C: Workflow Dependency Explanation
    doc_lines.append("## Section C: Workflow Dependency Explanation\n\n")
    doc_lines.append("### Complete ML Pipeline Flow\n\n")
    doc_lines.append("""
```
1. DATA PREPARATION
   ├── scripts/preprocessing/run_stage1_n4.py
   │   └── Output: data/processed/stage_1_n4/
   ├── scripts/preprocessing/run_stage2_zscore.py
   │   └── Output: data/processed/stage_2_zscore/
   ├── scripts/preprocessing/run_stage3_crop.py
   │   └── Output: data/processed/stage_3_crop/
   └── scripts/preprocessing/run_stage4_resize.py
       └── Output: data/processed/stage_4_resize/ (FINAL PREPROCESSED DATA)

2. DATA SPLITTING
   ├── scripts/splits/build_stage4_index.py
   │   └── Output: data/index/stage4_index.csv
   └── scripts/splits/make_kfold_splits.py
       └── Output: splits/fold_X_train.csv, splits/fold_X_val.csv

3. BASE MODEL TRAINING
   ├── scripts/training/train_resnet50_3d.py
   │   └── Output: results/ResNet50-3D/runs/fold_X/
   ├── scripts/training/train_swin_unetr_3d.py
   │   └── Output: results/SwinUNETR-3D/runs/fold_X/
   └── scripts/training/train_dual_stream_mil.py
       └── Output: results/DualStreamMIL-3D/runs/fold_X/

4. OOF GENERATION
   ├── scripts/ensemble/prepare_oof_predictions.py
   │   └── Output: ensemble/oof_predictions/per_fold/
   └── scripts/ensemble/verify_and_merge_oof.py
       └── Output: ensemble/oof_predictions/merged_oof_predictions.csv

5. CALIBRATION
   └── scripts/ensemble/calibrate_and_sweep_thresholds.py
       └── Output: ensemble/results/calibration/

6. NESTED CROSS-VALIDATION
   └── scripts/ensemble/nested_cv_meta_features.py
       └── Output: ensemble/results/nested_cv_meta_features/

7. META-LEARNER TRAINING
   └── scripts/ensemble/train_meta_learner_roi_mil.py
       └── Output: ensemble/results/meta_learner_roi_mil/

8. EVALUATION & VISUALIZATION
   ├── scripts/analysis/generate_final_ensemble_figures_42_48.py
   │   └── Output: ensemble/visualizations/FINAL13.2.2026/
   └── scripts/analysis/generate_nested_cv_publication_figures.py
       └── Output: ensemble/visualizations/nested_cv_final/

9. PUBLICATION ARTIFACTS
   └── ensemble/results/meta_learner_roi_mil/
       ├── meta_learner_logistic_regression.joblib
       ├── meta_learner_metrics.json
       └── predictions.csv
```
\n""")
    
    # Section D: Final Professional Architecture Summary
    doc_lines.append("## Section D: Final Professional Architecture Summary\n\n")
    doc_lines.append("""
### System Architecture Overview

This project implements a **multimodal deep learning ensemble framework** for brain tumor grade classification using 3D MRI volumes from the BraTS 2018 dataset. The architecture follows a **modular, stage-based design** with clear separation between data preprocessing, model training, ensemble learning, and evaluation stages.

### Directory Structure Summary

- **`scripts/`**: All executable Python scripts organized by functionality
- **`data/`**: Centralized data repository (raw, processed, metadata)
- **`models/`**: PyTorch model architecture definitions
- **`utils/`**: Reusable utility modules
- **`configs/`**: YAML configuration files
- **`splits/`**: K-Fold cross-validation split definitions
- **`results/`**: Base model training outputs
- **`ensemble/`**: Complete ensemble learning pipeline
- **`pretrained/`**: Pretrained model weights
- **`archive_minimal_runs/`**: Archived minimal artifacts
- **`logs/`**: Training and preprocessing logs
- **`docs/`**: Comprehensive technical documentation

### Key Workflow Components

1. **Data Preparation**: 4-stage preprocessing pipeline (N4, Z-score, Crop, Resize)
2. **Model Training**: 3 base models trained with K-Fold cross-validation
3. **OOF Generation**: Out-of-fold predictions extracted and merged
4. **Calibration**: Platt scaling applied to improve probability reliability
5. **Nested CV**: Strict nested cross-validation with meta-features
6. **Meta-Learner**: Logistic Regression trained on engineered features
7. **Evaluation**: Comprehensive metrics and visualizations
8. **Publication**: Final artifacts for thesis/reporting

### Data Flow

```
Raw Data → Preprocessing → K-Fold Splits → Base Model Training → 
OOF Predictions → Meta-Feature Engineering → Meta-Learner Training → 
Calibration → Threshold Selection → Nested CV Evaluation → 
Final Metrics & Visualizations
```

### Final Artifacts Location

- **Final Model**: `ensemble/results/meta_learner_roi_mil/meta_learner_logistic_regression.joblib`
- **Final Metrics**: `ensemble/results/meta_learner_roi_mil/meta_learner_metrics.json`
- **Final Predictions**: `ensemble/results/meta_learner_roi_mil/predictions.csv`
- **Nested CV Results**: `ensemble/results/nested_cv_meta_features/`
- **Publication Figures**: `ensemble/visualizations/FINAL13.2.2026/`

### Statistics

- **Total Project Items**: """ + str(len(all_paths)) + """
- **Base Models**: 3 (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D)
- **K-Fold Splits**: 5 folds (seed=42)
- **Meta-Features**: 14 engineered features
- **Final Ensemble**: Logistic Regression with 3 base model inputs
\n""")
    
    # Write document
    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(doc_lines)
    
    print(f"Document generated: {OUTPUT_FILE}")
    print(f"Total lines: {len(doc_lines)}")

if __name__ == "__main__":
    generate_document()

