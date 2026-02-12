# Selected Patients for Grad-CAM Visualization

This directory contains the selected patient IDs for Grad-CAM visualization.

## Selection Summary

**Total Patients**: 12

**Category Distribution**:
- **TP (True Positive)**: 3 patients - HGG correctly predicted as HGG (high confidence)
- **TN (True Negative)**: 3 patients - LGG correctly predicted as LGG (high confidence)
- **FP (False Positive)**: 3 patients - LGG incorrectly predicted as HGG (borderline errors)
- **FN (False Negative)**: 1 patient - HGG incorrectly predicted as LGG (actual error)
- **FN_nearmiss**: 2 patients - HGG correctly predicted but with low confidence (0.5 < prob < 0.7)

**Note**: Due to the excellent performance of ResNet50-3D (only 1 actual FN in OOF predictions), we included 2 "near-miss" cases (low-confidence HGG predictions) to complete the FN category. These are still clinically interesting as they represent borderline cases.

## Files

- `selected_patients.txt`: List of 12 patient IDs (one per line)
- `selected_patients_summary.csv`: Detailed summary with predictions, probabilities, and categories

## Selection Criteria

1. **TP**: High-confidence correct HGG predictions (prob > 0.99)
2. **TN**: High-confidence correct LGG predictions (prob < 0.3)
3. **FP**: Borderline false positives (prob ~0.7-0.76, true=LGG, pred=HGG)
4. **FN**: Actual false negatives + low-confidence HGG cases (prob < 0.7)

## Usage

To generate Grad-CAM heatmaps for these patients:

```bash
python scripts/analysis/generate_cnn_gradcam_3d.py \
    --checkpoint "AUTO" \
    --patient_ids_file data/selected_patients.txt \
    --fold 0 \
    --output_dir "ensemble/results/interpretability/cnn_gradcam"
```

## Patient Details

See `selected_patients_summary.csv` for full details including:
- Patient ID
- Category (TP/TN/FP/FN/FN_nearmiss)
- True label (0=LGG, 1=HGG)
- Predicted label (0=LGG, 1=HGG)
- HGG probability
- Confidence (|prob - 0.5|)
- Class name (LGG/HGG)
- Fold number

## Validation

All 12 patients have been verified to exist in the data directory:
- Location: `data/processed/stage_4_resize/train/<class>/<patient_id>/`
- All 4 modalities (T1, T1ce, T2, FLAIR) are present

## Selection Script

The selection was performed by:
```bash
python scripts/analysis/select_patients_for_gradcam.py
```

This script:
1. Loads ResNet50-3D OOF predictions
2. Computes TP/TN/FP/FN categories
3. Selects patients based on confidence and error characteristics
4. Verifies patients exist in data directory
5. Saves selection to files

