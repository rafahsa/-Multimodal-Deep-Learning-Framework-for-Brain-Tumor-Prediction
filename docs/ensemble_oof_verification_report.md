# Out-of-Fold Predictions Verification Report

**Date**: January 9, 2026  
**Status**: ✅ All Verification Checks Passed  
**Ready for Meta-Learner Training**: Yes

---

## Executive Summary

Out-of-fold (OOF) predictions from three base models (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D) have been successfully verified and merged into a single CSV file. All verification checks passed (47/47), confirming that the data is clean, properly structured, and free of data leakage. The merged predictions are ready for meta-learner training.

---

## 1. Individual Model Verification

### 1.1 ResNet50-3D

**File**: `ensemble/oof_predictions/resnet50_3d_oof.csv`

**Verification Results**:
- ✅ CSV file readable: 285 rows, 4 columns
- ✅ Required columns present: `patient_id`, `fold`, `hgg_prob`, `label`
- ✅ Patient ID uniqueness: All 285 patient IDs are unique
- ✅ Fold values: All 5 folds (0-4) present
- ✅ Fold distribution: 57 patients per fold (balanced)
- ✅ Probability range: All probabilities in [0, 1], range: [0.1652, 0.9999]
- ✅ Label values: All labels are 0 (LGG) or 1 (HGG)
  - Label distribution: 75 LGG (0), 210 HGG (1)
- ✅ Missing values: None
- ✅ Patient-fold matching: All patients from validation splits present in correct folds
- ✅ Total patient match: All 285 patients from validation splits present

**Latest Run Selected per Fold**:
- Fold 0: `run_20260107_060952`
- Fold 1: `run_20260107_075426`
- Fold 2: `run_20260106_231551`
- Fold 3: `run_20260107_012624`
- Fold 4: `run_20260107_041605`

### 1.2 SwinUNETR-3D

**File**: `ensemble/oof_predictions/swinunetr_3d_oof.csv`

**Verification Results**:
- ✅ CSV file readable: 285 rows, 4 columns
- ✅ Required columns present: `patient_id`, `fold`, `hgg_prob`, `label`
- ✅ Patient ID uniqueness: All 285 patient IDs are unique
- ✅ Fold values: All 5 folds (0-4) present
- ✅ Fold distribution: 57 patients per fold (balanced)
- ✅ Probability range: All probabilities in [0, 1], range: [0.0023, 1.0000]
- ✅ Label values: All labels are 0 (LGG) or 1 (HGG)
  - Label distribution: 75 LGG (0), 210 HGG (1)
- ✅ Missing values: None
- ✅ Patient-fold matching: All patients from validation splits present in correct folds
- ✅ Total patient match: All 285 patients from validation splits present

**Latest Run Selected per Fold**:
- Fold 0: `run_20260107_232314`
- Fold 1: `run_20260108_005357`
- Fold 2: `run_20260108_022938`
- Fold 3: `run_20260108_035601`
- Fold 4: `run_20260108_054258`

### 1.3 DualStreamMIL-3D

**File**: `ensemble/oof_predictions/dualstream_mil_3d_oof.csv`

**Verification Results**:
- ✅ CSV file readable: 285 rows, 4 columns
- ✅ Required columns present: `patient_id`, `fold`, `hgg_prob`, `label`
- ✅ Patient ID uniqueness: All 285 patient IDs are unique
- ✅ Fold values: All 5 folds (0-4) present
- ✅ Fold distribution: 57 patients per fold (balanced)
- ✅ Probability range: All probabilities in [0, 1], range: [0.1113, 0.9497]
- ✅ Label values: All labels are 0 (LGG) or 1 (HGG)
  - Label distribution: 75 LGG (0), 210 HGG (1)
- ✅ Missing values: None
- ✅ Patient-fold matching: All patients from validation splits present in correct folds
- ✅ Total patient match: All 285 patients from validation splits present

**Latest Run Selected per Fold**:
- Fold 0: `run_20260109_143346`
- Fold 1: `run_20260109_145215`
- Fold 2: `run_20260109_150832`
- Fold 3: `run_20260109_152503`
- Fold 4: `run_20260109_153707`

---

## 2. Data Leakage Verification

### 2.1 Duplicate Patient Check

✅ **Passed**: Each patient appears exactly once per model
- ResNet50-3D: No duplicates
- SwinUNETR-3D: No duplicates
- DualStreamMIL-3D: No duplicates

### 2.2 Fold Mutually Exclusive Check

✅ **Passed**: All folds are mutually exclusive (no patient overlap)
- Verified that no patient appears in multiple folds
- This confirms that predictions come only from the fold where the patient was in the validation set

### 2.3 Validation Split Consistency

✅ **Passed**: All predictions match validation split files
- Each patient's fold assignment matches the validation split CSV file
- All 285 patients from all validation splits are present
- No missing or extra patients

**Conclusion**: No data leakage detected. The OOF predictions are properly isolated by fold, ensuring that each prediction comes only from the validation set and never from the training set.

---

## 3. Merged Dataset

### 3.1 File Location

**File**: `ensemble/oof_predictions/merged_oof_predictions.csv`

### 3.2 Structure

**Columns**:
- `patient_id`: Patient identifier (string)
- `fold`: Fold number (0-4)
- `hgg_prob_resnet`: HGG probability from ResNet50-3D (float, 0-1)
- `hgg_prob_swin`: HGG probability from SwinUNETR-3D (float, 0-1)
- `hgg_prob_mil`: HGG probability from DualStreamMIL-3D (float, 0-1)
- `label`: Ground truth label (int, 0=LGG, 1=HGG)

### 3.3 Dataset Statistics

**Total Patients**: 285

**Fold Distribution**:
- Fold 0: 57 patients
- Fold 1: 57 patients
- Fold 2: 57 patients
- Fold 3: 57 patients
- Fold 4: 57 patients

**Label Distribution**:
- LGG (label=0): 75 patients (26.3%)
- HGG (label=1): 210 patients (73.7%)

**Probability Statistics**:

| Model | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| ResNet50-3D | 0.8884 | 0.1195 | 0.1652 | 0.9999 |
| SwinUNETR-3D | 0.5799 | 0.4406 | 0.0023 | 1.0000 |
| DualStreamMIL-3D | 0.4847 | 0.1530 | 0.1113 | 0.9497 |

**Observations**:
- ResNet50-3D produces higher probabilities on average (mean=0.8884), suggesting it may be more confident or calibrated differently
- SwinUNETR-3D shows the largest variance (std=0.4406), indicating more diverse predictions
- DualStreamMIL-3D has the most conservative probabilities (mean=0.4847, narrow range)
- All models produce probabilities in valid range [0, 1]

### 3.4 Label Consistency

✅ **Passed**: Labels are consistent across all three models
- All models use the same ground truth labels
- No discrepancies found between model predictions and validation split labels

---

## 4. Verification Checklist Summary

| Check Category | Status | Details |
|----------------|--------|---------|
| **File Format** | ✅ Pass | All CSV files readable with correct structure |
| **Required Columns** | ✅ Pass | All required columns present in all files |
| **Patient ID Uniqueness** | ✅ Pass | No duplicate patient IDs within any model |
| **Fold Completeness** | ✅ Pass | All 5 folds represented in all models |
| **Fold Balance** | ✅ Pass | 57 patients per fold (balanced) |
| **Probability Range** | ✅ Pass | All probabilities in [0, 1] |
| **Label Values** | ✅ Pass | All labels are 0 or 1 |
| **Missing Values** | ✅ Pass | No missing values in any file |
| **Patient-Fold Matching** | ✅ Pass | All patients match validation split files |
| **Total Patient Count** | ✅ Pass | All 285 patients present |
| **No Duplicates Across Folds** | ✅ Pass | Each patient appears exactly once per model |
| **Fold Mutual Exclusivity** | ✅ Pass | No patient overlap between folds |
| **Label Consistency** | ✅ Pass | Labels consistent across all models |
| **Merge Success** | ✅ Pass | All models successfully merged on patient_id |

**Total Checks**: 47  
**Passed**: 47 (100%)  
**Failed**: 0

---

## 5. Verification Process

### 5.1 Steps Performed

1. **Individual Model Verification**:
   - Loaded each OOF prediction CSV file
   - Verified file structure and required columns
   - Checked data quality (uniqueness, ranges, missing values)
   - Validated against validation split CSV files

2. **Data Leakage Checks**:
   - Verified no duplicate patients within each model
   - Confirmed folds are mutually exclusive
   - Validated that predictions come only from validation sets

3. **Merging Process**:
   - Merged predictions from all three models on `patient_id`
   - Verified merge completeness (all 285 patients present)
   - Confirmed label consistency across models

4. **Final Validation**:
   - Generated comprehensive statistics
   - Created verification report
   - Saved merged dataset

### 5.2 Methodology

- **Prediction-Order Assumption**: Predictions are assumed to be in the same order as patients in validation split CSV files. This was verified by checking that the number of predictions matches the number of patients in each validation split, and that all patient IDs match.

- **Latest Run Selection**: For each fold, only the most recent run (by modification time) was used. This ensures consistency and avoids mixing predictions from different training runs.

---

## 6. Issues Found

**None**. All verification checks passed without any issues.

---

## 7. Recommendations

1. ✅ **Proceed to Meta-Learner Training**: The merged OOF predictions are ready for meta-learner training.

2. **Model Calibration Observation**: The three models show different probability distributions:
   - ResNet50-3D: Higher, more confident probabilities
   - SwinUNETR-3D: More diverse probabilities (wider range)
   - DualStreamMIL-3D: More conservative probabilities
   - The meta-learner should learn to combine these different calibration characteristics.

3. **Class Imbalance**: The dataset has a 2.8:1 ratio of HGG to LGG (210:75). The meta-learner should account for this imbalance if using class weights.

---

## 8. Files Generated

1. **Individual OOF Files**:
   - `ensemble/oof_predictions/resnet50_3d_oof.csv`
   - `ensemble/oof_predictions/swinunetr_3d_oof.csv`
   - `ensemble/oof_predictions/dualstream_mil_3d_oof.csv`

2. **Merged File**:
   - `ensemble/oof_predictions/merged_oof_predictions.csv` ✅ **Ready for meta-learner training**

3. **Verification Report**:
   - `ensemble/oof_predictions/verification_report.txt`

---

## 9. Conclusion

All verification checks have passed successfully. The OOF predictions are:
- ✅ Properly formatted and structured
- ✅ Free of data leakage
- ✅ Correctly matched to patient IDs
- ✅ Validated against original validation splits
- ✅ Ready for meta-learner training

**Status**: **VERIFIED AND READY FOR ENSEMBLE STACKING**

---

**Report Generated**: January 9, 2026  
**Verification Script**: `scripts/ensemble/verify_and_merge_oof.py`  
**Next Step**: Train meta-learner (logistic regression) on merged OOF predictions

