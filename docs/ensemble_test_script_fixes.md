# Ensemble Test Script - Fixes and Improvements

**Date**: January 10, 2026  
**Status**: ✅ Comprehensive Validation and Error Handling Added

---

## Summary of Fixes

### 1. **Model Parameter Fixes**
   - ✅ **ResNet50-3D**: Fixed `input_channels` → `in_channels`
   - ✅ **SwinUNETR-3D**: Fixed `input_channels` → `in_channels` (verified)
   - ✅ **DualStreamMIL-3D**: 
     - Removed invalid `input_channels` parameter (hardcoded to 4 in InstanceEncoder)
     - Fixed `instance_encoder` → `instance_encoder_backbone`
     - Removed invalid `bag_size` parameter (only used during inference, not model creation)
     - Added proper parameters: `instance_encoder_input_size=224`, `attention_type='gated'`, `fusion_method='concat'`

### 2. **Comprehensive NaN/Inf Validation**
   - ✅ Added NaN/Inf checks at every step:
     - After loading each modality
     - After stacking modalities
     - After model forward passes (logits)
     - After softmax (probabilities)
     - After extracting HGG probability
     - Before creating features array
     - Before passing to meta-learner
     - After meta-learner prediction

### 3. **Prediction Extraction Improvements**
   - ✅ Explicit `.detach().cpu()` before `.item()` to ensure no gradient tracking
   - ✅ Explicit float conversion for all probabilities
   - ✅ Validation that probabilities are in range [0, 1]
   - ✅ Validation that probabilities sum to 1 (for softmax outputs)
   - ✅ Shape validation for all model outputs

### 4. **Feature Array Validation**
   - ✅ Feature order explicitly verified: `[ResNet, Swin, MIL]` matches training order
   - ✅ Feature array shape validation: `(1, 3)`
   - ✅ Feature dtype: `np.float32`
   - ✅ Meta-learner feature count validation (expects 3 features)
   - ✅ Comprehensive logging of feature values before passing to meta-learner

### 5. **Directory Structure Handling**
   - ✅ Supports patient subdirectories: `test/DATA_FOR_TEST/{PATIENT_ID}/files`
   - ✅ Supports flat structure: `test/DATA_FOR_TEST/files`
   - ✅ Flexible modality naming: handles both `T1ce` and `T1c`
   - ✅ Supports both `.nii` and `.nii.gz` file formats
   - ✅ Detailed error messages showing searched paths and available files

### 6. **Error Handling and Logging**
   - ✅ Comprehensive error messages at each step
   - ✅ Full traceback logging for debugging
   - ✅ Validation messages for successful operations
   - ✅ Clear indication of which step failed

### 7. **Volume Loading Validation**
   - ✅ Validates all 4 modalities are loaded
   - ✅ Checks for NaN/Inf in loaded volumes
   - ✅ Verifies volume shapes and dimensions
   - ✅ Ensures proper channel ordering: T1, T1ce/T1c, T2, FLAIR

### 8. **Meta-Learner Validation**
   - ✅ Validates meta-learner has required methods (`predict`, `predict_proba`)
   - ✅ Checks feature count matches expectations
   - ✅ Validates probability output shape and validity
   - ✅ Explicit extraction and validation of HGG probability

---

## Feature Order Verification

**Meta-Learner Training Order** (from `train_meta_learner.py`):
```python
FEATURE_COLUMNS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
```

**Test Script Feature Order**:
```python
features = np.array([[hgg_prob_resnet, hgg_prob_swin, hgg_prob_mil]])
```

✅ **Order matches correctly**: [0] = ResNet, [1] = Swin, [2] = MIL

---

## Validation Checklist

At each step, the script validates:

1. **File Loading**:
   - [x] All 4 modalities found
   - [x] Files are readable
   - [x] No NaN/Inf in loaded data

2. **Volume Preprocessing**:
   - [x] Correct shape: (4, D, H, W)
   - [x] Correct channel order
   - [x] No NaN/Inf after stacking

3. **Model Predictions**:
   - [x] Logits shape correct: (1, 2)
   - [x] No NaN/Inf in logits
   - [x] Probabilities sum to 1
   - [x] HGG probability in range [0, 1]
   - [x] Explicit detach and conversion to float

4. **Feature Array**:
   - [x] Shape: (1, 3)
   - [x] Dtype: float32
   - [x] Order: [ResNet, Swin, MIL]
   - [x] No NaN/Inf
   - [x] All values in [0, 1]

5. **Meta-Learner**:
   - [x] Feature count matches (3)
   - [x] Prediction shape correct
   - [x] Probability array shape: (1, 2)
   - [x] Probabilities sum to 1
   - [x] HGG probability valid

---

## Common Issues and Solutions

### Issue: NaN in predictions
**Solution**: Added comprehensive NaN checks at every step. If NaN appears:
- Check volume loading (file corruption)
- Check model outputs (model issue)
- Check feature array construction (extraction issue)

### Issue: Wrong feature order
**Solution**: Explicit validation that order matches training: `[ResNet, Swin, MIL]`

### Issue: Invalid probability values
**Solution**: Validates all probabilities are in [0, 1] and sum to 1 (for softmax outputs)

### Issue: Model parameter errors
**Solution**: Fixed all parameter names to match actual function signatures

### Issue: File not found
**Solution**: Improved error messages showing:
- Searched paths
- Available files
- Directory structure

---

## Usage

```bash
# Basic usage
python scripts/ensemble/test_ensemble_on_new_patients.py

# With custom test directory
python scripts/ensemble/test_ensemble_on_new_patients.py --test-dir /workspace/brain_tumor_project/test/DATA_FOR_TEST

# With specific fold
python scripts/ensemble/test_ensemble_on_new_patients.py --fold 0
```

---

## Expected Output Format

```
================================================================================
ENSEMBLE MODEL TEST RESULTS
================================================================================

Patient: UCSF-PDGM-0004
--------------------------------------------------------------------------------
  ResNet50-3D HGG probability:    0.923456
  SwinUNETR-3D HGG probability:   0.875432
  DualStreamMIL-3D HGG probability: 0.901234
  Ensemble HGG probability:       0.945678
  Ensemble prediction:            HGG
HGG probability for patient 0004: 0.95

Patient: UCSF-PDGM-0005
--------------------------------------------------------------------------------
  ...
HGG probability for patient 0005: 0.72

================================================================================
Processed 2 patients successfully
================================================================================
```

---

## Debugging Tips

1. **Check logs**: The script provides detailed logging at each step
2. **Verify file paths**: Error messages show searched paths and available files
3. **Check predictions**: Each model's prediction is logged before combining
4. **Validate features**: Feature array values are logged before meta-learner
5. **Check meta-learner**: Validates feature count and coefficient shape

---

**Script Location**: `scripts/ensemble/test_ensemble_on_new_patients.py`  
**Last Updated**: January 10, 2026

