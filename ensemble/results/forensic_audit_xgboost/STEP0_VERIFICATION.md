# Step 0: Verification of Existing Nested CV Implementation

## Analysis Date
2026-02-08

## Current Pipeline Structure

### 1. OOF Predictions Generation
- **Status**: ✅ **CORRECT**
- **Method**: Base models use 5-fold CV
- **Evidence**: 
  - `scripts/ensemble/prepare_oof_predictions.py` generates OOF predictions
  - Each fold's validation predictions are collected
  - `fold` column indicates which fold each prediction came from
  - Each patient appears exactly once (verified in audit)

### 2. Meta-Learner Training
- **Status**: ❌ **NO OUTER CV SPLIT**
- **Current Method**: 
  - `scripts/ensemble/train_meta_learner.py` loads ALL OOF predictions (285 samples)
  - Trains meta-learner on ALL samples
  - Evaluates on the SAME samples used for training
- **Evidence**: 
  - Line 336-352 in `train_meta_learner.py`: `X, y = prepare_data(df)` uses full dataset
  - Line 352: `metrics = evaluate_model(model, X, y, threshold=args.threshold)` evaluates on same data

### 3. Calibration & Threshold Selection
- **Status**: ❌ **USES SAME DATA AS TRAINING**
- **Current Method**:
  - `scripts/ensemble/calibrate_and_sweep_thresholds.py` uses full OOF set
  - Splits 70/30 for calibration/threshold selection
  - But this split is within the SAME data used for meta-learner training
- **Evidence**:
  - Line 167: `train_test_split(X, y, ...)` splits the full OOF set
  - No outer CV separation

### 4. Patient-Level Splitting
- **Status**: ⚠️ **NOT USED FOR META-LEARNER**
- **Current Method**:
  - `patient_id` column exists in merged OOF CSV
  - But meta-learner training does NOT split at patient level
  - Uses sample-level splitting (if any)

## Critical Findings

### Data Leakage Present
1. **Meta-learner trains and evaluates on same data**: All 285 samples are used for both training and evaluation
2. **Calibration/threshold selection uses training data**: The 70/30 split is within the training set, not separate
3. **No outer CV loop**: There is no outer cross-validation to separate meta-learner training from testing

### What IS Correct
1. ✅ OOF predictions are correctly generated using CV at base model level
2. ✅ Each patient appears exactly once in OOF predictions
3. ✅ Fold information is preserved

## Verdict

### ❌ **NOT IMPLEMENTED (leakage risk present)**

**Missing Components**:
1. **Outer CV split**: No separation between meta-learner training and test data
2. **Nested structure**: OOF predictions should be regenerated for each outer fold's training set only
3. **Patient-level splitting**: Should split at patient level, not sample level
4. **Independent evaluation**: Outer-test fold should never be used for training, calibration, or threshold selection

## Required Implementation

A true nested CV structure must:
1. **Outer Loop (5 folds)**:
   - Split data into outer-train (80%) and outer-test (20%) at patient level
   - For each outer fold:
     - Outer-train: Used for everything (OOF generation, meta-learner training, calibration, threshold selection)
     - Outer-test: Held out completely, only touched once for final evaluation

2. **Inner Pipeline (within outer-train only)**:
   - Generate OOF predictions for base models using CV within outer-train
   - Train meta-learner on outer-train OOF predictions
   - Fit calibration on subset of outer-train
   - Select threshold on subset of outer-train

3. **Final Evaluation (outer-test only)**:
   - Apply trained models, calibration, and threshold to outer-test
   - Record metrics (FN, FP, TN, TP, Recall, Precision, F1, Cost)
   - Aggregate across all outer folds

## Conclusion

**The current pipeline does NOT implement nested CV correctly.**
The forensic audit correctly identified that evaluation was performed on the same data used for training, leading to optimistic results (FN=0 vs realistic FN=9).

**Action Required**: Implement true nested CV from scratch as specified in Step 1.

