# Meta-Decision Layer for Swin-1: Post-Hoc FN Reduction

**Objective:** Implement a lightweight meta-decision layer on top of Swin-1 to reduce False Negatives (FN) while keeping False Positives (FP) under control.

**Constraints:**
- Swin-1 remains unchanged (no retraining, no threshold tuning)
- Use existing OOF predictions only
- Strictly post-hoc analysis
- No deep learning training

**Target Evaluation:**
- FN < 10 → research-level success
- FN < 15 → very strong
- FN < 25 → excellent

---

## Pipeline Overview

This pipeline consists of three steps:

1. **Feature Extraction:** Extract lightweight features from existing data
2. **Meta-Decision Training:** Train Logistic Regression on OOF data
3. **Evaluation:** Compare against Swin-1 baseline and provide GO/NO-GO decision

---

## Running the Pipeline

### Step 1: Extract Meta-Features

```bash
cd /workspace/brain_tumor_project

python scripts/analysis/extract_meta_features_swin1.py
```

**Outputs:**
- `ensemble/results/meta_decision/meta_features.csv`

**Expected Runtime:** 10-30 minutes (feature extraction from MRI volumes)

**Features Extracted:**
- `hgg_prob_swin` (from Swin-1)
- `prediction_entropy` (uncertainty measure)
- `t1ce_volume_proxy` (tumor volume proxy from T1ce)
- `t1ce_intensity_variance` (intensity variance from T1ce)
- `t1ce_glcm_contrast`, `t1ce_glcm_entropy`, `t1ce_glcm_homogeneity` (texture stats from T1ce)
- Same features for `flair` modality

---

### Step 2: Train Meta-Decision Model

```bash
cd /workspace/brain_tumor_project

python scripts/analysis/train_meta_decision_swin1.py
```

**Outputs:**
- `ensemble/results/meta_decision/meta_decision_predictions.csv`
- `ensemble/results/meta_decision/meta_decision_results.json`

**Expected Runtime:** < 1 minute

**Method:** Logistic Regression with nested CV (train on all folds except current, predict on current fold)

---

### Step 3: Evaluate and Compare

```bash
cd /workspace/brain_tumor_project

python scripts/analysis/evaluate_meta_decision_swin1.py
```

**Outputs:**
- `ensemble/results/meta_decision/comparison_table.csv`
- `ensemble/results/meta_decision/evaluation_report.md`

**Expected Runtime:** < 1 second

**Evaluation:**
- Compares Swin-1 baseline vs Swin-1 + Meta-Decision
- Computes FN reduction, FP change, precision/recall improvements
- Provides GO/NO-GO decision based on:
  - FN reduction level (research-level/very strong/excellent)
  - FP acceptability (not too high)
  - Meaningful FN reduction (≥5 FN reduction)

---

## Results Interpretation

### GO Decision

✅ **GO:** The meta-decision layer provides meaningful FN reduction while keeping FP under control.

**Criteria:**
- FN reduction is excellent/very strong/research-level
- FP is acceptable (not significantly higher than baseline)
- FN reduction is meaningful (≥5 FN reduction)

### NO-GO Decision

❌ **NO-GO:** The meta-decision layer does not meet the criteria.

**Possible Reasons:**
- FN reduction insufficient (FN ≥ 25)
- FP too high (significantly higher than baseline)
- FN reduction not meaningful (<5 FN reduction)

---

## File Structure

```
ensemble/results/meta_decision/
├── meta_features.csv                    # Extracted features
├── meta_decision_predictions.csv        # Meta-decision predictions
├── meta_decision_results.json           # Training results
├── comparison_table.csv                 # Baseline vs Meta-decision comparison
└── evaluation_report.md                 # Evaluation report with GO/NO-GO decision
```

---

## Technical Details

### Feature Extraction

**Tumor Volume Proxy:**
- Uses high-intensity regions (top 10% of brain values) as tumor proxy
- Computes fraction of high-intensity voxels

**Intensity Variance:**
- Computes variance of intensity values inside brain region

**GLCM Texture Features:**
- Computes Gray-Level Co-occurrence Matrix (GLCM) features
- Features: contrast, entropy, homogeneity
- Computed on 2D axial slices and averaged

### Meta-Decision Model

**Method:** Logistic Regression
- Class balancing: `class_weight='balanced'`
- Regularization: `C=1.0`
- Nested CV: Train on all folds except current, predict on current fold

**Features Used:**
- Swin-1 probability
- Prediction entropy
- Tumor volume proxy (T1ce, FLAIR)
- Intensity variance (T1ce, FLAIR)
- GLCM texture stats (T1ce, FLAIR)

---

## Validation

- ✅ All evaluation is strict 5-fold OOF (no data leakage)
- ✅ Model trained on train folds, evaluated on val fold
- ✅ No modifications to Swin-1 code or checkpoints
- ✅ No deep learning training
- ✅ Post-hoc analysis only

---

*Created: 2026-02-10*  
*Purpose: Post-hoc meta-decision layer for Swin-1 FN reduction*  
*Method: Lightweight Logistic Regression (no deep learning)*

