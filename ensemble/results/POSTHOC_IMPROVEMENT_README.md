# Post-Hoc Improvement of Swin-1: Complete Pipeline

**Objective:** Improve Swin-1 performance using ONLY post-hoc decision logic and lightweight feature-based rescue, WITHOUT retraining Swin-1.

**Hard Constraints:**
- Do NOT retrain, fine-tune, or modify Swin-1
- Use ONLY existing OOF predictions
- All evaluation must be strict 5-fold OOF (no leakage)
- All improvements must be reversible (wrapper on top of Swin-1)

**Target Performance:**
- FN < 10 (FN < 5 is excellent)
- FP < 10
- Precision ≥ 0.90
- Recall ≥ 0.90
- **All constraints must be met simultaneously**

---

## Pipeline Overview

This pipeline consists of two parts:

### Part A: Uncertainty-Aware Thresholding
Evaluates 4 decision policies:
1. Baseline (threshold=0.5)
2. Reject-band policy (prob in [0.35, 0.65] → HGG)
3. Confidence-aware thresholding (entropy-based)
4. Fold-specific calibrated threshold

### Part B: Feature-Level Rescue
Evaluates 2 methods:
1. Rule-based rescue (flip LGG→HGG based on high-risk features)
2. Lightweight logistic regression (if rule-based helps)

---

## Running the Pipeline

### Step 1: Run Part A (Thresholding)

```bash
cd /workspace/brain_tumor_project

python scripts/analysis/posthoc_thresholding_swin1.py
```

**Outputs:**
- `ensemble/results/posthoc_thresholding/thresholding_results.json`
- `ensemble/results/posthoc_thresholding/thresholding_results.md`

**Expected Runtime:** < 1 minute

---

### Step 2: Run Part B (Feature Rescue)

```bash
cd /workspace/brain_tumor_project

python scripts/analysis/feature_rescue_swin1.py
```

**Outputs:**
- `ensemble/results/feature_rescue/patient_features.csv` (cached features)
- `ensemble/results/feature_rescue/rescue_results.json`
- `ensemble/results/feature_rescue/rescue_results.md`

**Expected Runtime:** 10-30 minutes (feature extraction is slow)

**Note:** Features are cached after first run. Subsequent runs will be faster.

---

### Step 3: Generate Executive Summary

```bash
cd /workspace/brain_tumor_project

python scripts/analysis/executive_summary_posthoc.py
```

**Outputs:**
- `ensemble/results/posthoc_improvement_executive_summary.md`

**Expected Runtime:** < 1 second

---

## Results Interpretation

### If a Method Meets All Constraints

✅ **GO:** The method achieves:
- FN < 10 AND FP < 10 AND Precision ≥ 0.90 AND Recall ≥ 0.90

**Action:** Proceed with the best method for Swin-1 post-hoc improvement.

### If No Method Meets All Constraints

❌ **NO-GO:** None of the post-hoc methods achieve all target constraints.

**Action:** 
1. Consider model retraining (e.g., Swin-2)
2. Evaluate ensemble methods
3. Review target constraints (may be too strict)

---

## File Structure

```
ensemble/results/
├── posthoc_thresholding/
│   ├── thresholding_results.json
│   └── thresholding_results.md
├── feature_rescue/
│   ├── patient_features.csv
│   ├── rescue_results.json
│   └── rescue_results.md
└── posthoc_improvement_executive_summary.md
```

---

## Technical Details

### Part A: Thresholding Policies

1. **Baseline:** Simple threshold at 0.5
2. **Reject-band:** If prob in [0.35, 0.65], predict HGG; else use 0.5 threshold
3. **Confidence-aware:** Use entropy to identify high-uncertainty regions, apply different thresholds
4. **Fold-calibrated:** Learn optimal threshold on train folds, apply to val fold

### Part B: Feature Extraction

Features extracted from T1ce and FLAIR volumes:
- Mean intensity
- Standard deviation
- Variance
- Entropy (histogram-based)
- Skewness
- Kurtosis
- 95th percentile
- 99th percentile

### Part B: Rescue Methods

1. **Rule-based:** Learn feature thresholds from Swin-1 FN cases in training, flip LGG→HGG if high-risk features detected
2. **Lightweight model:** Train logistic regression on [Swin-1 prob + features], reject if violates constraints

---

## Validation

- ✅ All evaluation is strict 5-fold OOF (no data leakage)
- ✅ Feature thresholds learned on train folds only
- ✅ Models trained on train folds, evaluated on val fold
- ✅ No modifications to Swin-1 code or checkpoints

---

*Created: 2026-02-10*  
*Purpose: Post-hoc improvement of Swin-1 without retraining*

