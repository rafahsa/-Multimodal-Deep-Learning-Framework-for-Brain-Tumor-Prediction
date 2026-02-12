# Hybrid Safety-Net for Swin-1: Focused FN Reduction

**Objective:** Implement a focused hybrid safety-net on top of Swin-1 that acts only on uncertain cases to reduce FN while keeping FP under control.

**Core Philosophy:**
- Swin-1 remains the main decision maker
- Meta-decision model acts as secondary safety-net, triggered only when Swin-1 is uncertain
- Final decision: confident → Swin-1, uncertain → meta-decision

**Target Evaluation:**
- FN < 10 → research-level success
- FN < 15 → very strong
- FN < 25 → excellent

---

## Pipeline Overview

This pipeline consists of four steps:

1. **Define Uncertain Samples:** Tag samples as confident/uncertain based on Swin-1 predictions
2. **Train Meta-Decision on Uncertain Only:** Train Logistic Regression only on uncertain samples
3. **Hybrid Inference:** Apply hybrid decision logic (confident → Swin-1, uncertain → meta-decision)
4. **Evaluate:** Compare baseline vs hybrid system and provide GO/NO-GO decision

---

## Running the Pipeline

### Prerequisites

Ensure you have:
- `ensemble/oof_predictions/merged_oof_predictions.csv` (Swin-1 OOF predictions)
- `ensemble/results/meta_decision/meta_features.csv` (extracted features from `extract_meta_features_swin1.py`)

### Step 1: Define Uncertain Samples

```bash
cd /workspace/brain_tumor_project

python scripts/analysis/define_uncertain_samples_swin1.py
```

**Options:**
- `--prob-lower 0.30` (default): Lower bound for probability-based uncertainty
- `--prob-upper 0.60` (default): Upper bound for probability-based uncertainty
- `--entropy-percentile 75.0` (default): Percentile threshold for entropy-based uncertainty
- `--no-entropy`: Disable entropy-based uncertainty (use only probability-based)

**Outputs:**
- `ensemble/results/hybrid_safety_net/uncertain_samples.csv`

**Expected Runtime:** < 1 second

---

### Step 2: Train Meta-Decision on Uncertain Samples Only

```bash
cd /workspace/brain_tumor_project

python scripts/analysis/train_uncertain_meta_decision_swin1.py
```

**Outputs:**
- `ensemble/results/hybrid_safety_net/uncertain_meta_decision_results.json`
- `ensemble/results/hybrid_safety_net/uncertain_meta_predictions.csv`

**Expected Runtime:** < 1 minute

**Method:** Logistic Regression trained ONLY on uncertain samples using nested CV

---

### Step 3: Hybrid Inference

```bash
cd /workspace/brain_tumor_project

python scripts/analysis/hybrid_inference_swin1.py
```

**Outputs:**
- `ensemble/results/hybrid_safety_net/hybrid_predictions.csv`

**Expected Runtime:** < 1 second

**Logic:**
- Confident samples → use Swin-1 prediction
- Uncertain samples → use meta-decision prediction

---

### Step 4: Evaluate and Get GO/NO-GO Decision

```bash
cd /workspace/brain_tumor_project

python scripts/analysis/evaluate_hybrid_safety_net.py
```

**Outputs:**
- `ensemble/results/hybrid_safety_net/comparison_table.csv`
- `ensemble/results/hybrid_safety_net/evaluation_report.md`

**Expected Runtime:** < 1 second

**Evaluation:**
- Compares Swin-1 baseline vs Hybrid System
- Computes FN reduction, FP change, precision/recall improvements
- Provides GO/NO-GO decision based on:
  - FN < 25 AND
  - FN reduction ≥ 5 AND
  - FP increase ≤ +5

---

## Results Interpretation

### GO Decision

✅ **GO:** The hybrid safety-net provides meaningful FN reduction while keeping FP under control.

**Criteria:**
- FN < 25 (excellent/very strong/research-level)
- FN reduction ≥ 5 (meaningful improvement)
- FP increase ≤ +5 (acceptable)

### NO-GO Decision

❌ **NO-GO:** The hybrid safety-net does not meet the criteria.

**Possible Reasons:**
- FN reduction insufficient (FN ≥ 25)
- FN reduction not meaningful (<5 FN reduction)
- FP increase too high (>+5)

---

## File Structure

```
ensemble/results/hybrid_safety_net/
├── uncertain_samples.csv                    # Tagged samples (confident/uncertain)
├── uncertain_meta_decision_results.json     # Meta-decision training results
├── uncertain_meta_predictions.csv           # Meta-decision predictions (uncertain samples only)
├── hybrid_predictions.csv                   # Final hybrid predictions
├── comparison_table.csv                    # Baseline vs Hybrid comparison
└── evaluation_report.md                     # Evaluation report with GO/NO-GO decision
```

---

## Technical Details

### Uncertainty Definition

**Probability-based uncertainty:**
- Default: 0.30 ≤ hgg_prob_swin ≤ 0.60
- Samples in this range are considered uncertain

**Entropy-based uncertainty:**
- Default: Top 75th percentile of prediction entropy
- High entropy = high uncertainty

**Combined:**
- Uncertain if either probability-based OR entropy-based condition is met

### Meta-Decision Training

**Method:** Logistic Regression
- Trained ONLY on uncertain samples
- Nested CV: Train on uncertain samples from all other folds, predict on uncertain samples from current fold
- No data leakage: Confident samples never used

**Features Used:**
- Swin-1 probability
- Prediction entropy
- Tumor volume proxy (T1ce, FLAIR)
- Intensity variance (T1ce, FLAIR)
- GLCM texture stats (T1ce, FLAIR)

### Hybrid Inference Logic

**For each patient:**
1. Check uncertainty status
2. If confident → use Swin-1 prediction (threshold=0.5, unchanged)
3. If uncertain → use meta-decision prediction (threshold=0.5)

**Key Point:** Swin-1 threshold remains unchanged. Only uncertain cases are handled by meta-decision.

---

## Validation

- ✅ All evaluation is strict 5-fold OOF (no data leakage)
- ✅ Meta-decision trained only on uncertain samples
- ✅ Confident samples never used for meta-decision training
- ✅ No modifications to Swin-1 code or checkpoints
- ✅ No deep learning training
- ✅ Post-hoc analysis only

---

## Design Philosophy

This system is:
- **A clinical safety net** - NOT a replacement for Swin-1
- **NOT a global re-classifier** - Only acts on uncertain cases
- **Designed to:**
  - Catch hard FN cases
  - Preserve Swin-1 precision
  - Improve recall selectively

---

*Created: 2026-02-10*  
*Purpose: Focused hybrid safety-net for Swin-1 FN reduction*  
*Method: Lightweight Logistic Regression on uncertain samples only*

