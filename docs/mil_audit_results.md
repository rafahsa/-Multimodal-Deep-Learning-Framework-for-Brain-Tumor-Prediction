# Dual-Stream MIL Training Audit Results

**Date:** 2026-01-09  
**Model:** DualStreamMIL-3D  
**Task:** Brain Tumor Classification (HGG vs LGG)  
**Dataset:** BraTS 2018, 5-fold cross-validation

## 1. RESULTS AUDIT (Per-Fold Summary)

### Fold 0 (run_20260109_032250)
- **Best Epoch:** 12
- **Best Val AUC:** 0.9095
- **Final Metrics:** AUC=0.9095, F1=0.7689, Acc=0.7895
- **EMA Used:** Yes (best_ema.pt)
- **Confusion Matrix:**
  - LGG: 15/15 correct (100%)
  - HGG: 3/42 correct (7.1%) ⚠️ **SEVERE CLASS IMBALANCE**
- **Stability:** 
  - Val loss fluctuates: 0.679 → 0.451 (best) → 0.586 (final)
  - Val AUC: 0.6976 → 0.9095 (best) → 0.8857 (final)
  - Train/val loss gap: 0.30 (train) vs 0.59 (val) - **clear overfitting**
- **Anomalies:** Model heavily biased toward LGG predictions

### Fold 1 (run_20260109_033031)
- **Best Epoch:** 16
- **Best Val AUC:** 0.9746
- **Final Metrics:** AUC=0.9746, F1=0.6110, Acc=0.6140
- **EMA Used:** Yes (best_ema.pt)
- **Confusion Matrix:**
  - LGG: 15/15 correct (100%)
  - HGG: 29/42 correct (69.0%) ✅ **BETTER BALANCE**
- **Stability:**
  - Val loss fluctuates heavily: 0.676 → 0.350 (best) → 0.508 (final)
  - Val AUC: 0.7683 → 0.9746 (best) → 0.9730 (final)
  - Train/val loss gap: 0.29 (train) vs 0.51 (val) - **overfitting**
- **Anomalies:** High AUC but low F1/Acc suggests threshold calibration issues

### Fold 2 (run_20260109_033921)
- **Best Epoch:** 20
- **Best Val AUC:** 0.9730
- **Final Metrics:** AUC=0.9730, F1=0.8507, Acc=0.8772
- **EMA Used:** Yes (best_ema.pt)
- **Confusion Matrix:**
  - LGG: 15/15 correct (100%)
  - HGG: 30/42 correct (71.4%) ✅ **BEST BALANCE**
- **Stability:**
  - Val loss: 0.683 → 0.399 (best) → 0.559 (final)
  - Val AUC: 0.6833 → 0.9730 (best) → 0.9333 (final)
  - Train/val loss gap: 0.29 (train) vs 0.56 (val) - **overfitting**
- **Anomalies:** Most stable fold, best overall performance

### Fold 3 (run_20260109_034929)
- **Best Epoch:** 12
- **Best Val AUC:** 0.9730
- **Final Metrics:** AUC=0.9730, F1=0.8421, Acc=0.8596
- **EMA Used:** Yes (best_ema.pt)
- **Confusion Matrix:**
  - LGG: 15/15 correct (100%)
  - HGG: 11/42 correct (26.2%) ⚠️ **CLASS IMBALANCE**
- **Stability:**
  - Val loss: 0.683 → 0.379 (best) → 0.563 (final)
  - Val AUC: 0.8214 → 0.9730 (best) → 0.8921 (final)
  - Train/val loss gap: 0.37 (train) vs 0.56 (val) - **overfitting**
- **Anomalies:** Model biased toward LGG predictions

### Fold 4 (run_20260109_035711)
- **Best Epoch:** 11
- **Best Val AUC:** 0.9746
- **Final Metrics:** AUC=0.9746, F1=0.8787, Acc=0.9123
- **EMA Used:** Yes (best_ema.pt)
- **Confusion Matrix:**
  - LGG: 15/15 correct (100%)
  - HGG: 0/42 correct (0.0%) 🚨 **CRITICAL FAILURE - ALL PREDICTIONS ARE LGG**
- **Stability:**
  - Val loss: 0.688 → 0.377 (best) → 0.661 (final)
  - Val AUC: 0.6611 → 0.9746 (best) → 0.9476 (final)
  - Train/val loss gap: 0.40 (train) vs 0.66 (val) - **severe overfitting**
- **Anomalies:** **Model completely fails to predict HGG class** - all predictions are LGG. This is a catastrophic failure mode.

## 2. CROSS-VALIDATION SUMMARY

| Metric | Mean ± Std | Min | Max |
|--------|-----------|-----|-----|
| **AUC** | 0.9609 ± 0.0273 | 0.9095 | 0.9746 |
| **F1** | 0.7902 ± 0.1000 | 0.6110 | 0.8787 |
| **Accuracy** | 0.8088 ± 0.1130 | 0.6140 | 0.9123 |

**Critical Issues:**
1. **Severe class imbalance** in 3/5 folds (0, 3, 4)
2. **Fold 4 catastrophic failure** - model predicts all samples as LGG
3. **High variance** in F1 and Accuracy across folds
4. **Consistent overfitting** - train loss decreases while val loss increases/fluctuates

## 3. ROOT CAUSE ANALYSIS

### 3.1 Class Imbalance Failure
**Problem:** Model heavily biased toward LGG predictions in folds 0, 3, and 4.

**Root Causes:**
1. **WeightedRandomSampler** may not be sufficient for MIL bag-level supervision
2. **Label smoothing (0.1)** may be too weak to prevent overconfidence
3. **Temperature annealing** may cause selection collapse in later epochs
4. **Attention collapse** - model may be focusing on LGG-discriminative slices only

### 3.2 Attention/Selection Mechanism Issues
**Problem:** No logging of attention entropy or selection statistics makes diagnosis impossible.

**Missing Diagnostics:**
- Mean attention entropy per epoch
- Top-1 attention weight (collapse indicator)
- Effective number of instances (diversity metric)
- Selection confidence distribution

### 3.3 Overfitting Patterns
**Problem:** Consistent train/val loss gap across all folds.

**Contributing Factors:**
1. **Small dataset** (≈228 patients per fold)
2. **Large bag size** (32 slices) → high memorization capacity
3. **Random bag sampling** → instability across epochs
4. **Insufficient regularization** despite multiple mechanisms

### 3.4 Plot Saving Behavior
**Current:** Plots saved only when `epoch == best_epoch` (new best found)

**Issue:** Not a bug, but limits debugging capability. Missing per-epoch plots make it hard to track attention/selection evolution.

## 4. RECOMMENDED IMPROVEMENTS

### Priority 1: Fix Class Imbalance (Critical)
1. **Increase label smoothing** from 0.1 to 0.2-0.3
2. **Add class weights to CrossEntropyLoss** (in addition to WeightedRandomSampler)
3. **Implement Focal Loss** with moderate γ (1.5-2.0) to focus on hard examples
4. **Add threshold calibration** for final predictions

### Priority 2: Add MIL Diagnostics
1. **Log attention entropy** per epoch
2. **Log selection statistics** (top-1 weight, effective instances)
3. **Track attention collapse** indicators

### Priority 3: Improve Regularization
1. **Reduce bag size** further (32 → 24 or 16)
2. **Increase dropout** (0.5 → 0.6)
3. **Add stochastic depth** to encoder
4. **Implement top-k pooling** instead of full attention (reduces memorization)

### Priority 4: Stabilize Training
1. **Fix bag sampling** - use deterministic or curriculum sampling
2. **Add plot-every argument** for better debugging
3. **Improve early stopping** - monitor class-specific metrics

### Priority 5: Architecture Improvements
1. **Consider ResNet34** encoder (more stable than ResNet18)
2. **Implement gated attention** with stronger regularization
3. **Add instance-level dropout** before aggregation

## 5. NEXT STEPS

1. Implement comprehensive improvements (see implementation plan)
2. Re-train all 5 folds with new configuration
3. Verify class balance in all folds
4. Compare CV metrics before/after improvements

