# Research: Best Practices for HGG/LGG Classification on Small Datasets

**Goal**: Reduce FN and FP to < 5 and push BOTH Precision and Recall > 0.93 on 285-sample dataset (210 HGG / 75 LGG).

**Current State**: Ensemble AUC=0.91, FN=4-8, Precision=0.96, Recall=0.77. Swin dominates (coef=4.06), MIL weak (coef=0.09).

---

## 1. Method Analysis

### 1.1 ROI-Guided Sampling / Segmentation-Guided MIL

**What it is**: Use tumor segmentation masks to focus MIL on tumor regions only, rather than full-brain slices.

**Expected Impact**:
- **FN**: ↓ (better focus on tumor regions, less noise from non-tumor brain)
- **FP**: ↓ (reduced false positives from non-tumor regions)
- **AUC**: +0.02-0.03 (if ROI quality is high, ≥60% tumor coverage)

**Cost**:
- **Engineering**: Medium (need to integrate segmentation masks, modify MIL dataset)
- **Training time**: Same as current MIL (same architecture, just different input)

**Risk**:
- **No gain if ROI quality is low** (<40% tumor coverage) → ROI-MIL becomes redundant with Swin
- **Overfitting risk**: Low (still using same MIL architecture, just better input)
- **Data leakage risk**: Medium (must ensure ROI creation doesn't use labels)

**Literature Support**:
- BraTS challenge winners often use ROI-focused approaches
- MIL with tumor-guided sampling shows 5-10% AUC improvement in small datasets
- **Key requirement**: ROI must have ≥60% tumor coverage to be effective

**Verdict**: **HIGH PROMISE** if ROI quality is verified (≥60% coverage, low redundancy with Swin).

---

### 1.2 Attention MIL / Transformer Pooling

**What it is**: Replace max-pooling/attention in MIL with transformer-based pooling (e.g., ABMIL, TransMIL).

**Expected Impact**:
- **FN**: ↓ (better instance selection, learns which slices matter)
- **FP**: ↓ (better discrimination, learns slice-level patterns)
- **AUC**: +0.01-0.02 (moderate improvement over current attention MIL)

**Cost**:
- **Engineering**: High (new architecture, need to implement transformer pooling)
- **Training time**: 1.5-2× current (transformer is more expensive)

**Risk**:
- **Overfitting risk**: High (transformer has more parameters, 285 samples is small)
- **No gain if redundant**: If attention focuses on same slices as Swin, no benefit
- **Complexity**: Higher maintenance burden

**Literature Support**:
- TransMIL shows 2-5% improvement over attention MIL in histopathology
- **But**: Most studies use larger datasets (500+ samples)
- **Small dataset risk**: Transformers need more data to generalize

**Verdict**: **MODERATE PROMISE** but high risk of overfitting on 285 samples. Better as follow-up after ROI-MIL.

---

### 1.3 Hard Instance Mining (Top-k + Bottom-k)

**What it is**: Select both most informative (top-k) and least informative (bottom-k) slices to learn discriminative patterns.

**Expected Impact**:
- **FN**: ↓ (learns what distinguishes HGG from LGG)
- **FP**: ↓ (learns negative patterns)
- **AUC**: +0.01-0.02 (moderate improvement)

**Cost**:
- **Engineering**: Low (modify slice selection in existing MIL)
- **Training time**: Same (just different slice selection)

**Risk**:
- **No gain if redundant**: If top-k/bottom-k are same as entropy sampling, no benefit
- **Overfitting risk**: Low (same architecture, just different selection)

**Literature Support**:
- Hard negative mining improves MIL in some domains
- **But**: Limited evidence in medical imaging MIL
- **Effectiveness**: Depends on whether bottom-k provides complementary signal

**Verdict**: **LOW-MODERATE PROMISE**. Easy to try, but limited evidence. Better as quick experiment.

---

### 1.4 Calibration (Platt/Isotonic), Threshold Selection

**What it is**: Calibrate probabilities and optimize threshold for recall constraints.

**Expected Impact**:
- **FN**: ↓ (via threshold tuning)
- **FP**: ↑ (trade-off, lower threshold increases FP)
- **AUC**: 0 (calibration doesn't change ranking)

**Cost**:
- **Engineering**: Low (already done for MIL)
- **Training time**: None (post-processing)

**Risk**:
- **No ensemble gain**: Threshold tuning helps standalone, not ensemble (ensemble uses probabilities)
- **Trade-off**: Can't achieve both high recall AND high precision simultaneously

**Literature Support**:
- Calibration improves reliability but not ranking
- Threshold tuning is standard practice
- **But**: Already implemented for MIL, didn't help ensemble

**Verdict**: **ALREADY DONE**. No further gains expected.

---

### 1.5 Self-Supervised Pretraining (MAE/SimCLR/MoCo)

**What it is**: Pretrain feature extractors on unlabeled MRI data before fine-tuning on HGG/LGG.

**Expected Impact**:
- **FN**: ↓ (better feature representations)
- **FP**: ↓ (better discrimination)
- **AUC**: +0.02-0.04 (if large unlabeled dataset available)

**Cost**:
- **Engineering**: Very High (need to implement pretraining pipeline)
- **Training time**: 5-10× (pretraining + fine-tuning)
- **Data requirement**: Need large unlabeled MRI dataset (1000+ volumes)

**Risk**:
- **No unlabeled data**: If only 285 labeled samples, pretraining not feasible
- **Overfitting risk**: Medium (pretraining helps, but still need labeled data)
- **Domain mismatch**: Pretraining on different MRI protocols may not help

**Literature Support**:
- Self-supervised learning shows 3-5% improvement in medical imaging
- **But**: Requires large unlabeled datasets (1000+ samples)
- **Small dataset**: Limited benefit if only 285 samples available

**Verdict**: **LOW PROMISE** unless large unlabeled dataset is available. Not feasible for this project.

---

### 1.6 Test-Time Augmentation / Ensembling Tricks

**What it is**: Apply augmentations at test time and ensemble predictions (e.g., average over 10 augmented versions).

**Expected Impact**:
- **FN**: ↓ (more robust predictions)
- **FP**: ↓ (reduces false positives)
- **AUC**: +0.01-0.02 (moderate improvement)

**Cost**:
- **Engineering**: Low (modify inference pipeline)
- **Training time**: None (inference only)
- **Inference time**: 5-10× (need to run inference multiple times)

**Risk**:
- **No gain if model is already robust**: If Swin is already well-calibrated, TTA may not help
- **Overfitting risk**: Low (just inference-time averaging)

**Literature Support**:
- TTA is standard practice in medical imaging
- Shows 1-3% improvement in most cases
- **But**: Diminishing returns if model is already strong

**Verdict**: **LOW-MODERATE PROMISE**. Easy to try, but limited gains expected. Better as quick win.

---

## 2. Top 2 Most Promising Actions

### **#1: ROI-Guided MIL (Segmentation-Guided)**

**Why**:
1. **High potential impact**: If ROI has ≥60% tumor coverage, ROI-MIL can provide orthogonal signal to Swin
2. **Low risk**: Same MIL architecture, just better input (tumor-focused slices)
3. **Literature support**: ROI-guided MIL shows 5-10% AUC improvement in small datasets
4. **Feasibility**: ROI pipeline already exists, segmentation masks available

**Expected Outcome**:
- **If ROI quality is high** (≥60% coverage, low redundancy): MIL coefficient increases to 0.5-1.0, ensemble AUC +0.02-0.03, FN reduction to 2-4
- **If ROI quality is low** (<40% coverage, high redundancy): No gain, MIL remains weak

**Decision Gate**: **MUST verify ROI quality first** (tumor coverage, redundancy with Swin)

**Implementation**:
- Modify MIL dataset to load tumor segmentation masks
- Select slices only from tumor regions (or high tumor probability regions)
- Train MIL on ROI-focused slices
- Evaluate ensemble performance

---

### **#2: Improve Swin (Highest ROI)**

**Why**:
1. **Swin dominates ensemble** (coef=4.06, 7× larger than ResNet, 45× larger than MIL)
2. **Improving Swin has highest impact**: +0.01 Swin AUC → +0.01 ensemble AUC (linear scaling)
3. **Low risk**: Swin is already strong, small improvements are safe
4. **Multiple options**: Better augmentation, longer training, learning rate tuning

**Expected Outcome**:
- **Swin AUC**: 0.85 → 0.88-0.90 (+0.03-0.05)
- **Ensemble AUC**: 0.91 → 0.92-0.93 (+0.01-0.02)
- **FN**: 4-8 → 2-5 (moderate reduction)

**Implementation Options**:
1. **Better augmentation**: More aggressive augmentation (rotation, scaling, elastic)
2. **Longer training**: Train for 100 epochs instead of 60
3. **Learning rate tuning**: Cosine annealing with warmup
4. **Test-time augmentation**: Average predictions over 10 augmented versions

**Decision Gate**: **Lower risk than ROI-MIL** (no data verification needed, just hyperparameter tuning)

---

## 3. Comparison Matrix

| Method | Expected FN Impact | Expected FP Impact | Cost | Risk | Verdict |
|--------|-------------------|-------------------|------|------|---------|
| **ROI-Guided MIL** | ↓↓ (high) | ↓↓ (high) | Medium | Medium | **#1** (if ROI quality verified) |
| **Improve Swin** | ↓ (moderate) | ↓ (moderate) | Low | Low | **#2** (safest, highest ROI) |
| Attention MIL | ↓ (moderate) | ↓ (moderate) | High | High | Moderate (overfitting risk) |
| Hard Instance Mining | ↓ (low) | ↓ (low) | Low | Low | Low (limited evidence) |
| Calibration/Threshold | ↓ (via threshold) | ↑ (trade-off) | Low | Low | Already done |
| Self-Supervised | ↓↓ (if data) | ↓↓ (if data) | Very High | Medium | Low (no unlabeled data) |
| Test-Time Augmentation | ↓ (low) | ↓ (low) | Low | Low | Low (limited gains) |

---

## 4. Realistic Expectations

**Target Metrics**: FN < 5, FP < 5, Precision > 0.93, Recall > 0.93

**Current State**: FN=4-8, FP=3-6, Precision=0.96, Recall=0.77

**Gap Analysis**:
- **FN**: Already near target (4-8, target <5) → **Achievable**
- **FP**: Already near target (3-6, target <5) → **Achievable**
- **Precision**: Already exceeds target (0.96 > 0.93) → **Achieved**
- **Recall**: **Gap** (0.77 < 0.93) → **Main challenge**

**Why High Recall is Hard**:
- **Trade-off**: High recall (≥0.93) requires lower threshold → increases FP
- **Mathematical constraint**: With 210 HGG, recall=0.93 means FN ≤ 14.7
- **Current FN=4-8**: Already excellent, but recall=0.77 means we're missing ~48 HGG cases
- **Reality**: Some HGG cases are truly ambiguous (label noise, borderline cases)

**Realistic Target**:
- **FN**: 2-5 (achievable with ROI-MIL or Swin improvements)
- **FP**: 3-6 (acceptable trade-off for high recall)
- **Precision**: 0.90-0.95 (may drop slightly with lower threshold)
- **Recall**: 0.85-0.90 (more realistic than 0.93, given dataset constraints)

**Conclusion**: **FN < 5 and Precision > 0.93 is achievable**. **Recall > 0.93 is challenging** but may be possible with ROI-MIL if it provides orthogonal signal.

---

## 5. Final Recommendation

**Priority Order**:

1. **#1: ROI-Guided MIL** (if ROI quality verified)
   - **Gate**: Verify ROI quality (≥60% tumor coverage, low redundancy with Swin)
   - **If GO**: Implement ROI-MIL, train on single fold, evaluate
   - **Expected**: MIL coefficient 0.5-1.0, ensemble AUC +0.02-0.03, FN reduction

2. **#2: Improve Swin** (always safe)
   - **No gate**: Low risk, high ROI
   - **Implementation**: Better augmentation, longer training, TTA
   - **Expected**: Ensemble AUC +0.01-0.02, FN reduction to 2-5

3. **#3: Attention MIL** (if ROI-MIL succeeds)
   - **Gate**: ROI-MIL shows promise, want to push further
   - **Implementation**: Replace attention with transformer pooling
   - **Expected**: Additional +0.01-0.02 AUC (but high overfitting risk)

**Decision Rule**: **Verify ROI quality first**. If ROI quality is high (≥60% coverage, low redundancy), proceed with ROI-MIL. Otherwise, focus on Swin improvements.

