# Swin-1 vs Proposed Swin-2: Comparison Table

## Current Swin (Swin-1) Characteristics

| Aspect | Current Swin-1 | Analysis |
|--------|----------------|----------|
| **Architecture** | | |
| Input Size | 128×128×128 | Standard resolution |
| Patch Size | 2 | Coarse-grained patches |
| Window Size | 7 | Global attention |
| Feature Size | 48 | Memory-efficient |
| Depths | [2, 2, 2, 2] | Shallow network |
| **Performance** | | |
| AUC | 0.9065 | Strong ranking quality |
| Precision | 0.9874 | Excellent (only 2 FP) |
| Recall | 0.7476 | **Main weakness (53 FN)** |
| FN Count | 53 | **Primary issue** |
| FP Count | 2 | Excellent |
| **Error Patterns** | | |
| FN Mean Prob | 0.1386 ± 0.1375 | Low confidence on missed HGG |
| FP Mean Prob | 0.9852 ± 0.0202 | Overconfident on LGG |
| FN Distribution | Fold 0: 10, Fold 1: 7, Fold 2: 9, Fold 3: 9, Fold 4: 18 | Inconsistent (std=3.83) |
| **Strengths** | | |
| High-Confidence Cases | 73.68% | Strong on clear patterns |
| LGG Accuracy | 97.33% | Excellent LGG detection |
| HGG Accuracy | 74.76% | **Weak on subtle HGG** |
| **Redundancy** | | |
| Correlation with ResNet | 0.2535 | Low redundancy |
| Correlation with MIL | 0.1470 | Very low redundancy |
| Unique Correct Cases | 66 | Strong complementarity |

---

## Proposed Swin-2 Characteristics

| Aspect | Proposed Swin-2 | Rationale |
|--------|-----------------|-----------|
| **Architecture** | | |
| Input Size | **160×160×160** | Higher resolution for fine details |
| Patch Size | **1** (instead of 2) | Capture small tumor features |
| Window Size | **4** (instead of 7) | Local attention for subtle patterns |
| Feature Size | **64** (instead of 48) | More capacity for complex patterns |
| Depths | **[3, 3, 3, 3]** (instead of [2,2,2,2]) | Deeper representation |
| **Training Strategy** | | |
| Loss Function | **Focal Loss** (γ=2.0, α=0.25) | Focus on hard examples (FN cases) |
| Sampling | **Hard example mining** | Oversample FN cases from Swin-1 |
| Class Weights | **pos_weight=2.0-3.0** | Penalize FN more heavily |
| Augmentation | **Stronger** (zoom, rotation) | Simulate small/diffuse tumors |
| Regularization | **Dropout 0.3** | Prevent overfitting |
| **Expected Performance** | | |
| Target FN | **< 10** (from 53) | 60% reduction |
| Target FP | **< 10** (from 2) | Acceptable increase |
| Target Recall | **> 0.90** (from 0.75) | Significant improvement |
| Target Precision | **> 0.90** (from 0.99) | Slight decrease acceptable |
| Target AUC | **> 0.85** | Maintain strong ranking |
| **Complementarity** | | |
| Correlation with Swin-1 | **< 0.7** | Ensure non-redundancy |
| Unique Signal | **Small tumors, diffuse patterns** | Different from Swin-1 |

---

## Key Differences Summary

| Dimension | Swin-1 | Swin-2 | Impact |
|-----------|--------|--------|--------|
| **Spatial Resolution** | 128³ | 160³ | +25% resolution for fine details |
| **Patch Granularity** | Coarse (2) | Fine (1) | 2× more patches, better for small tumors |
| **Attention Scope** | Global (window=7) | Local (window=4) | Better for subtle local patterns |
| **Model Capacity** | Small (48, [2,2,2,2]) | Large (64, [3,3,3,3]) | More parameters, risk of overfitting |
| **Training Focus** | Balanced | FN-focused | Explicitly targets missed HGG cases |
| **Loss Function** | CrossEntropy | Focal Loss | Hard example mining |

---

## Expected Ensemble Interaction

| Scenario | Swin-1 | Swin-2 | Ensemble Benefit |
|----------|--------|--------|------------------|
| **Clear HGG (large tumor)** | ✓ High confidence | ✓ High confidence | Redundant but stable |
| **Subtle HGG (small/diffuse)** | ✗ Low confidence (FN) | ✓ Should catch | **Complementary** |
| **Clear LGG** | ✓ High confidence | ✓ High confidence | Redundant but stable |
| **Ambiguous LGG** | ✓ Low confidence | ? Uncertain | May help or hurt |

**Key Insight:** Swin-2 should excel where Swin-1 fails (subtle HGG), providing complementary signal.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Overfitting** | HIGH | CRITICAL | Strong regularization, early stopping, monitor train/val gap |
| **Redundancy** | MODERATE | HIGH | Monitor correlation, ensure < 0.7 |
| **Computational Cost** | HIGH | MODERATE | Higher resolution + deeper = 3-4× slower |
| **Data Insufficiency** | HIGH | CRITICAL | 285 samples may not support larger model |
| **FN Reduction Failure** | MODERATE | HIGH | Hard example mining may not be enough |

---

## GO/NO-GO Decision Matrix

| Criterion | Status | Decision Impact |
|-----------|--------|----------------|
| **Theoretical Feasibility** | ✓ Possible (min recall 0.9762) | Supports GO |
| **Practical Likelihood** | ⚠️ Moderate (requires near-perfect model) | Supports CONDITIONAL |
| **Improvement Needed** | ❌ Large (48 FN reduction = 90%) | Supports NO-GO |
| **Data Sufficiency** | ❌ Questionable (285 samples) | Supports NO-GO |
| **Complementarity Potential** | ✓ High (low correlation with Swin-1) | Supports GO |

**Final Decision: CONDITIONAL_GO** (revised from NO_GO)

**Reasoning:**
- Theoretically possible but extremely challenging
- High risk of overfitting with larger model on small dataset
- However, complementarity potential is high (Swin-1 has 66 unique correct cases)
- **Recommendation:** Start with conservative changes (Priority 1 only), validate on single fold first

---

## Prioritized Implementation Plan

### Phase 1: Conservative Approach (Recommended)
1. **Smaller patch size (1)** - Highest ROI, lowest risk
2. **Focal Loss** - Explicitly targets FN
3. **Hard example mining** - Oversample Swin-1 FN cases
4. **Keep resolution 128³** - Avoid overfitting risk
5. **Keep depths [2,2,2,2]** - Avoid overfitting risk

**Expected Outcome:** FN reduction from 53 to 30-40 (moderate improvement)

### Phase 2: Aggressive Approach (If Phase 1 succeeds)
1. **Higher resolution (160³)** - Better spatial detail
2. **Deeper network ([3,3,3,3])** - More capacity
3. **Larger feature size (64)** - More representation

**Expected Outcome:** FN reduction from 53 to 15-25 (significant improvement)

### Phase 3: Optimistic Approach (If Phase 2 succeeds)
1. **Multi-view ensemble** - Axial + Sagittal + Coronal
2. **Tumor-focused cropping** - If segmentation available

**Expected Outcome:** FN reduction from 53 to < 10 (target achieved)

---

## Validation Criteria

**Must achieve ALL of the following to proceed to next phase:**

1. **Correlation with Swin-1 < 0.7** - Ensures complementarity
2. **FN reduction ≥ 30%** - Meaningful improvement
3. **AUC > 0.85** - Maintains ranking quality
4. **Train/Val AUC gap < 0.10** - No severe overfitting
5. **FN std across folds < 5** - Stable performance

**If any criterion fails → STOP and reconsider approach**

---

*Analysis Date: 2026-02-10*  
*Current Swin Performance: FN=53, FP=2, Recall=0.75, Precision=0.99, AUC=0.91*  
*Target Performance: FN<5, FP<5, Recall>0.95, Precision>0.95*

