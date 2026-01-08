# MIL Performance Analysis & Improvement Strategy

## Executive Summary

Current performance (5-fold CV): **Accuracy ≈ 0.83, F1 ≈ 0.89, Recall ≈ 0.95, AUC ≈ 0.86**
Target (from literature): **Accuracy ≈ 0.92, AUC ≈ 0.98**

**Performance Gap**: ~9% accuracy, ~12% AUC difference

---

## 1️⃣ Root Cause Analysis

### ✅ VALID STRUCTURAL CAUSES (High Confidence)

#### **1.1 Single Modality Limitation** (Impact: HIGH)
- **Current**: FLAIR only
- **Literature**: Multi-modality (T1, T1ce, T2, FLAIR) or FLAIR+T1ce
- **Why it matters**: 
  - FLAIR excels at edema detection but T1ce is critical for enhancing tumor boundaries
  - Multi-modality fusion provides complementary information (T1ce: vascularity, FLAIR: edema)
  - Expected gain: +3-5% AUC, +2-4% accuracy

#### **1.2 Limited Instance Encoder Capacity** (Impact: MEDIUM-HIGH)
- **Current**: ResNet18 (11.56M params)
- **Literature**: ResNet34/50, EfficientNet-B0/B1, or custom architectures
- **Why it matters**:
  - ResNet18 may underfit complex slice-level patterns
  - Deeper encoders capture more hierarchical features (edges → textures → structures)
  - Expected gain: +2-4% AUC, +1-3% accuracy
  - Trade-off: 2-3x compute, but manageable with top-k selection

#### **1.3 Fixed Top-k Selection Strategy** (Impact: MEDIUM)
- **Current**: Fixed top-16 slices (entropy-based)
- **Limitation**: 
  - Patient volumes vary in tumor size/location (some need 8 slices, others need 24)
  - Fixed k may include noise or miss critical slices
  - No adaptivity to bag difficulty
- **Why it matters**: 
  - Adaptive k could preserve all informative slices
  - Expected gain: +1-2% AUC, +0.5-1% accuracy

#### **1.4 MIL Aggregation Limitations** (Impact: MEDIUM)
- **Current**: Max-pooling + Gated Attention (dual-stream)
- **Limitations**:
  - Max-pooling is non-learnable and loses spatial relationships
  - Gated attention may not enforce sparsity (attending to all slices weakly)
  - No interaction between streams until final fusion
- **Potential**: 
  - Attention regularization (sparsity penalty)
  - Cross-stream attention
  - Instance-level contrastive learning
  - Expected gain: +1-2% AUC (architectural refinement)

#### **1.5 Class Imbalance Handling Suboptimal** (Impact: LOW-MEDIUM)
- **Current**: LDAM + DRW + WeightedRandomSampler (default enabled)
- **Observation**: Recall=0.95 but Precision lower suggests over-prediction of HGG
- **Potential issue**: 
  - LDAM margins may be too aggressive or not tuned per-fold
  - DRW start epoch (15) might be too late
  - WeightedRandomSampler + LDAM could double-penalize majority class
- **Expected gain**: +0.5-1% F1 if tuned properly

#### **1.6 Optimization Hyperparameters** (Impact: LOW)
- **Current**: LR=3e-4, cosine scheduler, max_m=0.5, s=30
- **Potential**: 
  - Learning rate might be conservative (could try warmup + 5e-4 peak)
  - LDAM scaling factor s=30 is standard but could be tuned
  - Expected gain: +0.5-1% with tuning

### ❌ INVALID CAUSES (Exclude from Analysis)

- **Slice-level evaluation**: Literature likely uses patient-level (same as ours) — not a cause
- **Data leakage**: Our pipeline is clean (patient-level splits, entropy computed on train-only)
- **Different dataset**: Both use BraTS2018 — same data distribution expected

### 📊 **Summary Table: Valid Causes Ranked by Expected Impact**

| Cause | Expected AUC Gain | Expected Acc Gain | Effort | Risk |
|-------|------------------|-------------------|--------|------|
| Multi-modality fusion | +3-5% | +2-4% | Medium | Low |
| Stronger encoder (ResNet34) | +2-4% | +1-3% | Low | Low |
| Adaptive top-k | +1-2% | +0.5-1% | Medium | Low |
| MIL aggregation improvements | +1-2% | +0.5-1% | High | Medium |
| Class imbalance tuning | +0.5-1% | +0.5-1% | Low | Low |
| Hyperparameter tuning | +0.5-1% | +0.5-1% | Low | Low |

**Total Potential**: +8-15% AUC, +5-11% accuracy (theoretical upper bound)

---

## 2️⃣ High-Impact Improvements (Ranked by Expected Gain)

### **🥇 Priority 1: Multi-Modality MIL** (Expected: +3-5% AUC)

**Strategy**: 
- Stack FLAIR + T1ce channels (2-channel input) OR
- Early fusion: concatenate features from separate encoders

**Medical Rationale**:
- FLAIR: Best for edema and non-enhancing tumor
- T1ce: Best for enhancing tumor boundaries (critical for HGG/LGG distinction)
- Combination: Complementary information → better discrimination

**Implementation**:
```
Option A (Simple): 2-channel input
  - Modify InstanceEncoder: in_channels=2
  - Stack FLAIR and T1ce slices: (N_slices, 2, H, W)
  - ResNet18 first conv: 2→64 (pretrained weights duplicated or reinit)

Option B (Better): Dual-branch feature fusion
  - Two ResNet18 encoders (FLAIR branch, T1ce branch)
  - Feature concatenation: [feat_flair; feat_t1ce] → (N_slices, 1024)
  - Continue with attention/pooling as before
```

**Expected Effects**:
- **Recall**: +1-2% (better HGG detection from T1ce)
- **Precision**: +1-2% (fewer false positives from complementary info)
- **AUC**: +3-5% (better separability in feature space)
- **Accuracy**: +2-4%

**Risk**: Low (no leakage, standard approach)
**Compute**: 1.5-2x (dual modality loading + optional dual encoder)

---

### **🥈 Priority 2: Stronger Instance Encoder** (Expected: +2-4% AUC)

**Strategy**: 
- Replace ResNet18 with ResNet34 or EfficientNet-B0
- Keep pretrained ImageNet weights (adaptation for 1-channel)

**Medical Rationale**:
- Deeper networks capture more complex slice-level patterns
- ResNet34: 21M params vs 11M (still manageable)
- EfficientNet-B0: Similar params, better efficiency

**Expected Effects**:
- **AUC**: +2-4% (richer feature representations)
- **Accuracy**: +1-3%
- **Recall/Precision**: Balanced improvement

**Risk**: Low (straightforward swap)
**Compute**: 2x training time (but inference still fast with top-k)

**Implementation**:
```python
# In models/dual_stream_mil/model.py
from torchvision.models import resnet34, ResNet34_Weights

class InstanceEncoder(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        # Replace resnet18 with resnet34
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
            self.resnet = resnet34(weights=weights)
        # ... rest remains same
```

---

### **🥉 Priority 3: Adaptive Top-k Selection** (Expected: +1-2% AUC)

**Strategy**: 
- Patient-specific k based on entropy distribution
- Use entropy variance or percentile-based threshold
- Alternative: Learn k via attention (harder, research-grade)

**Medical Rationale**:
- Tumor size/location varies → optimal k varies
- Small tumors: fewer informative slices
- Large/complex tumors: more informative slices needed

**Expected Effects**:
- **AUC**: +1-2% (better instance selection)
- **Accuracy**: +0.5-1%
- **Recall**: Slight improvement (captures all informative slices)

**Risk**: Low (still deterministic, no leakage)
**Compute**: Negligible (entropy already computed)

**Implementation**:
```python
# In utils/entropy_analysis.py or MILDataset
def adaptive_top_k(entropy_scores, base_k=16, method='percentile'):
    if method == 'percentile':
        threshold = np.percentile(entropy_scores, 100 - base_k/len(entropy_scores)*100)
        selected = [i for i, e in enumerate(entropy_scores) if e >= threshold]
        return selected[:base_k*2]  # Cap at 2x base_k
    # ... other methods
```

---

### **4️⃣ Attention Regularization & Sparsity** (Expected: +1-2% AUC)

**Strategy**:
- Add L1/L2 penalty on attention weights to encourage sparsity
- Top-k attention selection (keep only top-k attended instances)
- Cross-entropy loss on attention distribution (encourage focus)

**Medical Rationale**:
- Current attention may be too diffuse (attending to all slices weakly)
- Sparse attention → model focuses on truly discriminative slices
- Improves interpretability (which slices matter)

**Expected Effects**:
- **Precision**: +0.5-1% (less noise from irrelevant slices)
- **AUC**: +1-2%
- **Accuracy**: +0.5-1%

**Risk**: Medium (hyperparameter tuning needed)
**Compute**: Negligible

---

### **5️⃣ Dual-Threshold Optimization** (Expected: +0.5-1% F1)

**Strategy**:
- Primary threshold: Maximize HGG recall (≥0.95) with constraint
- Secondary threshold: Balanced F1 for reporting
- Report both operating points

**Medical Rationale**:
- HGG misdiagnosis is more critical (false negatives)
- Allows reporting optimal F1 while maintaining high recall

**Expected Effects**:
- **Recall**: Maintained at ≥0.95 (clinical constraint)
- **F1**: +0.5-1% (better precision-recall trade-off)

**Risk**: None (evaluation-only)
**Compute**: Negligible

---

### **6️⃣ Curriculum MIL (Research-Grade)** (Expected: +1-2% AUC)

**Strategy**:
- Train on "easy" bags first (high entropy variance, clear tumors)
- Gradually introduce "hard" bags (low contrast, ambiguous cases)
- Can be implemented via sampling or loss weighting

**Medical Rationale**:
- Easier learning progression (simple → complex patterns)
- Prevents early overfitting to easy cases

**Risk**: Medium (implementation complexity, hyperparameter tuning)
**Compute**: Same (just sampling strategy)

---

### **7️⃣ Self-Supervised Pretraining (Research-Grade)** (Expected: +1-3% AUC)

**Strategy**:
- Pretrain instance encoder on unlabeled BraTS slices (contrastive/rotation)
- Fine-tune on MIL task

**Medical Rationale**:
- Better slice-level feature representations
- Leverages all available data (not just labeled patients)

**Risk**: High (requires unlabeled data, complex setup)
**Compute**: High (pretraining phase)

---

## 3️⃣ Metric Strategy Optimization

### **Medically-Aware Evaluation Framework**

#### **Primary Operating Point: HGG-Recall-First**
```
Constraint: HGG Recall ≥ 0.95 (clinical requirement)
Optimize: F1 under this constraint
Threshold selection: Find threshold that maximizes F1 while ensuring Recall ≥ 0.95
```

#### **Secondary Operating Point: Balanced F1**
```
For reporting/publication: Use threshold that maximizes F1 (current approach)
This provides fair comparison with literature
```

#### **Reporting Strategy**
```
In paper/thesis, report:
1. Primary: HGG Recall ≥ 0.95, F1 = X.XX, Precision = X.XX (clinical operating point)
2. Secondary: Optimal F1 threshold, all metrics (research operating point)
3. AUC-ROC: Always reported (threshold-independent, ranking quality)
4. Confusion matrices: Both thresholds
```

**Implementation**:
```python
# In find_optimal_threshold or separate function
def find_medical_threshold(y_true, y_prob, min_recall=0.95, pos_label=1):
    """Find threshold that maximizes F1 subject to Recall ≥ min_recall"""
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1 = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        if rec >= min_recall:
            f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    return best_threshold
```

---

## 4️⃣ Concrete Action Plan (Prioritized Roadmap)

### **Phase 1: Quick Wins (1-2 days, Expected: +2-3% AUC)**

#### Step 1.1: Multi-Modality Input (FLAIR + T1ce)
- **Action**: Modify `InstanceEncoder` to accept 2-channel input
- **Changes**: 
  - `utils/mil_dataset.py`: Stack FLAIR and T1ce slices
  - `models/dual_stream_mil/model.py`: `in_channels=2`, adapt first conv layer
  - Training: Ensure both modalities exist in splits
- **Effort**: 2-3 hours
- **Expected gain**: +3-5% AUC

#### Step 1.2: Stronger Encoder (ResNet34)
- **Action**: Swap ResNet18 → ResNet34
- **Changes**: One-line change in `model.py`
- **Effort**: 30 minutes
- **Expected gain**: +2-4% AUC

**Combined Phase 1 Expected**: +5-9% AUC improvement

---

### **Phase 2: Architecture Refinements (2-3 days, Expected: +1-2% AUC)**

#### Step 2.1: Attention Sparsity Regularization
- **Action**: Add L1 penalty on attention weights
- **Changes**: 
  - Modify `GatedAttention` forward to compute attention penalty
  - Add to loss: `loss = ldam_loss + lambda_sparse * attention_l1`
  - Tune `lambda_sparse` (start with 0.01)
- **Effort**: 4-6 hours
- **Expected gain**: +1-2% AUC

#### Step 2.2: Adaptive Top-k (Optional)
- **Action**: Implement percentile-based adaptive k
- **Changes**: `utils/mil_dataset.py` or entropy analysis
- **Effort**: 2-3 hours
- **Expected gain**: +1-2% AUC

---

### **Phase 3: Optimization Tuning (1 day, Expected: +0.5-1% AUC)**

#### Step 3.1: Hyperparameter Grid Search
- **Action**: Tune LR, LDAM params, DRW timing
- **Candidates**:
  - LR: [2e-4, 3e-4, 5e-4] with warmup
  - max_m: [0.3, 0.5, 0.7]
  - s: [20, 30, 40]
  - drw_start_epoch: [10, 15, 20]
- **Effort**: 1 day (automated grid search)
- **Expected gain**: +0.5-1% AUC

#### Step 3.2: Dual-Threshold Reporting
- **Action**: Implement medical threshold finder
- **Changes**: Add function to `train_mil.py`, report both thresholds
- **Effort**: 1 hour
- **Expected gain**: Better clinical applicability (no metric gain)

---

### **Phase 4: Research-Grade (Optional, 1-2 weeks, Expected: +1-3% AUC)**

#### Step 4.1: Curriculum MIL
- **Action**: Implement difficulty-based sampling
- **Effort**: 2-3 days
- **Expected gain**: +1-2% AUC

#### Step 4.2: Self-Supervised Pretraining
- **Action**: Pretrain on unlabeled slices
- **Effort**: 1 week
- **Expected gain**: +1-3% AUC (if unlabeled data available)

---

## 5️⃣ Code-Level Suggestions

### **5.1 Multi-Modality Input (Priority 1)**

**File**: `utils/mil_dataset.py`
```python
# In __getitem__, modify slice extraction:
if self.use_entropy:
    # ... get top_k_indices as before
    
    # Extract slices for both modalities
    slices_flair = [volume_array_flair[i, :, :] for i in slice_indices]
    slices_t1ce = [volume_array_t1ce[i, :, :] for i in slice_indices]
    
    # Stack channels: (num_slices, 2, H, W)
    slice_bag = np.stack([
        np.stack([s_flair, s_t1ce], axis=0) 
        for s_flair, s_t1ce in zip(slices_flair, slices_t1ce)
    ], axis=0)
```

**File**: `models/dual_stream_mil/model.py`
```python
class InstanceEncoder(nn.Module):
    def __init__(self, in_channels=2, pretrained=True):  # Changed from 1
        # ... load ResNet18/34
        self.resnet.conv1 = nn.Conv2d(
            in_channels,  # 2 instead of 1
            self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=False
        )
        # Initialize: average ImageNet RGB weights or Xavier
        if pretrained:
            # Average pretrained RGB weights: conv1.weight = mean(pretrained_conv1.weight, dim=1)
            with torch.no_grad():
                pretrained_conv1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).conv1.weight
                self.resnet.conv1.weight[:, 0] = pretrained_conv1.mean(dim=1)  # Channel 0 (FLAIR)
                self.resnet.conv1.weight[:, 1] = pretrained_conv1.mean(dim=1)  # Channel 1 (T1ce)
```

---

### **5.2 ResNet34 Encoder (Priority 2)**

**File**: `models/dual_stream_mil/model.py`
```python
from torchvision.models import resnet34, ResNet34_Weights

class InstanceEncoder(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1  # Changed from ResNet18_Weights
            self.resnet = resnet34(weights=weights)   # Changed from resnet18
        else:
            self.resnet = resnet34(weights=None)
        # ... rest remains same
```

---

### **5.3 Attention Sparsity (Priority 3)**

**File**: `models/dual_stream_mil/model.py`
```python
class GatedAttention(nn.Module):
    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ... existing attention computation ...
        A = F.softmax(A, dim=1)  # (K, num_instances)
        
        M = torch.mm(A, H)  # (K, L)
        
        # Compute sparsity penalty (L1 of attention)
        attention_l1 = torch.sum(torch.abs(A))
        
        return M, A, attention_l1  # Return penalty too
```

**File**: `scripts/training/train_mil.py`
```python
# In training loop:
logits, attention_weights, attention_l1 = model(bags)  # Unpack attention_l1
loss = loss_fn(logits, labels, epoch)
sparsity_penalty = 0.01 * attention_l1.mean()  # lambda_sparse = 0.01
loss = loss + sparsity_penalty
```

---

### **5.4 Medical Threshold Finder**

**File**: `scripts/training/train_mil.py`
```python
def find_medical_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.95, pos_label: int = 1) -> float:
    """Find threshold that maximizes F1 subject to Recall >= min_recall"""
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1 = -1.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rec = recall_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
        if rec >= min_recall:
            f1 = f1_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    return float(best_threshold)
```

---

## Expected Final Performance (After Phase 1+2)

**Conservative Estimate**:
- Accuracy: 0.83 → **0.88-0.91** (+5-8%)
- AUC: 0.86 → **0.91-0.95** (+5-9%)
- F1: 0.89 → **0.92-0.94** (+3-5%)
- Recall: 0.95 → **0.96-0.98** (+1-3%)

**Optimistic Estimate** (with all phases):
- Accuracy: **0.90-0.93**
- AUC: **0.93-0.97**
- F1: **0.93-0.95**
- Recall: **0.96-0.98**

This approaches the reported upper bounds (0.92 accuracy, 0.98 AUC) while maintaining robustness and no data leakage.

---

## Summary: Top 3 Actions to Execute Immediately

1. **Multi-Modality (FLAIR + T1ce)**: +3-5% AUC, 2-3 hours effort
2. **ResNet34 Encoder**: +2-4% AUC, 30 minutes effort
3. **Attention Sparsity**: +1-2% AUC, 4-6 hours effort

**Total Expected Gain**: +6-11% AUC improvement
**Total Effort**: 1-2 days
**Risk**: Low (all standard, well-tested approaches)

