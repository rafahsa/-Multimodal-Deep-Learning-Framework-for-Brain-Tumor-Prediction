# Dual-Stream Multiple Instance Learning (MIL) for Brain Tumor Classification

## Design Document: Third Model Architecture

**Model**: Dual-Stream MIL Classifier  
**Dataset**: BraTS 2018 (285 patients: 210 HGG, 75 LGG)  
**Task**: Binary classification (HGG vs LGG) at patient (bag) level  
**Complementary to**: ResNet50-3D (3D CNN) and SwinUNETR-3D (Transformer)

---

## 1. Design Rationale and Motivation

### 1.1 Why MIL for Brain Tumor Classification?

**Intra-Patient Heterogeneity**:
- Brain tumors exhibit spatial heterogeneity: different slices/regions show varying degrees of enhancement, edema, and necrosis
- A single critical slice (e.g., maximum enhancement) may contain the strongest diagnostic signal
- Surrounding slices provide contextual information (tumor boundaries, peritumoral edema, mass effect)

**Complementarity to Existing Models**:
- **ResNet50-3D**: Processes entire 3D volume holistically, may average out critical regions
- **SwinUNETR-3D**: Transformer captures long-range dependencies but treats all spatial locations equally
- **Dual-Stream MIL**: Explicitly identifies critical instances and aggregates context, providing interpretable, instance-aware predictions

**MIL Advantages**:
- **Interpretability**: Can identify which slices are most informative (critical instance)
- **Robustness**: Less sensitive to irrelevant slices (background, artifacts)
- **Efficiency**: Can process variable numbers of slices per patient
- **Ensemble Diversity**: Different inductive bias from 3D models

### 1.2 Why Dual-Stream Design?

**Stream 1 (Critical Instance)**: 
- Identifies the slice with strongest tumor evidence
- Represents the "smoking gun" diagnostic signal
- Analogous to radiologist focusing on most enhancing slice

**Stream 2 (Contextual Aggregation)**:
- Aggregates information from all slices with attention conditioned on critical instance
- Captures supportive evidence (tumor boundaries, mass effect, peritumoral changes)
- Prevents over-reliance on single slice

**Dual-Stream Fusion**:
- Combines critical signal with contextual support
- More robust than single-stream (max pooling or attention alone)
- Enables interpretability: can visualize critical slice and attention weights

---

## 2. Architecture Design

### 2.1 Bag Definition

**Patient = Bag, Slice = Instance**

- **Instance Type**: 2D axial slices extracted from 3D multi-modal volumes
- **Instance Format**: Multi-modal 2D slice `(4, H, W)` where channels are [T1, T1ce, T2, FLAIR]
- **Bag Size**: Variable (typically 128 slices per patient for 128×128×128 volumes)
- **Fixed Bag Size Strategy**: Sample or pad to fixed size (e.g., 64 slices) for batch processing
  - **Sampling**: Random sampling or entropy-based selection of informative slices
  - **Padding**: Zero-padding with mask to handle variable sizes

**Rationale for 2D Slices**:
- **Memory Efficiency**: 2D processing is much more memory-efficient than 3D patches
- **Interpretability**: Easy to visualize which slice is critical
- **Proven in Medical MIL**: 2D slice-based MIL is well-established in medical imaging
- **Computational Feasibility**: Can process many slices per patient efficiently

**Alternative Considered (3D Patches)**: Rejected due to:
- Higher memory requirements
- More complex aggregation
- Less interpretable (which patch is critical?)

### 2.2 Instance Encoder

**Architecture**: Lightweight 2D CNN (ResNet18 or EfficientNet-B0)

**Input**: Multi-modal 2D slice `(4, 224, 224)` or `(4, 128, 128)`
- Channels: T1, T1ce, T2, FLAIR (early fusion at slice level)
- Spatial size: Resized to standard 2D CNN input size

**Encoder Design**:
```
Option A: ResNet18 (Recommended)
- First conv: 4 → 64 channels (adapts to multi-modal input)
- Standard ResNet18 backbone
- Output: 512-dim feature vector per slice

Option B: EfficientNet-B0
- First conv: 4 → 32 channels
- EfficientNet-B0 backbone
- Output: 1280-dim feature vector per slice
```

**Rationale**:
- **Lightweight**: Must process many slices per patient efficiently
- **Multi-modal**: First layer adapts to 4-channel input (no pretrained weights)
- **Proven**: ResNet18/EfficientNet are standard, stable architectures
- **Feature Dimension**: 512 (ResNet18) or 1280 (EfficientNet) provides rich representation

**Training**: From scratch (no ImageNet pretraining due to multi-modal input)

### 2.3 Dual-Stream Architecture

#### Stream 1: Critical Instance Identification

**Goal**: Identify the most informative slice in the bag

**Architecture**:
```
Instance Features (N slices × D features)
  → Instance Scoring Network
  → Instance Scores (N scores)
  → Argmax or Softmax → Critical Instance Index
  → Extract Critical Instance Feature
```

**Scoring Network**:
- Small MLP: `D → 128 → 1` (sigmoid output)
- Produces importance score for each instance
- Trained end-to-end with bag-level supervision

**Critical Instance Selection**:
- **Hard Selection**: Argmax (select highest-scoring instance)
- **Soft Selection**: Weighted combination using softmax(score)
- **Hybrid**: Use hard selection for Stream 1, soft for Stream 2

**Output**: Critical instance feature vector `(D,)`

#### Stream 2: Contextual Attention Aggregation

**Goal**: Aggregate information from all slices with attention conditioned on critical instance

**Architecture**:
```
All Instance Features (N × D)
Critical Instance Feature (D)
  → Similarity Computation (cosine or learned)
  → Attention Weights (N weights)
  → Weighted Aggregation
  → Contextual Feature Vector (D)
```

**Attention Mechanism**:
- **Query**: Critical instance feature
- **Keys & Values**: All instance features
- **Similarity**: Cosine similarity or learned linear projection
- **Attention Weights**: Softmax(similarity) or Gated Attention

**Gated Attention (Recommended)**:
```
Attention_i = sigmoid(W_q * critical_feat + W_k * instance_i + b)
Context = sum(Attention_i * instance_i) / sum(Attention_i)
```

**Rationale**:
- **Conditioned on Critical**: Attention is explicitly conditioned on critical instance
- **Contextual Support**: Captures supportive evidence from other slices
- **Interpretable**: Attention weights show which slices support the critical instance

#### Stream Fusion

**Goal**: Combine critical instance signal with contextual aggregation

**Fusion Strategy**:
```
Option A: Concatenation + MLP
  fused = MLP([critical_feat, context_feat])
  
Option B: Weighted Sum
  fused = α * critical_feat + (1-α) * context_feat
  (α learned or fixed)

Option C: Cross-Attention
  fused = CrossAttention(critical_feat, context_feat)
```

**Recommended**: Concatenation + MLP (simple, effective, interpretable)

**Final Classification Head**:
```
Fused Feature (2D)
  → Dropout(0.4)
  → Linear(2D, 128) → ReLU
  → Dropout(0.4)
  → Linear(128, 2)
  → Binary Classification Logits
```

### 2.4 Complete Architecture Flow

```
Patient (Bag of N slices)
  ↓
[Slice 1 (4, H, W), ..., Slice N (4, H, W)]
  ↓ (Instance Encoder - Shared)
[Feat 1 (D,), ..., Feat N (D,)]
  ↓
┌─────────────────────────────────────┐
│  Stream 1: Critical Instance       │
│  Scoring Network → Critical Feat   │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  Stream 2: Contextual Aggregation   │
│  Attention(Critical, All) → Context │
└─────────────────────────────────────┘
  ↓
Fusion: [Critical, Context] → Fused Feature
  ↓
Classification Head → Patient Logits (2)
```

---

## 3. Loss Function Strategy

### 3.1 Analysis of Loss Options

**Context**:
- Moderate class imbalance: HGG ≈ 210, LGG ≈ 75 (2.8:1 ratio)
- MIL setting: Bag-level supervision only
- Training stability: Critical (learned from ResNet50-3D experience)

**Option A: CrossEntropyLoss + WeightedRandomSampler** ✅ **RECOMMENDED**
- **Pros**: 
  - Proven stable in ResNet50-3D and SwinUNETR-3D
  - Simple, well-understood
  - No complex hyperparameters
  - Works well with moderate imbalance
- **Cons**: 
  - May not fully exploit MIL structure
- **Verdict**: Use this for initial implementation (stability > complexity)

**Option B: Focal Loss**
- **Pros**: 
  - Handles class imbalance naturally
  - Focuses on hard examples
  - No need for weighted sampling
- **Cons**: 
  - Additional hyperparameter (γ)
  - May be less stable than CrossEntropy
  - Not specifically designed for MIL
- **Verdict**: Consider if CrossEntropy underperforms

**Option C: LDAM + DRW**
- **Pros**: 
  - Designed for class imbalance
  - Margin-based approach
- **Cons**: 
  - **Rejected based on ResNet50-3D experience**: Caused training instability
  - Complex interaction with MIL aggregation
  - Not recommended given prior failures

**Option D: MIL-Specific Losses**
- **Pros**: 
  - Designed for MIL (e.g., max-margin MIL loss)
- **Cons**: 
  - Less standard, harder to compare
  - May not be necessary if aggregation is well-designed
- **Verdict**: Not needed initially

### 3.2 Final Choice: CrossEntropyLoss + WeightedRandomSampler

**Rationale**:
1. **Stability**: Proven stable in both previous models
2. **Simplicity**: Easier to debug and reproduce
3. **Moderate Imbalance**: 2.8:1 ratio is manageable with weighted sampling
4. **MIL Compatibility**: Bag-level CrossEntropyLoss is standard in MIL literature
5. **Ensemble Consistency**: Same loss strategy enables fair comparison

**Implementation**:
- **Loss**: `nn.CrossEntropyLoss()` (bag-level, patient-level labels)
- **Sampling**: `WeightedRandomSampler` with `inverse_freq` strategy
- **No Label Smoothing**: Keep simple initially

**Future Consideration**: If performance is suboptimal, try Focal Loss with γ=2.0

---

## 4. Training Configuration

### 4.1 Data Processing

**Slice Extraction**:
- Extract all 128 axial slices from each 3D volume
- Each slice: `(4, 128, 128)` [T1, T1ce, T2, FLAIR]
- **Fixed Bag Size**: Sample 64 slices per patient (or pad to 64)
  - **Sampling Strategy**: Random sampling (can be entropy-based later)
  - **Rationale**: Fixed size enables batching, 64 is memory-efficient

**Augmentation** (per slice, consistent across modalities):
- Random horizontal/vertical flip
- Random rotation (±15°)
- Random intensity scaling
- **No spatial augmentation across slices** (maintains slice correspondence)

**Normalization**:
- Per-slice normalization (mean=0, std=1) or per-modality normalization
- Consistent with existing 3D models

### 4.2 Optimization

**Batch Size**: 4-8 patients (bags) per batch
- Each bag contains 64 slices
- Effective instances per batch: 256-512 slices
- Memory-efficient with 2D processing

**Optimizer**: AdamW
- **Learning Rate**: 1e-4 (instance encoder)
- **Classifier LR**: 2e-4 (2× encoder LR)
- **Weight Decay**: 1e-4

**Scheduler**: Cosine annealing with warmup
- Warmup: 10% of epochs (min 5)
- Cosine decay to 0.5× initial LR

**Gradient Clipping**: 1.0 (max norm)

**Mixed Precision**: Enabled (AMP)

**EMA**: Optional (0.995 decay) - test if beneficial

### 4.3 Regularization

- **Dropout**: 0.4 in classification head
- **Weight Decay**: 1e-4
- **Early Stopping**: Patience 10, min epochs 15
- **No additional regularization** (keep simple)

### 4.4 Cross-Validation

- **5-fold CV**: Same splits as ResNet50-3D and SwinUNETR-3D
- **Patient-level splits**: No slice-level leakage
- **Checkpoint Selection**: Best validation AUC (bag-level)
- **Evaluation**: EMA checkpoint if available, else best checkpoint

---

## 5. Implementation Details

### 5.1 Dataset Class

**New Class**: `MILSliceDataset`
- Inherits from `torch.utils.data.Dataset`
- Loads patient-level labels
- Extracts slices from 3D volumes
- Returns: `(bag_of_slices, label, patient_id)`
  - `bag_of_slices`: `(N, 4, H, W)` where N is fixed (e.g., 64)

### 5.2 Model Architecture

**File**: `models/dual_stream_mil.py`

**Key Components**:
1. `InstanceEncoder`: 2D CNN (ResNet18 or EfficientNet-B0)
2. `CriticalInstanceSelector`: Scoring network + selection
3. `ContextualAggregator`: Attention mechanism
4. `DualStreamMIL`: Complete model

### 5.3 Training Script

**File**: `scripts/training/train_dual_stream_mil.py`

**Compatibility**:
- Same argument structure as `train_resnet50_3d.py` and `train_swin_unetr_3d.py`
- Same logging format
- Same metrics structure
- Same checkpoint policy

---

## 6. Interpretability Features

### 6.1 Critical Instance Visualization

**Output**: Index of critical slice for each patient
- Can visualize the critical slice (most informative)
- Compare across patients
- Validate against radiologist intuition

### 6.2 Attention Weight Visualization

**Output**: Attention weights for all slices
- Heatmap showing which slices contribute most to prediction
- Can identify supportive slices beyond critical instance

### 6.3 Complementarity to Other Models

**Grad-CAM Comparison**:
- ResNet50-3D/SwinUNETR-3D: Show 3D spatial attention
- Dual-Stream MIL: Show which slices are critical
- Combined: Full interpretability (spatial + slice-level)

---

## 7. Expected Advantages and Challenges

### 7.1 Advantages

1. **Interpretability**: Clear identification of critical diagnostic slices
2. **Robustness**: Less sensitive to irrelevant slices
3. **Efficiency**: 2D processing is memory-efficient
4. **Complementarity**: Different inductive bias from 3D models
5. **Ensemble Diversity**: Predictions should be meaningfully different

### 7.2 Challenges

1. **Fixed Bag Size**: Sampling/padding may lose information
2. **Slice Selection**: Random sampling may miss critical slices
3. **Memory**: Still need to process many slices per patient
4. **Training Time**: More instances per patient than 3D models

### 7.3 Mitigation Strategies

1. **Entropy-Based Sampling**: Select most informative slices (future improvement)
2. **Gradient Checkpointing**: If memory is an issue
3. **Efficient Aggregation**: Use efficient attention mechanisms
4. **Progressive Training**: Start with fewer slices, increase during training

---

## 8. Success Criteria

### 8.1 Performance Targets

- **AUC**: > 0.85 (competitive with existing models)
- **F1-score**: > 0.70
- **Stability**: Smooth training curves, no instability

### 8.2 Ensemble Readiness

- **Diversity**: Predictions should differ meaningfully from ResNet50-3D and SwinUNETR-3D
- **Correlation**: Moderate correlation (0.6-0.8) with other models (not too similar, not too different)
- **Complementarity**: Should improve ensemble when combined

### 8.3 Interpretability

- **Critical Slices**: Should identify anatomically meaningful slices
- **Attention Weights**: Should highlight tumor regions
- **Visualization**: Should provide clear insights for clinicians

---

## 9. Implementation Plan

### Phase 1: Core Architecture
1. Implement `InstanceEncoder` (ResNet18-based)
2. Implement `CriticalInstanceSelector`
3. Implement `ContextualAggregator`
4. Implement `DualStreamMIL` model

### Phase 2: Dataset and Training
1. Implement `MILSliceDataset`
2. Implement training script
3. Test on single fold

### Phase 3: Optimization
1. Tune hyperparameters
2. Test different aggregation strategies
3. Validate interpretability

### Phase 4: Full Evaluation
1. Train all 5 folds
2. Generate metrics and visualizations
3. Compare with existing models
4. Prepare for ensemble

---

## 10. References and Justification

**MIL in Medical Imaging**:
- Ilse et al. (2018): Attention-based MIL for histopathology
- Li et al. (2021): MIL for medical image classification
- Dual-stream designs in medical MIL literature

**Architecture Choices**:
- ResNet18: Standard, lightweight, proven
- Attention mechanisms: Well-established in MIL
- CrossEntropyLoss: Proven stable in this project

**Complementarity**:
- 3D models: Holistic volume processing
- MIL model: Instance-aware, interpretable processing
- Different inductive biases → ensemble diversity

---

**Document Status**: Design Complete, Ready for Implementation  
**Next Steps**: Implement core architecture and test on single fold

