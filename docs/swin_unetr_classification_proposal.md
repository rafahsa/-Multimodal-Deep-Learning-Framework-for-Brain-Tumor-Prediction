# Swin UNETR-Based Classification Architecture Proposal
## Brain Tumor Grading (HGG vs LGG) - BraTS 2018

**Date**: 2025-01-07  
**Context**: Design proposal for Swin UNETR-based binary classification model  
**Goal**: Stable, reproducible, high-performing architecture compatible with ensemble

---

## Executive Summary

This proposal recommends an **end-to-end classification architecture** using Swin UNETR encoder with a global pooling and classification head. The design prioritizes **stability and reproducibility** over complexity, learning from the ResNet50-3D training experience. The recommended approach uses **CrossEntropyLoss + WeightedRandomSampler** (proven stable) rather than complex loss functions that caused instability.

**Key Decision**: Encoder-only classification (no segmentation pretraining) for simplicity and direct optimization.

---

## 1. Architecture Design

### 1.1 Recommended Approach: End-to-End Classification

**Selected Strategy**: **End-to-End Classification with Swin UNETR Encoder**

**Architecture Components**:
```
Input: (B, 4, 128, 128, 128)  [Multi-modal: T1, T1ce, T2, FLAIR]
    ↓
Swin UNETR Encoder (Transformer-based feature extraction)
    ↓
Global Pooling (Adaptive Average Pooling or Attention-based)
    ↓
Classification Head (FC layers + Dropout)
    ↓
Output: (B, 2)  [Binary classification: LGG vs HGG]
```

**Rationale**:
1. **Simplicity**: Single-stage training, no segmentation labels required
2. **Direct Optimization**: Model learns classification-relevant features end-to-end
3. **Stability**: Fewer moving parts = fewer failure modes
4. **Reproducibility**: Easier to debug and reproduce than multi-stage approaches

### 1.2 Why NOT Two-Stage (Segmentation → Classification)?

**Reasons Against**:
- **Additional Complexity**: Requires segmentation labels, two training phases
- **Feature Mismatch**: Segmentation features may not be optimal for classification
- **Small Dataset**: Two-stage training further reduces effective training data per stage
- **No Proven Benefit**: For small datasets, end-to-end often outperforms two-stage

**When Two-Stage Might Be Justified**:
- If segmentation labels are available and high-quality
- If dataset is large enough to support both tasks
- If explicit tumor localization is required (not our case)

**Decision**: **Reject two-stage approach** for this project.

### 1.3 Why NOT Hybrid Approach?

**Reasons Against**:
- **Complexity**: Combining segmentation-derived features with encoder features adds design decisions
- **Feature Engineering**: Requires domain expertise to design meaningful hybrid features
- **Maintenance**: More complex codebase, harder to debug
- **Uncertain Benefit**: No clear evidence hybrid outperforms pure encoder approach

**Decision**: **Reject hybrid approach** for simplicity.

### 1.4 Detailed Architecture Specification

#### 1.4.1 Swin UNETR Encoder Configuration

**Base Model**: Swin UNETR (MONAI implementation)

**Key Parameters**:
- **Input Channels**: 4 (multi-modal: T1, T1ce, T2, FLAIR)
- **Image Size**: (128, 128, 128)
- **Patch Size**: (2, 2, 2) - Memory-efficient, preserves spatial detail
- **Window Size**: (7, 7, 7) - Standard for 3D Swin Transformers
- **Embedding Dimension**: 96 (base model) or 48 (smaller, if memory constrained)
- **Depths**: [2, 2, 2, 2] (4 stages, standard configuration)
- **Number of Heads**: [3, 6, 12, 24] (scales with depth)

**Memory Considerations**:
- Patch size 2×2×2 → 64³ = 262,144 patches (manageable)
- Window size 7×7×7 → Local attention (memory-efficient)
- Gradient checkpointing: Enable if memory is tight

#### 1.4.2 Encoder Output Processing

**Option A: Adaptive Average Pooling (Recommended)**
```python
# After encoder: (B, C, D', H', W') where D', H', W' depend on patch size
# Adaptive pooling: (B, C, D', H', W') → (B, C, 1, 1, 1)
# Flatten: (B, C)
features = F.adaptive_avg_pool3d(encoder_output, (1, 1, 1))
features = features.view(batch_size, -1)  # (B, C)
```

**Option B: Attention-Based Pooling (Alternative)**
```python
# Learnable attention weights over spatial dimensions
# More parameters, potentially better for small datasets
attention_weights = self.attention_pool(encoder_output)  # (B, 1, D', H', W')
features = (encoder_output * attention_weights).sum(dim=(2, 3, 4))  # (B, C)
```

**Recommendation**: **Option A (Adaptive Average Pooling)**
- Simpler, fewer parameters
- Less prone to overfitting on small dataset
- Standard practice for classification

#### 1.4.3 Classification Head

**Architecture**:
```python
ClassificationHead(
    in_features=encoder_dim,  # e.g., 768 for base model
    num_classes=2,
    hidden_dim=256,  # Optional hidden layer
    dropout=0.4
)
```

**Structure**:
```
Encoder Features (B, C)
    ↓
FC Layer: C → 256 (optional, can be skipped for simplicity)
    ↓
ReLU + Dropout(0.4)
    ↓
FC Layer: 256 → 2 (or C → 2 if no hidden layer)
    ↓
Output Logits (B, 2)
```

**Recommendation**: **Single FC layer** (C → 2) for smallest dataset
- Fewer parameters = less overfitting risk
- Can add hidden layer if underfitting observed

### 1.5 Complete Model Architecture

```python
class SwinUNETRClassifier(nn.Module):
    def __init__(
        self,
        img_size=(128, 128, 128),
        in_channels=4,
        num_classes=2,
        patch_size=(2, 2, 2),
        window_size=(7, 7, 7),
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        dropout=0.4,
        use_hidden_layer=False
    ):
        super().__init__()
        
        # Swin UNETR Encoder (MONAI)
        self.encoder = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=embed_dim,  # Output feature dimension
            depths=depths,
            num_heads=num_heads,
            feature_size=embed_dim,
            norm_name="instance",
            drop_rate=0.0,  # Dropout handled in classification head
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,  # Enable if memory constrained
            spatial_dims=3
        )
        
        # Remove decoder (not needed for classification)
        # Encoder outputs features at multiple resolutions
        # Use deepest (most abstract) features
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        encoder_output_dim = embed_dim * (2 ** (len(depths) - 1))  # After all stages
        if use_hidden_layer:
            self.classifier = nn.Sequential(
                nn.Linear(encoder_output_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(encoder_output_dim, num_classes)
            )
    
    def forward(self, x):
        # x: (B, 4, 128, 128, 128)
        
        # Encoder forward pass
        # Swin UNETR encoder returns features at multiple resolutions
        # Use the deepest (final) feature map
        encoder_features = self.encoder.encoder(x)  # Get encoder output only
        final_features = encoder_features[-1]  # Deepest resolution
        
        # Global pooling: (B, C, D', H', W') → (B, C, 1, 1, 1)
        pooled = self.global_pool(final_features)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        
        # Classification
        logits = self.classifier(pooled)  # (B, 2)
        
        return logits
```

**Note**: Actual MONAI SwinUNETR API may differ. Implementation should extract encoder features correctly.

---

## 2. Loss Function and Class Imbalance Strategy

### 2.1 Recommended: CrossEntropyLoss + WeightedRandomSampler

**Selected Approach**: **CrossEntropyLoss with WeightedRandomSampler**

**Rationale** (Based on ResNet50-3D Experience):

1. **Proven Stability**: ResNet50-3D achieved stable training with this combination
2. **No Conflicts**: Single balancing mechanism (data-level only)
3. **Simplicity**: Standard PyTorch components, well-understood behavior
4. **Small Dataset**: Complex loss functions (LDAM, Focal) designed for large datasets

**Configuration**:
```python
# Loss function
loss_fn = nn.CrossEntropyLoss()

# Class balancing
train_sampler = WeightedRandomSampler(
    weights=class_weights,  # Inverse frequency
    num_samples=len(train_dataset),
    replacement=True
)
```

### 2.2 Why NOT Focal Loss?

**Reasons Against**:
- **Hyperparameter Sensitivity**: Requires tuning `alpha` and `gamma`
- **Small Dataset**: Focal Loss designed for extreme imbalance (100:1+), our ratio is moderate (2.8:1)
- **No Proven Benefit**: WeightedRandomSampler already handles moderate imbalance effectively
- **Complexity**: Additional hyperparameters to tune and validate

**When Focal Loss Might Be Justified**:
- Extreme class imbalance (10:1 or worse)
- Large dataset where hyperparameter tuning is feasible
- Clear evidence of hard example mining benefits

**Decision**: **Reject Focal Loss** - unnecessary complexity for moderate imbalance.

### 2.3 Why NOT LDAM + DRW?

**Reasons Against** (From ResNet50-3D Experience):
- **Instability**: Caused loss fluctuations and erratic validation metrics
- **Triple Conflict**: When combined with WeightedRandomSampler, creates conflicting signals
- **Small Dataset**: LDAM margins are too aggressive for small batches
- **DRW Never Activated**: Training stopped before DRW could apply (early stopping)

**Decision**: **Reject LDAM + DRW** - proven to cause instability in this project.

### 2.4 Final Loss Configuration

```python
# Simple, stable configuration
loss_fn = nn.CrossEntropyLoss()

# Optional: Add class weights in loss (static, not DRW)
# Only if WeightedRandomSampler alone is insufficient
use_class_weights = False  # Start without, add if needed

if use_class_weights:
    class_weights = torch.tensor([
        n_total / (2 * n_minority),  # LGG
        n_total / (2 * n_majority)   # HGG
    ], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
else:
    loss_fn = nn.CrossEntropyLoss()
```

**Recommendation**: Start with **CrossEntropyLoss only** (no class weights in loss), rely on WeightedRandomSampler for balancing.

---

## 3. Training Strategy

### 3.1 End-to-End Training (Recommended)

**Approach**: Train entire model (encoder + classifier) end-to-end from scratch or with pretrained weights.

**Pretraining Options**:

**Option A: From Scratch (Recommended for Small Dataset)**
- Train encoder and classifier jointly
- No pretrained weights required
- Full control over feature learning

**Option B: MedicalNet Pretrained Encoder (If Available)**
- Initialize encoder with MedicalNet pretrained weights
- Adapt first layer for 4-channel input (similar to ResNet50-3D)
- Fine-tune encoder + train classifier

**Option C: Self-Supervised Pretraining (Advanced)**
- Pretrain encoder on unlabeled medical images
- Fine-tune for classification
- **Not recommended**: Adds complexity, uncertain benefit for small dataset

**Recommendation**: **Option A (From Scratch)** or **Option B (MedicalNet)** if pretrained Swin UNETR weights are available.

### 3.2 Why NOT Freeze Encoder?

**Reasons Against**:
- **Small Dataset**: Freezing reduces learnable parameters, but also limits adaptation
- **Domain Mismatch**: Pretrained weights (if any) may not match BraTS data distribution
- **End-to-End Benefit**: Joint training allows encoder to learn classification-relevant features

**When Freezing Might Be Justified**:
- Very limited GPU memory (freeze encoder, train only classifier)
- Pretrained weights are highly relevant (e.g., trained on similar brain MRI data)
- Initial experiments show overfitting with full training

**Decision**: **Train end-to-end** (no freezing), but monitor for overfitting.

### 3.3 Training Phases (If Needed)

**Phase 1: Encoder Warmup (Optional)**
- Train encoder with lower LR for few epochs
- Then add classifier and train jointly

**Phase 2: Joint Training (Standard)**
- Train encoder + classifier together
- Differential learning rates (encoder lower, classifier higher)

**Recommendation**: **Single phase (joint training)** for simplicity. Add warmup only if initial experiments show instability.

---

## 4. Optimization Configuration

### 4.1 Learning Rate Strategy

**Recommended**: **Differential Learning Rates** (Encoder vs Classifier)

**Configuration**:
```python
# Encoder (backbone): Lower LR for stability
encoder_lr = 5e-5  # Conservative for transformer

# Classifier: Higher LR for faster adaptation
classifier_lr = 1e-4  # 2× encoder LR

# Separate parameter groups
encoder_params = [p for name, p in model.named_parameters() 
                  if 'classifier' not in name]
classifier_params = [p for name, p in model.named_parameters() 
                     if 'classifier' in name]

optimizer = torch.optim.AdamW(
    [
        {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': 1e-4},
        {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': 1e-4}
    ],
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Rationale**:
- Transformers benefit from lower learning rates than CNNs
- Classifier head is small, can learn faster
- Conservative encoder LR prevents pretrained features (if any) from degrading

### 4.2 Learning Rate Schedule

**Recommended**: **Cosine Annealing with Warmup**

**Configuration**:
```python
# Warmup: 3-5 epochs (10% of total)
warmup_epochs = max(3, total_epochs // 10)

# Cosine annealing after warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_epochs - warmup_epochs,
    eta_min=1e-6
)

# Apply warmup manually or use LambdaLR
```

**Rationale**:
- Warmup prevents early instability
- Cosine annealing provides smooth decay
- Proven effective in ResNet50-3D training

### 4.3 Optimizer Selection

**Recommended**: **AdamW** (Adam with decoupled weight decay)

**Configuration**:
```python
optimizer = torch.optim.AdamW(
    params,
    lr=base_lr,
    weight_decay=1e-4,  # Decoupled weight decay
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Rationale**:
- AdamW is standard for Transformers
- Decoupled weight decay improves generalization
- More stable than SGD for transformer architectures

**Alternative**: Adam (if AdamW unavailable), but AdamW is preferred.

### 4.4 Batch Size and Memory Configuration

**Memory-Aware Design**:

**Recommended Configuration**:
```python
# Batch size: Start small, increase if memory allows
batch_size = 2  # Conservative for 3D transformers
gradient_accumulation_steps = 4  # Effective batch = 8

# Memory optimizations
use_gradient_checkpointing = True  # Trade compute for memory
mixed_precision = True  # FP16 training (AMP)
```

**Memory Estimation** (Rough):
- Input: (2, 4, 128, 128, 128) × 4 bytes = ~67 MB
- Encoder features: ~500-1000 MB (depends on model size)
- Gradients: ~500-1000 MB
- **Total**: ~2-3 GB per batch (with gradient checkpointing)

**If Memory Constrained**:
- Reduce batch size to 1
- Increase gradient accumulation to 8 (effective batch = 8)
- Enable gradient checkpointing
- Use smaller model (embed_dim=48 instead of 96)

### 4.5 Regularization

**Configuration**:
```python
# Dropout
encoder_dropout = 0.1  # Light dropout in encoder
classifier_dropout = 0.4  # Stronger dropout in classifier

# Weight decay
weight_decay = 1e-4  # Standard for transformers

# Early stopping
patience = 10
min_epochs = 15
```

**Rationale**:
- Light encoder dropout preserves feature learning
- Strong classifier dropout prevents overfitting
- Early stopping prevents overfitting on small dataset

---

## 5. Data Pipeline

### 5.1 Dataset Compatibility

**Reuse Existing**: `MultiModalVolume3DDataset`

**No Changes Required**:
- Already supports multi-modal input (4 channels)
- Compatible with MONAI transforms
- Returns shape: `(4, 128, 128, 128)`

**Transform Compatibility**:
- Use same transforms as ResNet50-3D
- MONAI transforms work with Swin UNETR
- Ensure transforms preserve channel dimension

### 5.2 Augmentation Strategy

**Reuse**: Same augmentation pipeline as ResNet50-3D

**Configuration**:
```python
# Training transforms
train_transforms = get_swin_unetr_transforms_3d(
    mode='train',
    num_channels=4
)

# Validation transforms
val_transforms = get_swin_unetr_transforms_3d(
    mode='val',
    num_channels=4
)
```

**Augmentation Parameters** (from ResNet50-3D, proven stable):
- Rotation: ±11.5° (0.2 radians)
- Flip: 0.5 probability per axis
- Zoom: ±8% (0.92-1.08)
- Translation: ±8%

---

## 6. Expected Risks and Mitigations

### 6.1 Risk: Memory Constraints

**Problem**: 3D Transformers are memory-intensive, may exceed GPU memory.

**Mitigations**:
1. **Gradient Checkpointing**: Trade compute for memory (~30% slower, ~50% less memory)
2. **Smaller Model**: Reduce `embed_dim` from 96 to 48
3. **Smaller Batch Size**: Use batch_size=1 with gradient accumulation
4. **Mixed Precision**: FP16 training reduces memory by ~50%

**Monitoring**: Track GPU memory usage, adjust if >90% utilization.

### 6.2 Risk: Overfitting on Small Dataset

**Problem**: Transformers have many parameters, may overfit on 228 training samples.

**Mitigations**:
1. **Strong Regularization**: Dropout 0.4 in classifier, weight decay 1e-4
2. **Early Stopping**: Patience=10, monitor validation AUC
3. **Data Augmentation**: Aggressive augmentation (within medical safety limits)
4. **Smaller Model**: Start with smaller `embed_dim` if overfitting observed

**Monitoring**: Track train/val gap. If train AUC >> val AUC, increase regularization.

### 6.3 Risk: Training Instability

**Problem**: Transformers can be unstable, especially with small batches.

**Mitigations**:
1. **Conservative Learning Rates**: Encoder LR = 5e-5 (lower than CNN)
2. **Gradient Clipping**: Clip gradients to max_norm=1.0
3. **Warmup**: 3-5 epoch warmup before full LR
4. **Mixed Precision**: Use AMP with careful scaling

**Monitoring**: Track loss curves. If loss explodes or collapses, reduce LR.

### 6.4 Risk: Slow Convergence

**Problem**: Transformers may require more epochs than CNNs.

**Mitigations**:
1. **Adequate Epochs**: Plan for 40-60 epochs (more than ResNet50-3D)
2. **Learning Rate Schedule**: Cosine annealing with sufficient decay
3. **Patience**: Early stopping patience=10 (allow more epochs)

**Monitoring**: Track validation metrics. If still improving at epoch 40+, continue training.

### 6.5 Risk: Incompatibility with Existing Pipeline

**Problem**: Swin UNETR may not integrate cleanly with existing codebase.

**Mitigations**:
1. **MONAI Integration**: Use MONAI's SwinUNETR (standard API)
2. **Consistent Interface**: Model should match ResNet50-3D interface (forward returns logits)
3. **Shared Utilities**: Reuse dataset, transforms, evaluation code

**Monitoring**: Ensure model works with existing training script structure.

---

## 7. Implementation Plan

### 7.1 Model Implementation

**File**: `models/swin_unetr_classifier.py`

**Key Components**:
1. `SwinUNETRClassifier` class
2. Encoder extraction (remove decoder)
3. Global pooling layer
4. Classification head
5. Forward pass implementation

**Dependencies**:
- `monai` (for SwinUNETR base)
- `torch` (standard PyTorch)

### 7.2 Training Script

**File**: `scripts/training/train_swin_unetr.py`

**Structure** (mirror ResNet50-3D training script):
1. Argument parsing
2. Dataset loading (reuse `MultiModalVolume3DDataset`)
3. Model initialization
4. Loss function setup (CrossEntropyLoss)
5. Optimizer setup (AdamW with differential LRs)
6. Training loop
7. Validation loop
8. Checkpoint saving (best.pt, best_ema.pt)
9. Final evaluation (explicit checkpoint loading)

**Key Differences from ResNet50-3D**:
- Different model architecture
- Lower learning rates (transformer-specific)
- Potentially longer training (more epochs)

### 7.3 Configuration File

**File**: `configs/swin_unetr_config.yaml`

**Parameters**:
```yaml
model:
  img_size: [128, 128, 128]
  in_channels: 4
  num_classes: 2
  patch_size: [2, 2, 2]
  window_size: [7, 7, 7]
  embed_dim: 96
  depths: [2, 2, 2, 2]
  num_heads: [3, 6, 12, 24]
  dropout: 0.4

training:
  batch_size: 2
  gradient_accumulation_steps: 4
  encoder_lr: 5e-5
  classifier_lr: 1e-4
  weight_decay: 1e-4
  epochs: 60
  warmup_epochs: 5
  early_stopping_patience: 10
  early_stopping_min_epochs: 15
  grad_clip: 1.0
  mixed_precision: true
  use_gradient_checkpointing: true
```

---

## 8. Expected Performance and Comparison

### 8.1 Performance Expectations

**Realistic Targets** (based on ResNet50-3D baseline):
- **AUC**: 0.85-0.90 (similar or slightly better than ResNet50-3D: 0.8794 ± 0.0436)
- **F1-score**: 0.75-0.80 (similar to ResNet50-3D: 0.7619 ± 0.0677)
- **Stability**: Smooth training curves, consistent validation metrics

**Why Similar Performance?**:
- Small dataset limits model capacity utilization
- Both architectures (CNN and Transformer) can learn effective features
- Performance ceiling determined by data quality and quantity, not architecture alone

**Potential Advantages of Swin UNETR**:
- **Long-range Dependencies**: Transformers capture global context better than CNNs
- **Multi-scale Features**: Hierarchical attention may capture tumor characteristics at different scales
- **Complementary to ResNet**: Different inductive biases → good for ensemble

### 8.2 Ensemble Potential

**Goal**: Ensemble Swin UNETR + ResNet50-3D + (potentially MIL)

**Benefits**:
- **Diversity**: Different architectures → different failure modes → better generalization
- **Robustness**: Ensemble reduces variance across folds
- **Performance**: Typically 2-5% improvement over single models

**Ensemble Strategy** (Future Work):
- Weighted average of predictions
- Stacking with meta-learner
- Voting (majority or weighted)

---

## 9. Alternative Considerations

### 9.1 Smaller Transformer Variants

**Option**: Use smaller Swin UNETR variant (embed_dim=48, fewer layers)

**When to Consider**:
- GPU memory is severely limited
- Overfitting observed with full model
- Faster training needed

**Trade-off**: Smaller model = less capacity, but may still perform well on small dataset.

### 9.2 Attention-Based Pooling

**Option**: Replace adaptive average pooling with learnable attention pooling

**When to Consider**:
- Initial experiments show underfitting
- Need to focus on specific spatial regions
- Have sufficient data to learn attention weights

**Trade-off**: More parameters, risk of overfitting, but potentially better feature aggregation.

### 9.3 Multi-Scale Feature Fusion

**Option**: Combine features from multiple encoder stages (not just deepest)

**When to Consider**:
- Need fine-grained spatial information
- Tumor characteristics span multiple scales
- Have capacity for additional parameters

**Trade-off**: More complex, more parameters, but may capture multi-scale patterns.

**Recommendation**: **Start simple** (single-stage features), add complexity only if needed.

---

## 10. Final Recommendations Summary

### 10.1 Architecture

✅ **End-to-End Classification with Swin UNETR Encoder**
- Encoder-only (no decoder)
- Global average pooling
- Simple classification head (single FC layer)
- Input: (4, 128, 128, 128) multi-modal

### 10.2 Loss Function

✅ **CrossEntropyLoss + WeightedRandomSampler**
- Standard CrossEntropyLoss (no class weights in loss)
- WeightedRandomSampler for data-level balancing
- No LDAM, no DRW, no Focal Loss

### 10.3 Training Strategy

✅ **End-to-End Joint Training**
- Train encoder + classifier together
- Differential learning rates (encoder: 5e-5, classifier: 1e-4)
- No freezing, no multi-stage training

### 10.4 Optimization

✅ **AdamW with Cosine Annealing**
- Encoder LR: 5e-5 (conservative)
- Classifier LR: 1e-4 (2× encoder)
- Cosine annealing with 3-5 epoch warmup
- Gradient clipping: 1.0
- Mixed precision: Enabled

### 10.5 Regularization

✅ **Conservative Regularization**
- Dropout: 0.4 (classifier), 0.1 (encoder)
- Weight decay: 1e-4
- Early stopping: Patience=10, min_epochs=15
- Data augmentation: Same as ResNet50-3D

### 10.6 Memory Management

✅ **Memory-Aware Configuration**
- Batch size: 2 (with gradient accumulation=4)
- Gradient checkpointing: Enabled
- Mixed precision: Enabled
- Monitor GPU memory usage

### 10.7 Checkpointing

✅ **Explicit Checkpoint Policy** (same as ResNet50-3D)
- Save best.pt (by validation AUC)
- Save best_ema.pt (if EMA enabled)
- Final evaluation: Load best_ema.pt or best.pt explicitly
- Never use last.pt for final evaluation
- Verify checkpoint AUC matches evaluation AUC

---

## 11. Implementation Checklist

### Phase 1: Model Implementation
- [ ] Create `models/swin_unetr_classifier.py`
- [ ] Implement encoder extraction (remove decoder)
- [ ] Implement global pooling
- [ ] Implement classification head
- [ ] Test forward pass with dummy input
- [ ] Verify output shape: (B, 2)

### Phase 2: Training Script
- [ ] Create `scripts/training/train_swin_unetr.py`
- [ ] Reuse dataset loading (MultiModalVolume3DDataset)
- [ ] Implement model initialization
- [ ] Implement loss function (CrossEntropyLoss)
- [ ] Implement optimizer (AdamW with differential LRs)
- [ ] Implement training loop
- [ ] Implement validation loop
- [ ] Implement checkpoint saving (best.pt, best_ema.pt)
- [ ] Implement final evaluation (explicit checkpoint loading)

### Phase 3: Configuration
- [ ] Create `configs/swin_unetr_config.yaml`
- [ ] Set memory-efficient parameters
- [ ] Set conservative learning rates
- [ ] Set regularization parameters

### Phase 4: Testing
- [ ] Dry-run: Load one batch, verify shapes
- [ ] Test training for 1 epoch (no errors)
- [ ] Verify checkpoint saving/loading
- [ ] Verify final evaluation uses correct checkpoint

### Phase 5: Full Training
- [ ] Train fold 0 (pilot)
- [ ] Monitor: Loss curves, GPU memory, training time
- [ ] Adjust parameters if needed
- [ ] Train all 5 folds
- [ ] Evaluate and compare with ResNet50-3D

---

## 12. Success Criteria

### 12.1 Training Stability
- ✅ Smooth, monotonic loss curves (no oscillations)
- ✅ Consistent validation metrics (AUC variance < 0.05)
- ✅ No loss explosions or collapses
- ✅ Stable GPU memory usage

### 12.2 Performance
- ✅ Validation AUC: 0.85-0.90 (competitive with ResNet50-3D)
- ✅ Final evaluation AUC matches checkpoint AUC (verified)
- ✅ Consistent results across folds

### 12.3 Reproducibility
- ✅ Same configuration produces similar results
- ✅ Explicit checkpoint loading (no ambiguity)
- ✅ Clear logging of all hyperparameters

### 12.4 Integration
- ✅ Compatible with existing data pipeline
- ✅ Compatible with k-fold cross-validation
- ✅ Ready for ensemble with ResNet50-3D

---

## 13. Conclusion

This proposal recommends a **simple, stable, end-to-end classification architecture** using Swin UNETR encoder with CrossEntropyLoss and WeightedRandomSampler. The design prioritizes **stability and reproducibility** over complexity, learning from the ResNet50-3D training experience.

**Key Principles**:
1. **Simplicity**: Single-stage training, standard components
2. **Stability**: Proven loss/balancing strategy, conservative hyperparameters
3. **Memory-Aware**: Gradient checkpointing, mixed precision, small batches
4. **Reproducibility**: Explicit checkpointing, clear logging, consistent evaluation

**Expected Outcome**: A robust Swin UNETR classifier that complements ResNet50-3D in an ensemble, achieving similar or slightly better performance with different inductive biases.

---

**Document Version**: 1.0  
**Date**: 2025-01-07  
**Author**: Medical Imaging Pipeline

