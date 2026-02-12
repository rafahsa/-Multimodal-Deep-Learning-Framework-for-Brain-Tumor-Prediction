# Technical Decision Brief: Heatmap Visualization for Medical Imaging Ensemble

**Date**: 2026-02-10  
**Author**: Senior ML Researcher (Model Interpretability)  
**Purpose**: Determine whether, where, and how to add heatmaps/visual explanations to the current ensemble pipeline

---

## A) Inventory of Current Models

### Model Architecture Summary

| Model | Type | Input Resolution | Spatial Info Preserved | Intermediate Activations Available | Attention Weights Available |
|-------|------|----------------|----------------------|-----------------------------------|------------------------------|
| **ResNet50-3D** | 3D CNN | 128×128×128 (4 channels) | ❌ No (Global AvgPool) | ✅ Yes (via hooks) | ❌ No |
| **SwinUNETR-3D** | Transformer Encoder | 128×128×128 (4 channels) | ❌ No (Global AvgPool) | ✅ Yes (via hooks) | ⚠️ Partial (self-attention in encoder, but pooled) |
| **DualStreamMIL** | MIL (2D slices) | 64 slices × 224×224 (4 channels) | ⚠️ Partial (slice-level) | ✅ Yes (instance encoder) | ✅ **YES** (via `return_interpretability=True`) |
| **Meta-Learner** | Logistic Regression | 3 features (probabilities) | ❌ No (decision-level) | ❌ N/A | ❌ N/A |

### Detailed Model Specifications

#### 1. ResNet50-3D
- **Architecture**: 3D ResNet50 (Bottleneck3D blocks)
- **Input**: `(B, 4, 128, 128, 128)` - Multi-modal 3D volumes
- **Backbone**: MedicalNet pretrained (optional)
- **Pooling**: `AdaptiveAvgPool3d((1,1,1))` → **spatial information lost**
- **Output**: `(B, 2)` logits
- **Spatial Level**: Patient-level (no spatial preservation)
- **Activation Recovery**: ✅ Possible via forward hooks on `layer4` (before pooling)
- **File**: `models/resnet50_3d_fast/model.py`
- **Checkpoint Path**: `results/ResNet50-3D/runs/fold_X/YYYYMMDD_HHMMSS/checkpoints/best.pt`

#### 2. SwinUNETR-3D
- **Architecture**: Swin Transformer encoder (MONAI)
- **Input**: `(B, 4, 128, 128, 128)` - Multi-modal 3D volumes
- **Encoder**: SwinViT with 4 stages, patch_size=2, window_size=7
- **Pooling**: `AdaptiveAvgPool3d((1,1,1))` → **spatial information lost**
- **Output**: `(B, 2)` logits
- **Spatial Level**: Patient-level (no spatial preservation)
- **Activation Recovery**: ✅ Possible via hooks on encoder stages (before pooling)
- **Self-Attention**: ✅ Available in encoder blocks, but aggregated before pooling
- **File**: `models/swin_unetr_encoder.py`
- **Checkpoint Path**: `results/Swin_UNETR/runs/fold_X/YYYYMMDD_HHMMSS/checkpoints/best.pt`

#### 3. DualStreamMIL
- **Architecture**: Dual-Stream MIL with instance encoder
- **Input**: `(B, N, 4, 224, 224)` - Bag of N 2D slices (N=64 default)
- **Instance Encoder**: ResNet18/EfficientNet-B0 (2D CNN) → `(B, N, 512)` features
- **Stream 1**: Critical Instance Selector (soft selection with temperature)
- **Stream 2**: Contextual Aggregator (gated/cosine attention)
- **Output**: `(B, 2)` logits
- **Spatial Level**: Slice-level (spatial info preserved per slice, but not 3D coordinates)
- **Attention Weights**: ✅ **AVAILABLE** via `return_interpretability=True`:
  - `attention_weights`: `(B, N)` - Contextual attention per slice
  - `selection_weights`: `(B, N)` - Critical instance selection weights
  - `instance_scores`: `(B, N)` - Raw scores before softmax
  - `critical_idx`: `(B,)` - Index of highest-scored slice
- **File**: `models/dual_stream_mil.py`
- **Checkpoint Path**: `results/MIL/runs/fold_X/YYYYMMDD_HHMMSS/checkpoints/best.pt`
- **Slice Coordinates**: ⚠️ **NOT EXPLICITLY SAVED** - slices are selected via entropy, but z-coordinates not stored

#### 4. Meta-Learner
- **Architecture**: Logistic Regression (scikit-learn)
- **Input**: `(N, 3)` - Patient-level probabilities from base models
  - `hgg_prob_resnet`: ResNet50-3D probability
  - `hgg_prob_swin`: SwinUNETR-3D probability
  - `mil_prob`: DualStreamMIL probability
- **Output**: `(N, 2)` probabilities (patient-level)
- **Spatial Level**: Decision-level (no spatial information)
- **File**: `scripts/ensemble/train_meta_learner.py`
- **Model Path**: `ensemble/models/meta_learner_logistic_regression.joblib`
- **Coefficients**: Available via `model.coef_` (feature importance)

---

## B) Heatmap Feasibility Analysis (Per Model)

### 1. ResNet50-3D

**Is spatial heatmap meaningful?** ⚠️ **PARTIALLY**

**Rationale**:
- Model operates on full 3D volumes → spatial heatmaps are **conceptually meaningful**
- However, global pooling removes spatial information → requires **gradient-based methods**

**Method**: **Grad-CAM / Grad-CAM++**
- **Target Layer**: `layer4` (final convolutional layer before pooling)
- **Gradient Source**: Classification logit (HGG class)
- **Output Shape**: `(128, 128, 128)` - Full 3D heatmap
- **Implementation**: Standard Grad-CAM on 3D CNN
  - Hook gradients on `layer4` output
  - Compute weighted feature maps
  - Upsample to input resolution (if needed)

**Alternative**: Score-CAM (no gradients needed, but slower)

**Challenges**:
- 3D volumes are large → memory considerations
- Need to handle multi-modal input (4 channels) → aggregate across channels or show per-modality
- No existing infrastructure for gradient hooks

**Verdict**: ✅ **FEASIBLE** with moderate effort

---

### 2. SwinUNETR-3D

**Is spatial heatmap meaningful?** ⚠️ **PARTIALLY**

**Rationale**:
- Transformer encoder with self-attention → **attention-based heatmaps are meaningful**
- However, global pooling removes spatial information → requires **gradient-based or attention-based methods**

**Method Options**:

**Option A: Attention Rollout / Attention Flow**
- **Target**: Self-attention weights in SwinViT encoder blocks
- **Output**: Attention-weighted feature maps
- **Pros**: Native to transformer architecture, interpretable
- **Cons**: Requires extracting attention from MONAI SwinUNETR (may need code modification)

**Option B: Grad-CAM on Encoder Features**
- **Target Layer**: Final encoder stage (before pooling)
- **Gradient Source**: Classification logit
- **Output Shape**: `(128, 128, 128)` (after upsampling from patch resolution)
- **Pros**: Standard method, works with existing architecture
- **Cons**: Patch-based architecture → lower resolution heatmap

**Challenges**:
- MONAI SwinUNETR may not expose attention weights directly
- Patch-based architecture → heatmap resolution is lower than input
- Need to handle multi-head attention aggregation

**Verdict**: ⚠️ **FEASIBLE** but requires investigation of MONAI attention extraction

---

### 3. DualStreamMIL

**Is spatial heatmap meaningful?** ✅ **YES - HIGHEST PRIORITY**

**Rationale**:
- **Attention weights are already available** via `return_interpretability=True`
- Model operates on 2D slices → **slice-level heatmaps are directly meaningful**
- Two interpretability signals:
  1. **Contextual Attention**: Which slices are most important for aggregation
  2. **Critical Instance Selection**: Which slice is the most critical

**Method**: **Attention-Based MIL Heatmap**
- **Data Available**:
  - `attention_weights`: `(B, N)` - Per-slice attention (Stream 2)
  - `selection_weights`: `(B, N)` - Per-slice selection weights (Stream 1)
  - `critical_idx`: `(B,)` - Index of critical slice
- **Visualization**:
  - **Slice-level heatmap**: Bar chart or color-coded slice indices
  - **2D slice visualization**: Overlay attention on actual slices
  - **3D reconstruction**: Map slice indices back to z-coordinates (if available)

**Implementation**:
```python
# Already available in model forward pass:
logits, interpretability = model(bag_of_slices, return_interpretability=True)
attention_weights = interpretability['attention_weights']  # (B, N)
selection_weights = interpretability['selection_weights']  # (B, N)
critical_idx = interpretability['critical_idx']  # (B,)
```

**Challenges**:
- ⚠️ **Slice z-coordinates not saved** → need to reconstruct from entropy JSON or dataset
- Need to map slice indices back to original 3D volume positions
- Entropy-based slice selection → slices may not be sequential

**Verdict**: ✅ **HIGHLY FEASIBLE** - infrastructure already exists

---

### 4. Meta-Learner

**Is spatial heatmap meaningful?** ❌ **NO**

**Rationale**:
- Operates on patient-level probabilities (no spatial information)
- Decision-level model (combines base model outputs)

**Alternative Explanation**:
- ✅ **Feature Importance**: Coefficient magnitudes (`model.coef_`)
- ✅ **SHAP Values**: Per-patient feature contributions
- ✅ **Decision Boundaries**: Visualization in 3D probability space

**Verdict**: ❌ **NOT APPLICABLE** - use feature importance instead

---

## C) Data & Code Requirements

### 1. Access to Trained Weights

**Status**: ✅ **AVAILABLE**

| Model | Checkpoint Path Pattern | Status |
|-------|------------------------|--------|
| ResNet50-3D | `results/ResNet50-3D/runs/fold_X/YYYYMMDD_HHMMSS/checkpoints/best.pt` | ✅ Available |
| SwinUNETR-3D | `results/Swin_UNETR/runs/fold_X/YYYYMMDD_HHMMSS/checkpoints/best.pt` | ✅ Available |
| DualStreamMIL | `results/MIL/runs/fold_X/YYYYMMDD_HHMMSS/checkpoints/best.pt` | ✅ Available |
| Meta-Learner | `ensemble/models/meta_learner_logistic_regression.joblib` | ✅ Available |

**Action Required**: Identify latest/best checkpoints per fold (may need to query filesystem)

---

### 2. Need to Re-run Inference?

**Status**: ⚠️ **PARTIALLY REQUIRED**

| Model | Current Inference Output | Heatmap Generation Needs |
|-------|-------------------------|-------------------------|
| ResNet50-3D | Patient-level probabilities only | ✅ **YES** - Need to re-run with gradient hooks |
| SwinUNETR-3D | Patient-level probabilities only | ✅ **YES** - Need to re-run with attention/gradient hooks |
| DualStreamMIL | Patient-level probabilities only | ⚠️ **PARTIAL** - Attention weights available, but need to save them |
| Meta-Learner | Patient-level probabilities | ❌ **NO** - Use existing coefficients |

**Current Inference Scripts**:
- `scripts/ensemble/test_ensemble_on_new_patients.py` - Does not save activations/attention
- `scripts/training/train_*.py` - Training scripts, not inference-focused

**Action Required**:
- Create dedicated inference script with interpretability hooks
- For MIL: Modify existing inference to save `interpretability` dict

---

### 3. Tile Coordinates / WSI Mapping Availability

**Status**: ⚠️ **PARTIALLY AVAILABLE**

**MIL Slice Mapping**:
- **Entropy JSON**: `data/entropy/<patient_id>_entropy.json` - Contains slice indices and entropy scores
- **Slice Selection**: Entropy-based top-k selection (not sequential)
- **Z-Coordinates**: ⚠️ **NOT EXPLICITLY SAVED** - Need to reconstruct from:
  1. Entropy JSON (slice indices)
  2. Dataset loading code (`utils/dataset_mil.py`)
  3. Original volume dimensions (128×128×128)

**3D CNN Models**:
- **Full Volume**: 128×128×128 voxels
- **Spatial Mapping**: Direct (1:1 mapping to input volume)
- **No tile/patch coordinates needed**

**Action Required**:
- For MIL: Create utility to map slice indices → z-coordinates
- For 3D CNNs: Direct mapping (no conversion needed)

---

### 4. Current Logging/Saving

**Status**: ❌ **NOT SAVED**

**What's Currently Saved**:
- ✅ Patient-level predictions (`hgg_prob_*`, `mil_prob`)
- ✅ Model checkpoints (weights only)
- ✅ Training metrics (AUC, F1, etc.)
- ❌ **Attention weights** - NOT saved
- ❌ **Activations** - NOT saved
- ❌ **Slice coordinates** - NOT saved

**What's Available at Runtime**:
- ✅ MIL attention weights (via `return_interpretability=True`)
- ✅ CNN activations (via forward hooks - not currently implemented)
- ⚠️ SwinUNETR attention (needs investigation)

**Action Required**:
- Create inference script that saves interpretability outputs
- Add hooks for CNN activations
- Investigate SwinUNETR attention extraction

---

## D) Integration Points (WHERE Heatmaps Fit)

### 1. Generation Stage

**Recommended**: **Post-hoc on Selected Samples**

**Rationale**:
- Heatmaps are computationally expensive (especially 3D Grad-CAM)
- Not needed for all patients (validation set has 285 patients)
- Focus on:
  1. **High-confidence correct predictions** (validate model behavior)
  2. **High-confidence incorrect predictions** (error analysis)
  3. **Uncertain predictions** (understand failure modes)
  4. **Representative samples** (per class, per fold)

**Alternative**: During validation (too expensive for all samples)

**Implementation**:
- Create `scripts/analysis/generate_heatmaps.py`
- Input: List of patient IDs (or automatic selection criteria)
- Output: Heatmap visualizations + metadata JSON

---

### 2. Relationship to Ensemble

**Strategy**: **Single Representative Model + MIL (Primary Focus)**

**Rationale**:
- **MIL**: Highest interpretability value (attention weights available)
- **One CNN**: Choose ResNet50-3D (simpler than SwinUNETR, standard Grad-CAM)
- **Meta-Learner**: Skip heatmaps, use feature importance instead

**Justification**:
- MIL attention maps show **which slices** the model focuses on → directly interpretable
- CNN Grad-CAM shows **which regions** in 3D volume → complementary
- Showing both ResNet50-3D and SwinUNETR-3D is redundant (both are 3D CNNs)

**Alternative**: Show all three base models (higher effort, diminishing returns)

---

### 3. Results Location

**Recommended Structure**:
```
ensemble/results/interpretability/
├── heatmaps/
│   ├── mil_attention/
│   │   ├── patient_001_slice_heatmap.png
│   │   ├── patient_001_3d_overlay.png
│   │   └── patient_001_metadata.json
│   └── cnn_gradcam/
│       ├── patient_001_gradcam_3d.nii.gz
│       ├── patient_001_gradcam_slices.png
│       └── patient_001_metadata.json
├── summaries/
│   ├── mil_attention_summary.csv
│   └── cnn_gradcam_summary.csv
└── README.md
```

**Integration with Existing Results**:
- **Main Results**: `ensemble/results/` (existing)
- **Interpretability**: `ensemble/results/interpretability/` (new)
- **Paper/Thesis**: Reference in methods section, include in appendix

---

## E) Scientific Justification

### 1. Methodological Soundness

**Question**: Is it methodologically sound to visualize only one CNN + one MIL model?

**Answer**: ✅ **YES**

**Justification**:
- **MIL attention maps** are the **primary interpretability signal** (native to architecture, directly meaningful)
- **One CNN Grad-CAM** provides **complementary spatial information** (3D region importance)
- **SwinUNETR-3D** is architecturally similar to ResNet50-3D (both 3D CNNs) → showing both is redundant
- **Meta-learner** is decision-level → heatmaps not applicable

**Standard Practice**:
- Medical imaging papers typically show **1-2 representative models** for interpretability
- Focus on **highest-impact visualizations** (MIL attention) rather than exhaustive coverage
- **Feature importance** (meta-learner coefficients) is standard for ensemble interpretability

---

### 2. Citations & Standard Practices

**MIL Attention Visualization**:
- **Standard**: Attention weight visualization on instances (slices) is standard in MIL literature
- **Reference**: Ilse et al. (2018) "Attention-based Deep Multiple Instance Learning" - attention weights as interpretability
- **Medical MIL**: Common practice to visualize attention on patches/slices (e.g., histopathology, radiology)

**CNN Explainability in Medical Imaging**:
- **Grad-CAM**: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks" - standard for CNN interpretability
- **Medical Imaging**: Grad-CAM widely used in radiology papers (e.g., brain MRI, chest X-ray)
- **3D Grad-CAM**: Extension to 3D volumes is straightforward (same principles)

**Ensemble Interpretability**:
- **Feature Importance**: Standard for meta-learners (coefficient magnitudes)
- **Multi-Model Visualization**: Typically show 1-2 base models + ensemble decision

---

### 3. Claims We CAN Make

✅ **Supported Claims**:
1. **MIL Model**: "The model focuses on slices X, Y, Z (high attention weights) for this patient"
2. **CNN Model**: "The model's decision is driven by regions in the [anterior/posterior/left/right] hemisphere"
3. **Complementary Signals**: "MIL identifies critical slices, while CNN highlights 3D spatial regions"
4. **Error Analysis**: "False negatives show attention on [specific regions/slices]"

---

### 4. Claims We CANNOT Make

❌ **Unsupported Claims** (to avoid reviewer criticism):
1. **Causal Attribution**: "Region X **causes** the prediction" (heatmaps show correlation, not causation)
2. **Complete Explanation**: "This heatmap explains **all** model behavior" (partial explanation only)
3. **Clinical Validation**: "Heatmaps match radiologist annotations" (would require ground truth annotations)
4. **Generalization**: "These heatmaps represent all patients" (shown for selected samples only)
5. **Meta-Learner Spatial Info**: "Meta-learner focuses on spatial regions" (decision-level only)

**Mitigation**:
- Use cautious language: "suggests", "indicates", "may reflect"
- Acknowledge limitations in methods section
- Show multiple examples (not cherry-picked)

---

## F) Time & Effort Estimation

### 1. Engineering Effort Breakdown

#### A) MIL Attention Heatmap
**Effort**: 🟢 **LOW** (2-4 hours)

**Tasks**:
1. Create inference script with `return_interpretability=True` (30 min)
2. Extract and save attention weights (30 min)
3. Map slice indices to z-coordinates (1 hour)
4. Visualize slice-level attention (bar chart, color-coded) (1 hour)
5. Overlay attention on 2D slices (1 hour)
6. Testing and validation (30 min)

**Dependencies**: ✅ All infrastructure exists

---

#### B) CNN Grad-CAM (ResNet50-3D)
**Effort**: 🟡 **MEDIUM** (4-6 hours)

**Tasks**:
1. Implement Grad-CAM for 3D CNN (2 hours)
   - Forward hook on `layer4`
   - Gradient computation
   - Weighted feature map aggregation
2. Handle multi-modal input (aggregate across channels) (1 hour)
3. Upsample to input resolution (if needed) (30 min)
4. Visualize 3D heatmap (slice-by-slice or 3D rendering) (1 hour)
5. Testing and validation (1 hour)

**Dependencies**: 
- Need to install/implement Grad-CAM library (e.g., `grad-cam` or custom)
- Memory considerations for 3D volumes

---

#### C) SwinUNETR Attention (Optional)
**Effort**: 🔴 **HIGH** (6-8 hours)

**Tasks**:
1. Investigate MONAI SwinUNETR attention extraction (2 hours)
2. Implement attention rollout/flow (2 hours)
3. Handle multi-head attention aggregation (1 hour)
4. Visualize attention maps (1 hour)
5. Testing and validation (2 hours)

**Dependencies**: 
- MONAI documentation/investigation
- May require code modifications

**Recommendation**: ⚠️ **SKIP** (not in minimal plan)

---

#### D) Aggregation & Visualization
**Effort**: 🟡 **MEDIUM** (3-4 hours)

**Tasks**:
1. Create unified visualization pipeline (1 hour)
2. Generate summary statistics (attention distribution, etc.) (1 hour)
3. Create comparison visualizations (MIL vs CNN) (1 hour)
4. Documentation and README (1 hour)

---

### 2. Total Time Estimation

| Scenario | MIL Only | MIL + ResNet50-3D | MIL + ResNet50-3D + SwinUNETR |
|----------|----------|-------------------|-------------------------------|
| **Best Case** | 2-4 hours | 6-10 hours | 12-18 hours |
| **Realistic** | 3-5 hours | 8-12 hours | 15-22 hours |
| **Worst Case** | 4-6 hours | 10-15 hours | 18-25 hours |

**Recommended**: **MIL + ResNet50-3D** (8-12 hours realistic)

---

### 3. Risk Factors

**High Risk** (could increase time significantly):
1. ⚠️ **Slice coordinate mapping** - If entropy JSON doesn't contain z-coordinates, need to reverse-engineer (adds 2-4 hours)
2. ⚠️ **Memory issues with 3D Grad-CAM** - Large volumes may require batching/chunking (adds 1-2 hours)
3. ⚠️ **MONAI SwinUNETR attention** - If attention not easily extractable, need custom implementation (adds 4-6 hours)

**Medium Risk**:
1. **Checkpoint loading** - Need to identify correct checkpoints per fold (adds 1 hour)
2. **Visualization aesthetics** - Medical imaging standards may require specific colormaps/formats (adds 1-2 hours)

**Low Risk**:
1. **Dependencies** - Standard libraries (grad-cam, matplotlib) should be straightforward

---

## G) Final Recommendation

### 1. Decision: ✅ **YES - Add Heatmaps (Minimal, High-Impact Plan)**

**Rationale**:
- **MIL attention maps** are **highly feasible** (infrastructure exists) and **highly interpretable**
- **CNN Grad-CAM** provides **complementary spatial information** with moderate effort
- **Scientific value** is significant (standard practice in medical imaging papers)
- **Time investment** is reasonable (8-12 hours for core implementation)

---

### 2. Minimal Implementation Plan

#### Phase 1: MIL Attention Heatmaps (Priority 1)
**Deliverables**:
1. Inference script that saves MIL attention weights
2. Slice-level attention visualization (bar chart, color-coded)
3. 2D slice overlay (attention weights on actual slices)
4. Summary statistics (attention distribution per patient)

**Time**: 3-5 hours  
**Files to Create**:
- `scripts/analysis/generate_mil_attention_heatmaps.py`
- `utils/interpretability/mil_visualization.py`

---

#### Phase 2: ResNet50-3D Grad-CAM (Priority 2)
**Deliverables**:
1. 3D Grad-CAM implementation for ResNet50-3D
2. Slice-by-slice visualization (axial, coronal, sagittal)
3. Summary heatmap (aggregated across modalities)

**Time**: 4-6 hours  
**Files to Create**:
- `scripts/analysis/generate_cnn_gradcam.py`
- `utils/interpretability/gradcam_3d.py`

---

#### Phase 3: Integration & Documentation (Priority 3)
**Deliverables**:
1. Unified visualization pipeline
2. Comparison visualizations (MIL vs CNN)
3. Documentation (README, methods section text)

**Time**: 2-3 hours  
**Files to Create**:
- `ensemble/results/interpretability/README.md`
- `scripts/analysis/generate_all_heatmaps.py` (orchestration script)

---

### 3. What to Skip Deliberately

❌ **SwinUNETR-3D Attention**:
- Higher effort, lower added value (similar to ResNet50-3D)
- Requires investigation of MONAI internals
- **Rationale**: One CNN Grad-CAM is sufficient

❌ **Meta-Learner Heatmaps**:
- Not applicable (decision-level model)
- **Alternative**: Feature importance bar chart (coefficient magnitudes)

❌ **All-Patient Heatmaps**:
- Too expensive (285 validation patients)
- **Alternative**: Selected samples (high-confidence, errors, representative)

❌ **3D Volume Rendering**:
- Complex, may not add value over slice-by-slice
- **Alternative**: Multi-planar slice visualization (axial, coronal, sagittal)

---

### 4. Success Criteria

**Minimum Viable**:
- ✅ MIL attention heatmaps for 10-20 selected patients
- ✅ ResNet50-3D Grad-CAM for same patients
- ✅ Clear visualization (interpretable by non-experts)

**Ideal**:
- ✅ 30-50 patients (representative across classes, folds, confidence levels)
- ✅ Comparison visualizations (MIL vs CNN side-by-side)
- ✅ Summary statistics (attention distribution, heatmap statistics)

---

## Appendix: Implementation Checklist

### Prerequisites
- [ ] Identify best checkpoints per fold for each model
- [ ] Verify entropy JSON structure (slice indices, z-coordinates)
- [ ] Install Grad-CAM library (or implement custom)

### MIL Attention Heatmaps
- [ ] Create inference script with `return_interpretability=True`
- [ ] Map slice indices to z-coordinates
- [ ] Implement slice-level attention visualization
- [ ] Implement 2D slice overlay
- [ ] Generate summary statistics

### ResNet50-3D Grad-CAM
- [ ] Implement 3D Grad-CAM (forward/backward hooks)
- [ ] Handle multi-modal input aggregation
- [ ] Implement slice-by-slice visualization
- [ ] Generate summary heatmaps

### Integration
- [ ] Create unified visualization pipeline
- [ ] Generate comparison visualizations
- [ ] Write documentation
- [ ] Validate on selected patients

---

**End of Technical Decision Brief**

*Generated: 2026-02-10*  
*Based on codebase inspection of: models/, scripts/, ensemble/, results/*

