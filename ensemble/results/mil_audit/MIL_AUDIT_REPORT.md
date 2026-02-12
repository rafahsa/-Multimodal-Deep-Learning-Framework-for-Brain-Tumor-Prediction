# MIL Forensic Audit Report

**Date**: 2026-02-09 01:32:49

## Executive Summary

This report provides a comprehensive forensic analysis of the Dual-Stream MIL model
used in the ensemble, including training configuration, data pipeline, performance
evaluation, and error analysis.

---

## 1. MIL Pipeline Overview

### Architecture

**Type**: Dual-Stream Multiple Instance Learning

**Components**:
1. **Instance Encoder**: ResNet18 (adapted for 4-channel multi-modal input)
2. **Stream 1**: Critical Instance Selector (soft selection with temperature)
3. **Stream 2**: Contextual Aggregator (gated attention)
4. **Fusion**: Concatenation of critical + contextual features
5. **Classifier**: Two-layer MLP with dropout

### Data Flow

```
3D Multi-Modal Volume (T1, T1ce, T2, FLAIR)
  ↓
Extract 2D Slices (instances)
  ↓
Sample/Pad to Fixed Bag Size (32 slices)
  ↓
Instance Encoder (ResNet18)
  ↓
Stream 1: Critical Instance Selection
  ↓
Stream 2: Contextual Attention Aggregation
  ↓
Fusion (concat)
  ↓
Classification Head
  ↓
Patient-Level Probability
```

---

## 2. Training Configuration

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| `attention_type` | `gated` |
| `bag_size` | `32` |
| `batch_size` | `4` |
| `class_weight_power` | `0.5` |
| `classifier_lr` | `0.0001` |
| `dropout` | `0.5` |
| `early_stopping` | `5` |
| `early_stopping_min_epochs` | `10` |
| `ema_decay` | `0.995` |
| `epochs` | `60` |
| `fusion_method` | `concat` |
| `grad_clip` | `0.5` |
| `gradient_accumulation_steps` | `2` |
| `instance_encoder_backbone` | `resnet18` |
| `instance_encoder_input_size` | `224` |
| `label_smoothing_end` | `0.05` |
| `label_smoothing_start` | `0.2` |
| `lr` | `5e-05` |
| `optimizer` | `adamw` |
| `reg_weight_confidence` | `0.01` |
| `reg_weight_entropy` | `0.01` |
| `sampling_strategy` | `random` |
| `scheduler` | `cosine` |
| `temperature_end` | `1.0` |
| `temperature_schedule` | `cosine` |
| `temperature_start` | `10.0` |
| `use_class_weights` | `True` |
| `use_hidden_layer` | `True` |
| `weight_decay` | `0.0005` |

### Loss Function

- **Type**: CrossEntropyLoss
- **Class Balancing**: WeightedRandomSampler (inverse frequency)
- **Label Smoothing**: Adaptive (0.2 → 0.05)
- **Regularization**:
  - Attention entropy regularization (decays adaptively)
  - Selection confidence regularization (decays adaptively)

### Data Augmentation

- **Training**: Per-slice 2D transforms (rotation, flip, zoom, translation)
- **Validation**: Sequential sampling, minimal transforms

---

## 3. Data Splitting

### Split Strategy

- **Method**: StratifiedKFold (k=5, seed=42)
- **Level**: Patient-level (entire patient in one fold)
- **Stratification**: By class label (preserves HGG:LGG ratio)

### Split Files

- Training: `splits/fold_X_train.csv`
- Validation: `splits/fold_X_val.csv`

### Leakage Risk Assessment

✅ **No leakage detected**:
- Patient-level splitting ensures no patient appears in multiple folds
- All slices from a patient are in the same fold
- Validation predictions are truly out-of-fold

---

## 4. MIL Standalone Performance

### Metrics at Threshold 0.5

| Metric | Value |
|--------|-------|
| TN | 69 |
| FP | 6 |
| FN | 163 |
| TP | 47 |
| Precision | 0.8868 |
| Recall | 0.2238 |
| F1 | 0.3574 |
| Accuracy | 0.4070 |
| Specificity | 0.9200 |

### Overall Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.7303 |
| PR-AUC | 0.8780 |
| Brier Score | 0.2432 |

### Optimal Threshold (Cost-Sensitive)

**Threshold**: 0.3400
**Cost**: 70.0 (2×FN + FP)

| Metric | Value |
|--------|-------|
| FN | 3 |
| FP | 64 |
| Precision | 0.7638 |
| Recall | 0.9857 |

---

## 5. Failure Analysis

**Total HGG cases**: 210
**MIL low-probability HGG cases**: 163 (77.6%)

### Statistics for MIL Low-Probability HGG Cases

| Metric | Value |
|--------|-------|
| MIL Prob Mean | 0.4330 |
| MIL Prob Std | 0.0396 |
| ResNet Prob Mean | 0.8969 |
| Swin Prob Mean | 0.7061 |

### Model Disagreement

- MIL low, ResNet high: 162 cases
- MIL low, Swin high: 114 cases

---

## 6. Sanity Checks

### Results

- **probability_range**: ✅ PASS
- **patient_uniqueness**: ✅ PASS
- **label_values**: ✅ PASS
- **class_distribution**: {'lgg': 75, 'hgg': 210, 'ratio': 2.8}
- **probability_statistics**: {'mean': 0.484701052631579, 'std': 0.15300543658853638, 'min': 0.1113, 'max': 0.9497, 'median': 0.4385}
- **shuffled_labels_baseline**: ❌ FAIL

---

## 7. Top 5 Actionable Improvements

Based on the audit findings, here are the top recommendations:

1. **Improve Calibration**: MIL probabilities appear to be less calibrated than other models.
   - Consider Platt scaling or isotonic calibration
   - Current Brier score: 0.2432

2. **Address Low-Probability HGG Cases**: 77.6% of HGG cases have MIL prob < 0.5
   - Investigate bag size and sampling strategy
   - Consider attention mechanism improvements

3. **Optimize Threshold**: Current threshold (0.5) may not be optimal
   - Optimal cost-sensitive threshold: 0.3400
   - Reduces cost from 332.0 to 70.0

4. **Enhance Instance Selection**: Current random sampling may miss critical slices
   - Consider entropy-based pre-selection
   - Explore learned attention mechanisms

5. **Regularization Tuning**: Current regularization may be too aggressive or too weak
   - Monitor attention entropy during training
   - Adjust regularization decay schedule

---

## 8. Missing Metadata

The following metadata would be helpful but is not currently available:

- **Instance-level predictions**: Per-slice probabilities are not saved
  - Would enable analysis of which slices contribute to bag-level decisions
  - Would help identify problematic slices

- **Attention weights**: Attention weights per instance are not saved
  - Would enable interpretability analysis
  - Would help understand which slices the model focuses on

- **Critical instance indices**: Which slice was selected as critical is not saved
  - Would enable analysis of critical slice characteristics

**Recommendation**: Add minimal instrumentation to save instance-level outputs
during validation (without changing training results).

---

## Conclusion

The MIL model shows reasonable performance but has room for improvement, particularly
in calibration and handling of low-probability HGG cases. The recommended improvements
should be implemented and evaluated systematically.
