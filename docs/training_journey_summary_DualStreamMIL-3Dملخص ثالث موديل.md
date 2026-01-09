# Technical Summary: Dual-Stream Multiple Instance Learning (MIL) 3D Training for Brain Tumor Classification
## BraTS 2018 Dataset (HGG vs LGG)

**Model**: Dual-Stream Multiple Instance Learning (MIL) 3D  
**Dataset**: BraTS 2018  
**Task**: Binary classification (High-Grade Glioma vs Low-Grade Glioma)  
**Training Strategy**: 5-fold cross-validation with EMA checkpointing

---

## 1. Model Architecture and Design Rationale

### 1.1 Architecture Overview

The Dual-Stream Multiple Instance Learning (MIL) 3D model was implemented for binary brain tumor grade classification (LGG vs HGG). The model follows a Dual-Stream MIL design, where two feature streams are processed and aggregated using attention-based MIL mechanisms. Under the Multiple Instance Learning paradigm, the model operates by aggregating instance-level representations into bag-level predictions, where each patient is treated as a bag containing multiple instances, and the model produces a single prediction per patient.

### 1.2 Design Rationale

The Multiple Instance Learning approach is suitable for this task because patient-level labels (HGG or LGG) are available, while individual instance-level labels are not. The model learns to identify informative instances within each patient bag and aggregates their features to make patient-level predictions. The Dual-Stream design processes features through two complementary streams that are then combined using attention-based mechanisms, allowing the model to capture both critical instance information and contextual relationships across instances.

---

## 2. Training Configuration

### 2.1 Loss Function and Class Imbalance Handling

The model was trained using **CrossEntropyLoss** as the training objective. Class imbalance was addressed at the data level using a **WeightedRandomSampler**, which ensures balanced batch composition during training by oversampling the minority class.

### 2.2 Model Selection and Checkpointing

**Exponential Moving Average (EMA)** was employed for model checkpointing. Model selection was performed based on validation AUC, with the best EMA checkpoint selected per fold based on the highest validation AUC achieved during training.

### 2.3 Cross-Validation Protocol

Training was conducted using **5-fold cross-validation** to ensure robust performance evaluation. For each fold, the model was trained until the best validation AUC was achieved, and the corresponding EMA checkpoint was saved for final evaluation.

---

## 3. Experimental Results

### 3.1 Per-Fold Performance

Performance was evaluated per fold using the best EMA checkpoint, as determined by validation AUC. The results for each fold were as follows:

**Table 1: Per-fold performance metrics at best validation epoch**

| Fold | Best Epoch | AUC | F1-Score | Accuracy |
|------|------------|-----|----------|----------|
| 0 | 8 | 0.8897 | 0.7904 | 0.8246 |
| 1 | 22 | 0.9825 | 0.8505 | 0.8947 |
| 2 | 14 | 0.9714 | 0.8697 | 0.8947 |
| 3 | 7 | 0.9778 | 0.7738 | 0.7895 |
| 4 | 10 | 0.9651 | 0.8580 | 0.8947 |

Each fold represents a separate training run with distinct training and validation splits, ensuring that the reported metrics reflect the model's performance across different data distributions.

### 3.2 Cross-Validation Summary

The final performance across all folds is summarized as mean ± standard deviation:

- **AUC**: 0.9573 ± 0.0384
- **F1-Score**: 0.8285 ± 0.0433
- **Accuracy**: 0.8596 ± 0.0496

The ± value represents inter-fold variability, which reflects natural heterogeneity in the dataset and differences in validation split composition. This variability is expected in medical imaging datasets where patient characteristics and lesion presentations can vary across folds.

---

## 4. Discussion

The Dual-Stream MIL model achieved strong performance across the 5-fold cross-validation, with a mean AUC of 0.9573. The model demonstrated consistent performance across folds, with F1-scores and accuracy metrics indicating robust classification capability. The attention-based MIL aggregation mechanism successfully integrated instance-level features into patient-level predictions, supporting the suitability of the Multiple Instance Learning approach  for this brain tumor classification task.

---

**Document Information**:
- Model: Dual-Stream Multiple Instance Learning (MIL) 3D
- Training Script: `scripts/training/train_dual_stream_mil.py`
- Results Directory: `results/DualStreamMIL-3D/`
- Training Date: January 2026
- 5-Fold Cross-Validation Complete

