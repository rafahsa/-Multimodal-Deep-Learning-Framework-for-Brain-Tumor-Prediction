# Nested Cross-Validation Visualizations

**Generated**: 2026-02-09 00:12:27

## Important Note

**ALL visualizations in this directory are based STRICTLY on nested cross-validation results.**

- ✅ Metrics computed on **outer-test folds only** (never seen during training)
- ✅ Results aggregated across **5 outer folds** (mean ± std)
- ✅ **NO optimistic results**, **NO full OOF evaluation**
- ✅ Suitable for **academic publication** and **medical justification**

---

## Generated Plots

### 1. `fn_fp_tradeoff_nested_cv.png`
**FN-FP Trade-off (per outer fold)**

- Scatter plot showing FN vs FP for each outer fold
- Each point = one outer fold's test performance
- Mean ± std annotated for each meta-learner
- **Purpose**: Show realistic clinical trade-off under true generalization

### 2. `cost_distribution_nested_cv.png`
**Cost Distribution Across Folds**

- Boxplot showing cost (2×FN + FP) distribution
- One distribution per meta-learner
- **Purpose**: Demonstrate stability and robustness

### 3. `recall_vs_precision_nested_cv.png`
**Recall vs Precision (Nested CV)**

- Scatter plot with error bars (mean ± std)
- Each point = outer fold
- **Purpose**: Visualize sensitivity–specificity balance under strict evaluation

### 4. `per_fold_confusion_summary_nested_cv.png`
**Per-Fold Confusion Matrix Summary**

- Bar charts showing FN and FP per fold
- Worst-case FN fold highlighted
- **Purpose**: Medical safety transparency

### 5. `meta_learner_comparison_nested_cv.png`
**Meta-Learner Comparison Summary**

- Bar plot with error bars (mean ± std)
- Metrics: FN, FP, Cost, Recall, Precision
- **Purpose**: Final model comparison for paper

---

## Results Summary


### LogisticRegression

- **FN**: 4.20 ± 2.04 (range: [1, 7])
- **FP**: 6.40 ± 2.73
- **Cost**: 14.80 ± 2.79
- **Recall**: 0.9000 ± 0.0486
- **Precision**: 0.8595 ± 0.0479


### XGBoost

- **FN**: 5.20 ± 0.98 (range: [4, 7])
- **FP**: 8.20 ± 2.79
- **Cost**: 18.60 ± 3.01
- **Recall**: 0.8762 ± 0.0233
- **Precision**: 0.8211 ± 0.0502


---

## Scientific Safeguards

All plots are clearly labeled with:
- "Nested Cross-Validation (Outer-Test Only)"
- Footnote: "Results reflect true generalization performance under patient-level nested CV."

These visualizations are:
- ✅ **Honest**: No optimistic bias
- ✅ **Non-optimistic**: Reflect true generalization
- ✅ **Defensible**: Suitable for thesis/journal paper
- ✅ **Strict**: Based only on nested CV outer-test results

---

## Comparison with Previous Optimistic Results

**Previous (Optimistic)**: FN=0, FP≈1, Cost≈1.0
- ❌ Evaluated on same data used for training
- ❌ Data leakage present
- ❌ Not suitable for publication

**Nested CV (Realistic)**: FN≈4-5, FP≈6-8, Cost≈15-19
- ✅ Truly independent test set
- ✅ No data leakage
- ✅ Publication-ready

The nested CV results represent **realistic, trustworthy performance** suitable for medical decision-making.
