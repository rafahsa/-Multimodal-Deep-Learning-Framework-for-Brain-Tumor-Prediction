# Consistency Audit Report
## MICCAI 2026 Documents vs Figures/Data

**Date**: 2026-02-23  
**Scope**: 5 documents, figures, and data sources

---

## Checklist Results

### A) Final System Definition

| Item | CONSOLIDATED | PAPER_READY | INTERPRETABILITY | CONTRIBUTION | STRATEGIC |
|------|--------------|-------------|------------------|--------------|-----------|
| Ensemble = baseline LR (meta_learner_metrics.json) | ✅ | ✅ | N/A | ✅ | ✅ |
| Base models: ResNet50, SwinUNETR, DualStreamMIL entropy-based | ✅ | ✅ | ✅ | ✅ | ✅ |
| Calibration: Platt | ✅ | ✅ | N/A | ✅ | N/A |
| Thresholds: 0.41 (balanced), 0.38 (high-sens) | ✅ | ✅ | N/A | ✅ | N/A |
| ROI-MIL = interpretability only, NOT final | ✅ | N/A | ✅ | ✅ | ✅ |

**A) PASS** — All documents consistent.

### B) Figure-Number Consistency

| Figure | Expected | Source Verified |
|--------|----------|-----------------|
| ROC (Fig 1) | Full OOF n=285, AUC Swin=0.9065, Ensemble=0.9126 | reports/figures/figure_1_roc.png, baseline_ensemble_oof.csv |
| Confusion (Fig 4) | Held-out n=86, calibrated, (A) TN=19 FP=4 FN=4 TP=59, (B) TN=17 FP=6 FN=3 TP=60 | reports/figures/figure_4, baseline_ensemble_oof_calibrated.csv, threshold_selection_set_seed42.csv |

**B) PASS** — Figures match data.

### C) Full OOF AUC 0.9126

| Document | Mentions 0.9126? | Tied to baseline? |
|----------|------------------|-------------------|
| CONSOLIDATED | ✅ "Full OOF (Threshold 0.5): AUC 0.9126" | ✅ baseline ensemble |
| PAPER_READY | ❌ (uses 0.9114 ± 0.0423 in tables) | N/A |
| Others | ❌ | N/A |

**C) PASS** — 0.9126 in CONSOLIDATED is baseline. PAPER_READY correctly uses 0.9114 ± 0.0423 for tables; ROC figure shows 0.9126 (correct).

### D) Mean ± Std 0.9114 ± 0.0423

| Document | In tables? | In ROC legend? |
|----------|------------|----------------|
| All | ✅ | ❌ (ROC shows Full OOF 0.9126) |

**D) PASS** — Tables use mean ± std; ROC legend uses Full OOF AUC.

### E) Contradictions

**None found.** No minimal edits required.

---

## Per-File Summary

| File | Status | Notes |
|------|--------|-------|
| MICCAI_2026_CONSOLIDATED_EXPERIMENTAL_DESIGN.md | PASS | Full system definition, 0.9126, thresholds, ROI-MIL interpretability only |
| MICCAI_2026_PAPER_READY_RESULTS.md | PASS | Tables 0.9114±0.0423, Fig 1 ROC clarification (n=285), calibration 0.41/0.38 |
| MICCAI_2026_INTERPRETABILITY_AND_QUALITATIVE_ANALYSIS.md | PASS | Entropy-based final, ROI-MIL validation only |
| MICCAI_2026_CONTRIBUTION_STATEMENT.md | PASS | 0.9114 vs 0.9140, thresholds, entropy-based MIL |
| STRATEGIC_REFRAMING_SUMMARY.md | PASS | Summary of prior edits, no metrics |

---

**Conclusion**: All 5 documents pass the consistency audit. No edits required.
