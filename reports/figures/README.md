# Paper Figures (MICCAI 2026)

Main figures generated from **baseline meta-learner** and **SwinUNETR-3D** Full OOF predictions only.
Do NOT use meta_decision or meta_learner_roi_mil for these figures.

## Data Sources

| File | Source | Full OOF AUC |
|------|--------|--------------|
| `data/baseline_ensemble_oof.csv` | `merged_oof_predictions_backup_20260209_233113.csv` + meta_learner_metrics.json coefficients | 0.9126 |
| Swin probs | `ensemble/oof_predictions/swinunetr_3d_oof.csv` | 0.9065 |

## Figures

- **figure_1_roc.png**: ROC curves (Swin + Ensemble), legend shows Full OOF AUC 0.9065 and 0.9126
- **figure_2_pr.png**: Precision-Recall curves (Swin + Ensemble)
- **figure_3_calibration.png**: Calibration curves (reliability diagram) for Swin and Ensemble
- **figure_4_confusion_matrix.png/.pdf**: Two-panel confusion matrix (A) 0.41 balanced, (B) 0.38 high-sensitivity.
  Uses calibrated probs on held-out threshold selection set (n=86).

## Regenerating

```bash
python scripts/analysis/create_baseline_ensemble_csv.py
python scripts/analysis/create_baseline_calibrated_and_split.py
python scripts/analysis/generate_paper_figures_1_2_3.py
```
