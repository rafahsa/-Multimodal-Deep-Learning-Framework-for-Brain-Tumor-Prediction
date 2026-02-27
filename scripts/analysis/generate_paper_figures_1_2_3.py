#!/usr/bin/env python3
"""
Generate Figures 1-3 for MICCAI 2026 Paper.

Uses BASELINE ensemble (Full OOF AUC=0.9126) and SwinUNETR-3D (Full OOF AUC=0.9065).
Does NOT use meta_decision or meta_learner_roi_mil.

Figures:
  1. ROC curve (Swin + Baseline ensemble, Full OOF AUC in legend)
  2. Precision-Recall curve (Swin + Baseline ensemble)
  3. Calibration curve / reliability diagram (Swin + Baseline ensemble)
  4. Confusion matrix (Baseline ensemble, threshold 0.41)

Output: reports/figures/figure_1_roc.png, figure_2_pr.png, figure_3_calibration.png,
        figure_4_confusion_matrix.png, figure_4_confusion_matrix.pdf
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
import seaborn as sns

# Paths
DATA_DIR = PROJECT / 'reports/figures/data'
OUTPUT_DIR = PROJECT / 'reports/figures'
BASELINE_CSV = DATA_DIR / 'baseline_ensemble_oof.csv'
SWIN_OOF = PROJECT / 'ensemble/oof_predictions/swinunetr_3d_oof.csv'

# Full OOF AUC values (from forensic verification)
AUC_SWIN = 0.9065
AUC_ENSEMBLE = 0.9126

DPI = 300
FIG_SIZE = (8, 8)


def compute_ece(y_true, y_proba, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
        prop = in_bin.mean()
        if prop > 0:
            acc = y_true[in_bin].mean()
            conf = y_proba[in_bin].mean()
            ece += np.abs(acc - conf) * prop
    return float(ece)


def load_data():
    """Load Swin and baseline ensemble OOF predictions, merged on patient_id."""
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"Run create_baseline_ensemble_csv.py first to create {BASELINE_CSV}"
        )
    if not SWIN_OOF.exists():
        raise FileNotFoundError(f"Swin OOF not found: {SWIN_OOF}")
    
    baseline = pd.read_csv(BASELINE_CSV)
    swin = pd.read_csv(SWIN_OOF)
    
    # Merge on patient_id
    df = baseline.merge(
        swin[['patient_id', 'hgg_prob']],
        on='patient_id',
        how='inner',
        suffixes=('', '_swin')
    )
    df = df.rename(columns={'hgg_prob': 'swin_prob'})
    
    assert len(df) == 285
    return df


def figure_1_roc(df):
    """Figure 1: ROC curve with Swin and Baseline ensemble."""
    y_true = df['label'].values
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # Swin
    fpr_s, tpr_s, _ = roc_curve(y_true, df['swin_prob'])
    ax.plot(fpr_s, tpr_s, linewidth=2.5, color='#2E86AB',
            label=f'SwinUNETR-3D (AUC = {AUC_SWIN:.4f})')
    
    # Baseline ensemble
    fpr_e, tpr_e, _ = roc_curve(y_true, df['ensemble_prob_baseline'])
    ax.plot(fpr_e, tpr_e, linewidth=2.5, color='#A23B72',
            label=f'Ensemble (AUC = {AUC_ENSEMBLE:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves: SwinUNETR-3D vs Ensemble (Full OOF)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / 'figure_1_roc.png'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved {out}")


def figure_2_pr(df):
    """Figure 2: Precision-Recall curve with Swin and Baseline ensemble."""
    y_true = df['label'].values
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # Swin
    prec_s, rec_s, _ = precision_recall_curve(y_true, df['swin_prob'])
    ap_s = average_precision_score(y_true, df['swin_prob'])
    ax.plot(rec_s, prec_s, linewidth=2.5, color='#2E86AB',
            label=f'SwinUNETR-3D (AP = {ap_s:.4f})')
    
    # Baseline ensemble
    prec_e, rec_e, _ = precision_recall_curve(y_true, df['ensemble_prob_baseline'])
    ap_e = average_precision_score(y_true, df['ensemble_prob_baseline'])
    ax.plot(rec_e, prec_e, linewidth=2.5, color='#A23B72',
            label=f'Ensemble (AP = {ap_e:.4f})')
    
    baseline_ap = y_true.mean()
    ax.axhline(y=baseline_ap, color='k', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'No Skill (AP = {baseline_ap:.4f})')
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves: SwinUNETR-3D vs Ensemble (Full OOF)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / 'figure_2_pr.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved {out}")


def figure_3_calibration(df):
    """Figure 3: Calibration curve (reliability diagram) for Swin and Ensemble."""
    y_true = df['label'].values
    n_bins = 10
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # Swin
    frac_s, mean_s = calibration_curve(y_true, df['swin_prob'], n_bins=n_bins, strategy='uniform')
    ece_s = compute_ece(y_true, df['swin_prob'].values, n_bins)
    ax.plot(mean_s, frac_s, 'o-', linewidth=2, markersize=8, color='#2E86AB',
            label=f'SwinUNETR-3D (ECE = {ece_s:.3f})')
    
    # Ensemble
    frac_e, mean_e = calibration_curve(y_true, df['ensemble_prob_baseline'], n_bins=n_bins, strategy='uniform')
    ece_e = compute_ece(y_true, df['ensemble_prob_baseline'].values, n_bins)
    ax.plot(mean_e, frac_e, 's-', linewidth=2, markersize=8, color='#A23B72',
            label=f'Ensemble (ECE = {ece_e:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Curves: SwinUNETR-3D vs Ensemble (Full OOF)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / 'figure_3_calibration.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved {out}")


def figure_4_confusion_matrix():
    """
    Figure 4: Two-panel confusion matrix for Baseline Ensemble (calibrated probs).
    (A) Threshold 0.41 (balanced), (B) Threshold 0.38 (high-sensitivity).
    Evaluation set: held-out threshold selection set (n=86).
    """
    cal_path = DATA_DIR / 'baseline_ensemble_oof_calibrated.csv'
    held_path = DATA_DIR / 'threshold_selection_set_seed42.csv'
    if not cal_path.exists() or not held_path.exists():
        raise FileNotFoundError(
            f"Run create_baseline_calibrated_and_split.py first to create\n"
            f"  {cal_path}\n  {held_path}"
        )
    cal = pd.read_csv(cal_path)
    held = pd.read_csv(held_path)
    df = cal.merge(held, on='patient_id')
    assert len(df) == 86, f"Expected 86, got {len(df)}"

    y_true = df['label'].values
    p_cal = df['ensemble_prob_baseline_calibrated'].values

    def metrics(y_true, y_pred):
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        return tn, fp, fn, tp, prec, rec

    pred41 = (p_cal >= 0.41).astype(int)
    pred38 = (p_cal >= 0.38).astype(int)
    tn41, fp41, fn41, tp41, prec41, rec41 = metrics(y_true, pred41)
    tn38, fp38, fn38, tp38, prec38, rec38 = metrics(y_true, pred38)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)
    for ax, panel, thresh, tn, fp, fn, tp, prec, rec in [
        (axes[0], 'A', 0.41, tn41, fp41, fn41, tp41, prec41, rec41),
        (axes[1], 'B', 0.38, tn38, fp38, fn38, tp38, prec38, rec38),
    ]:
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['LGG (0)', 'HGG (1)'],
            yticklabels=['LGG (0)', 'HGG (1)'],
            cbar_kws={'label': 'Count'},
            linewidths=1, linecolor='gray',
            annot_kws={'size': 16, 'weight': 'bold'},
        )
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_title(
            f'({panel}) Threshold = {thresh}\n'
            f'TN={tn}, FP={fp}, FN={fn}, TP={tp}\n'
            f'Precision={prec:.4f}, Recall={rec:.4f}',
            fontsize=12, fontweight='bold',
        )
    plt.suptitle(
        'Confusion Matrices: Baseline Ensemble (Calibrated Probs, Held-Out n=86)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        out = OUTPUT_DIR / f'figure_4_confusion_matrix.{ext}'
        plt.savefig(out, dpi=DPI if ext == 'png' else None,
                    bbox_inches='tight', facecolor='white')
        print(f"✓ Saved {out}")
    plt.close()
    return (tn41, fp41, fn41, tp41), (tn38, fp38, fn38, tp38)


def main():
    print("Loading Swin + Baseline ensemble OOF data...")
    df = load_data()
    print(f"  {len(df)} samples")
    print(f"  Swin Full OOF AUC (verify): {roc_auc_score(df['label'], df['swin_prob']):.4f}")
    print(f"  Ensemble Full OOF AUC (verify): {roc_auc_score(df['label'], df['ensemble_prob_baseline']):.4f}")
    
    print("\nGenerating Figures 1-4...")
    figure_1_roc(df)
    figure_2_pr(df)
    figure_3_calibration(df)
    m41, m38 = figure_4_confusion_matrix()
    print(f"\nFigure 4 (A) 0.41: TN={m41[0]}, FP={m41[1]}, FN={m41[2]}, TP={m41[3]}")
    print(f"Figure 4 (B) 0.38: TN={m38[0]}, FP={m38[1]}, FN={m38[2]}, TP={m38[3]}")
    print("\nDone.")


if __name__ == '__main__':
    main()
