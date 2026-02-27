#!/usr/bin/env python3
"""
Forensic AUC Verification - READ ONLY.
Recomputes metrics from existing files. No modifications.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def main():
    print("="*80)
    print("FORENSIC AUC VERIFICATION (READ-ONLY)")
    print("="*80)

    # -------------------------------------------------------------------------
    # PART 1: Ensemble Full OOF AUC
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 1: ENSEMBLE FULL OOF AUC")
    print("="*80)

    # 1a) From meta_decision_predictions.csv (meta_decision pipeline - Swin1 meta-layer)
    meta_dec_path = PROJECT / "ensemble/results/meta_decision/meta_decision_predictions.csv"
    if meta_dec_path.exists():
        df_meta_dec = pd.read_csv(meta_dec_path)
        print(f"\n1a) meta_decision_predictions.csv: {len(df_meta_dec)} rows")
        print(f"    Columns: {list(df_meta_dec.columns)}")
        y_true = df_meta_dec["label"].values
        y_score = df_meta_dec["meta_prob"].values
        auc_as_is = roc_auc_score(y_true, y_score)
        auc_flipped = roc_auc_score(1 - y_true, y_score)
        print(f"    label distribution: 0={np.sum(y_true==0)}, 1={np.sum(y_true==1)} (HGG=1)")
        print(f"    AUC (label as-is):   {auc_as_is:.10f}")
        print(f"    AUC (label flipped): {auc_flipped:.10f}")
        print(f"    -> meta_decision pipeline Full OOF AUC: {auc_as_is:.6f}")
    else:
        print("    meta_decision_predictions.csv NOT FOUND")

    # 1b) From merged_oof + meta-learner coefficients (baseline meta-learner -> 0.9126)
    merged_path = PROJECT / "ensemble/oof_predictions/merged_oof_predictions.csv"
    metrics_path = PROJECT / "ensemble/results/meta_learner_metrics.json"
    if merged_path.exists() and metrics_path.exists():
        import json
        df_merged = pd.read_csv(merged_path)
        with open(metrics_path) as f:
            metrics = json.load(f)
        coef = metrics["model_coefficients"]
        # JSON keys: hgg_prob_resnet, hgg_prob_swin, hgg_prob_mil (merged uses mil_prob)
        feat_cols = ["hgg_prob_resnet", "hgg_prob_swin", "mil_prob"]
        intercept = metrics["model_intercept"]
        X = df_merged[feat_cols].values
        y = df_merged["label"].values
        c1 = coef.get("hgg_prob_resnet")
        c2 = coef.get("hgg_prob_swin")
        c3 = coef.get("hgg_prob_mil") or coef.get("mil_prob")
        coef_vec = np.array([c1, c2, c3])
        meta_prob = sigmoid(intercept + X @ coef_vec)
        auc_merged = roc_auc_score(y, meta_prob)
        print(f"\n1b) merged_oof + meta_learner_metrics.json (baseline meta-learner):")
        print(f"    n_samples: {len(df_merged)}, HGG=1: {np.sum(y==1)}, LGG=0: {np.sum(y==0)}")
        print(f"    Reported auc_roc in JSON: {metrics['auc_roc']:.10f}")
        print(f"    Recomputed AUC:           {auc_merged:.10f}")
        print(f"    -> MATCH: {np.isclose(metrics['auc_roc'], auc_merged)}")
    else:
        print("    merged_oof or meta_learner_metrics.json NOT FOUND")

    # -------------------------------------------------------------------------
    # PART 2: Origin of 0.9126 (already identified)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 2: ORIGIN OF 0.9126")
    print("="*80)
    print("  Source: ensemble/results/meta_learner_metrics.json")
    print("  Script: scripts/ensemble/train_meta_learner.py")
    print("  Data:   merged_oof_predictions.csv (hgg_prob_resnet, hgg_prob_swin, mil_prob)")
    print("  Eval:   Full OOF (all 285 samples), in-sample AUC on meta-learner predictions")

    # -------------------------------------------------------------------------
    # PART 3: 5-fold mean ± std for Ensemble
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 3: ENSEMBLE 5-FOLD MEAN ± STD (0.9114 ± 0.0423 claim)")
    print("="*80)

    # Need predictions per fold. Use merged + meta-learner to get per-fold AUC.
    if merged_path.exists() and metrics_path.exists():
        df = df_merged.copy()
        df["meta_prob"] = meta_prob
        fold_aucs = []
        for k in sorted(df["fold"].unique()):
            sub = df[df["fold"] == k]
            a = roc_auc_score(sub["label"], sub["meta_prob"])
            fold_aucs.append(a)
            print(f"  Fold {k}: AUC = {a:.6f}")
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"  Mean ± std: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  Claimed:    0.9114 ± 0.0423")
        print(f"  Match: mean={np.isclose(mean_auc, 0.9114)}, std={np.isclose(std_auc, 0.0423)}")

    # Also from meta_decision_predictions
    if meta_dec_path.exists():
        df_md = pd.read_csv(meta_dec_path)
        fold_aucs_md = []
        for k in sorted(df_md["fold"].unique()):
            sub = df_md[df_md["fold"] == k]
            a = roc_auc_score(sub["label"], sub["meta_prob"])
            fold_aucs_md.append(a)
        print(f"\n  (meta_decision pipeline) Mean ± std: {np.mean(fold_aucs_md):.4f} ± {np.std(fold_aucs_md):.4f}")

    # -------------------------------------------------------------------------
    # PART 4: Swin Full OOF AUC and per-fold
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 4: SWIN FULL OOF AUC (~0.9065 check, 0.9140 ± 0.0414 claim)")
    print("="*80)

    swin_path = PROJECT / "ensemble/oof_predictions/swinunetr_3d_oof.csv"
    if swin_path.exists():
        df_swin = pd.read_csv(swin_path)
        print(f"  File: {swin_path.name}, columns: {list(df_swin.columns)}")
        y_true = df_swin["label"].values
        y_score = df_swin["hgg_prob"].values
        full_auc = roc_auc_score(y_true, y_score)
        print(f"  Full OOF AUC: {full_auc:.10f}")
        print(f"  ROC figure ~0.9065: diff = {full_auc - 0.9065:.4f}")
        fold_aucs_swin = []
        for k in sorted(df_swin["fold"].unique()):
            sub = df_swin[df_swin["fold"] == k]
            a = roc_auc_score(sub["label"], sub["hgg_prob"])
            fold_aucs_swin.append(a)
            print(f"  Fold {k}: AUC = {a:.6f}")
        mean_s = np.mean(fold_aucs_swin)
        std_s = np.std(fold_aucs_swin)
        print(f"  5-fold mean ± std: {mean_s:.4f} ± {std_s:.4f}")
        print(f"  Claimed:          0.9140 ± 0.0414")
        print(f"  Match: mean={np.isclose(mean_s, 0.9140)}, std={np.isclose(std_s, 0.0414)}")

    # -------------------------------------------------------------------------
    # PART 5: meta_learner_roi_mil (used by ROC figures)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 5: META_LEARNER_ROI_MIL (ROC figure source)")
    print("="*80)
    roi_pred = PROJECT / "ensemble/results/meta_learner_roi_mil/predictions.csv"
    roi_metrics = PROJECT / "ensemble/results/meta_learner_roi_mil/meta_learner_metrics.json"
    if roi_pred.exists() and roi_metrics.exists():
        import json
        df_roi = pd.read_csv(roi_pred)
        with open(roi_metrics) as f:
            m = json.load(f)
        auc_roi = roc_auc_score(df_roi["true_label"], df_roi["predicted_probability"])
        print(f"  predictions.csv Full OOF AUC (recomputed): {auc_roi:.10f}")
        print(f"  meta_learner_metrics.json auc_roc:         {m['auc_roc']:.10f}")
        fold_aucs_roi = []
        for k in sorted(df_roi["fold"].unique()):
            sub = df_roi[df_roi["fold"] == k]
            a = roc_auc_score(sub["true_label"], sub["predicted_probability"])
            fold_aucs_roi.append(a)
        print(f"  5-fold mean ± std: {np.mean(fold_aucs_roi):.4f} ± {np.std(fold_aucs_roi):.4f}")

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
