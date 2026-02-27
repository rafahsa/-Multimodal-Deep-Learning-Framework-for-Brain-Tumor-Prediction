#!/usr/bin/env python3
"""
Final Ensemble Inference for MICCAI 2026 Paper

Runs the baseline ensemble (AUC=0.9126) on new patient data.
Uses:
- meta_learner_metrics.json coefficients (baseline formula, NOT joblib)
- Platt calibrator from 2026-02-07_22-29-29_platt_seed42
- Entropy-based MIL slice selection (k=16)
- Thresholds: 0.41 (balanced), 0.38 (high-sensitivity)

Input: Patient folder with 4 modalities (T1, T1ce, T2, FLAIR)
Output: Saves to test/outputs/final_ensemble_inference_results.csv

Usage:
  python scripts/test/run_final_ensemble_inference.py test/DATA_FOR_TEST/UCSF-PDGM-0004
  python scripts/test/run_final_ensemble_inference.py test/DATA_FOR_TEST/UCSF-PDGM-0004 test/DATA_FOR_TEST/UCSF-PDGM-0005
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import SimpleITK as sitk
import joblib
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from models.resnet50_3d_fast.model import create_resnet50_3d
from models.swin_unetr_encoder import create_swin_unetr_classifier
from models.dual_stream_mil import create_dual_stream_mil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths (baseline paper configuration)
RESULTS_DIR = PROJECT / 'results'
METRICS_PATH = PROJECT / 'ensemble/results/meta_learner_metrics.json'
CALIBRATOR_PATH = PROJECT / 'ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/calibrator_platt.joblib'
OUTPUT_CSV = PROJECT / 'test/outputs/final_ensemble_inference_results.csv'

MIL_TOP_K = 16  # Entropy-based slice selection (paper: k=16)


def _select_slices_entropy(volume_np: np.ndarray, k: int = 16) -> np.ndarray:
    """Select top-k slices by Shannon entropy (matches paper)."""
    # volume_np: (4, D, H, W)
    D = volume_np.shape[1]
    slices = np.transpose(volume_np, (1, 0, 2, 3))  # (D, 4, H, W)
    N = slices.shape[0]
    entropies = []
    for i in range(N):
        slice_2d = slices[i].flatten()
        slice_2d = slice_2d[np.isfinite(slice_2d)]
        if len(slice_2d) == 0:
            entropies.append(0.0)
            continue
        smin, smax = slice_2d.min(), slice_2d.max()
        if smax - smin < 1e-10:
            entropies.append(0.0)
            continue
        sn = (slice_2d - smin) / (smax - smin)
        hist, _ = np.histogram(sn, bins=256, range=(0.0, 1.0))
        hist_sum = hist.sum()
        if hist_sum == 0:
            entropies.append(0.0)
            continue
        probs = hist[hist > 0] / hist_sum
        entropies.append(float(-np.sum(probs * np.log2(probs + 1e-12))))
    entropies = np.nan_to_num(np.array(entropies), nan=0.0, posinf=0.0, neginf=0.0)
    top_idx = np.argsort(entropies)[-k:]
    return slices[sorted(top_idx)]


def _preprocess_volume(vol: np.ndarray, target: tuple = (128, 128, 128)) -> np.ndarray:
    """Resize and z-score normalize to match training."""
    if vol.shape != target:
        sitk_img = sitk.GetImageFromArray(vol)
        old_size = np.array(sitk_img.GetSize())
        old_sp = np.array(sitk_img.GetSpacing())
        new_sp = old_sp * (old_size / np.array(target))
        r = sitk.ResampleImageFilter()
        r.SetSize(tuple(int(x) for x in target))
        r.SetOutputSpacing(new_sp.tolist())
        r.SetOutputOrigin(sitk_img.GetOrigin())
        r.SetOutputDirection(sitk_img.GetDirection())
        r.SetInterpolator(sitk.sitkLinear)
        sitk_img = r.Execute(sitk_img)
        vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    mask = vol > 0
    if mask.sum() > 0:
        m, s = vol[mask].mean(), vol[mask].std()
        if s > 1e-8:
            vol = (vol - m) / (s + 1e-8)
    vol[~mask] = 0.0
    return vol


def _load_patient(test_dir: Path, patient_id: str) -> torch.Tensor:
    """Load 4 modalities, preprocess, return (4,128,128,128) tensor."""
    pat_dir = test_dir / patient_id
    if not pat_dir.exists():
        raise FileNotFoundError(f"Patient dir not found: {pat_dir}")
    order = [('T1', ['T1']), ('T1ce', ['T1ce', 'T1c']), ('T2', ['T2']), ('FLAIR', ['FLAIR'])]
    chans = []
    for name, aliases in order:
        found = None
        for a in aliases:
            for ext in ['.nii.gz', '.nii']:
                p = pat_dir / f"{patient_id}_{a}{ext}"
                if p.exists():
                    found = p
                    break
            if found:
                break
        if not found:
            raise FileNotFoundError(f"Modality {name} not found in {pat_dir}")
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(found))).astype(np.float32)
        chans.append(_preprocess_volume(arr))
    return torch.from_numpy(np.stack(chans, axis=0)).float()


def _find_checkpoint(model_name: str, fold: int = 0) -> Path:
    """Find first valid checkpoint in any run directory (best.pt > best_ema.pt > last.pt)."""
    base = RESULTS_DIR / model_name / 'runs' / f'fold_{fold}'
    if not base.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {base}\n"
            f"Train models first (e.g. train_resnet50_3d.py, train_swin_unetr_3d.py, train_dual_stream_mil.py).\n"
            f"Expected: results/{model_name}/runs/fold_{fold}/run_*/checkpoints/best.pt or best_ema.pt"
        )
    runs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith('run_')],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No run directories in {base}")
    ckpt_names = ['best.pt', 'best_ema.pt', 'last.pt']
    for run_dir in runs:
        ckpt_dir = run_dir / 'checkpoints'
        if not ckpt_dir.exists():
            continue
        for name in ckpt_names:
            p = ckpt_dir / name
            if p.exists():
                logger.info(f"Using checkpoint for {model_name}: {run_dir.name}/checkpoints/{name}")
                return p
    raise FileNotFoundError(
        f"No checkpoint (.pt) found in any run directory for {model_name} under fold_{fold}."
    )


def _load_resnet(ckpt: Path, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = create_resnet50_3d(num_classes=2, in_channels=4, dropout=0.4)
    sd = ck.get('model_state_dict', ck)
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.to(device).eval()


def _load_swin(ckpt: Path, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = create_swin_unetr_classifier(num_classes=2, in_channels=4, img_size=(128, 128, 128),
                                      feature_size=48, use_checkpoint=False, dropout=0.3)
    sd = ck.get('model_state_dict', ck)
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.to(device).eval()


def _load_mil(ckpt: Path, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = create_dual_stream_mil(num_classes=2, instance_encoder_backbone='resnet18',
                               instance_encoder_input_size=224, attention_type='gated',
                               fusion_method='concat', dropout=0.5, use_hidden_layer=True)
    sd = ck.get('model_state_dict', ck)
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.to(device).eval()


def _predict_resnet(model: nn.Module, vol: torch.Tensor, device: torch.device) -> float:
    with torch.no_grad(), autocast():
        logits = model(vol.unsqueeze(0).to(device))
        return float(torch.softmax(logits, dim=1)[0, 1].cpu().item())


def _predict_swin(model: nn.Module, vol: torch.Tensor, device: torch.device) -> float:
    with torch.no_grad(), autocast():
        logits = model(vol.unsqueeze(0).to(device))
        return float(torch.softmax(logits, dim=1)[0, 1].cpu().item())


def _predict_mil_entropy(model: nn.Module, vol: torch.Tensor, device: torch.device, k: int = 16) -> float:
    """MIL with entropy-based slice selection (k=16)."""
    v = vol.cpu().numpy()
    bag = _select_slices_entropy(v, k=k)
    bag = torch.from_numpy(bag).float().unsqueeze(0).to(device)
    with torch.no_grad(), autocast():
        out = model(bag, temperature=1.0, return_interpretability=False)
        logits = out[0] if isinstance(out, tuple) else out
        return float(torch.softmax(logits, dim=1)[0, 1].cpu().item())


def _baseline_ensemble(p_resnet: float, p_swin: float, p_mil: float, metrics: dict) -> float:
    """Baseline ensemble: sigmoid(intercept + coefs). Same formula as AUC=0.9126."""
    c = metrics['model_coefficients']
    intercept = metrics['model_intercept']
    logit = intercept + c['hgg_prob_resnet'] * p_resnet + c['hgg_prob_swin'] * p_swin + c['hgg_prob_mil'] * p_mil
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500))))


def _apply_platt(calibrator: dict, p: float) -> float:
    if calibrator.get('type') != 'platt':
        return p
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    log_odds = np.log(clipped / (1 - clipped))
    return float(calibrator['model'].predict_proba(np.array([[log_odds]]))[0, 1])


def run_inference(patient_paths: list, device: torch.device, dry_run: bool = False) -> list:
    if dry_run:
        logger.info("DRY RUN: checking paths and imports only")
        if not METRICS_PATH.exists():
            raise FileNotFoundError(f"Metrics not found: {METRICS_PATH}")
        if not CALIBRATOR_PATH.exists():
            raise FileNotFoundError(f"Calibrator not found: {CALIBRATOR_PATH}")
        _find_checkpoint('ResNet50-3D')
        _find_checkpoint('SwinUNETR-3D')
        _find_checkpoint('DualStreamMIL-3D')
        logger.info("Dry run OK: all paths resolve")
        return []

    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    calibrator = joblib.load(CALIBRATOR_PATH)

    resnet_ckpt = _find_checkpoint('ResNet50-3D')
    swin_ckpt = _find_checkpoint('SwinUNETR-3D')
    mil_ckpt = _find_checkpoint('DualStreamMIL-3D')
    resnet = _load_resnet(resnet_ckpt, device)
    swin = _load_swin(swin_ckpt, device)
    mil = _load_mil(mil_ckpt, device)

    results = []
    for pp in patient_paths:
        pp = Path(pp)
        patient_id = pp.name
        test_dir = pp.parent
        logger.info(f"Processing {patient_id}")
        vol = _load_patient(test_dir, patient_id)
        p_resnet = _predict_resnet(resnet, vol, device)
        p_swin = _predict_swin(swin, vol, device)
        p_mil = _predict_mil_entropy(mil, vol, device, k=MIL_TOP_K)
        ensemble_uncal = _baseline_ensemble(p_resnet, p_swin, p_mil, metrics)
        ensemble_cal = _apply_platt(calibrator, ensemble_uncal)
        pred_041 = int(ensemble_cal >= 0.41)
        pred_038 = int(ensemble_cal >= 0.38)
        results.append({
            'patient_id': patient_id,
            'p_hgg_resnet50_3d': p_resnet,
            'p_hgg_swinunetr_3d': p_swin,
            'p_hgg_mil_entropy': p_mil,
            'ensemble_prob_baseline_uncalibrated': ensemble_uncal,
            'ensemble_prob_baseline_calibrated': ensemble_cal,
            'pred_balanced_tau_0_41': pred_041,
            'pred_high_sens_tau_0_38': pred_038,
        })
    return results


def main():
    ap = argparse.ArgumentParser(description='Final ensemble inference (baseline, paper config)')
    ap.add_argument('paths', nargs='+', help='Patient folder path(s), e.g. test/DATA_FOR_TEST/UCSF-PDGM-0004')
    ap.add_argument('--dry-run', action='store_true', help='Only check paths/imports')
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = ap.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu')
    paths = [Path(p) for p in args.paths]
    for p in paths:
        if not p.exists():
            logger.error(f"Path not found: {p}")
            sys.exit(1)

    results = run_inference(paths, device, dry_run=args.dry_run)
    if not results:
        return

    df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=['patient_id'], keep='last')
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved to {OUTPUT_CSV}")
    for r in results:
        logger.info(f"  {r['patient_id']}: ensemble_cal={r['ensemble_prob_baseline_calibrated']:.4f}, "
                    f"pred@0.41={r['pred_balanced_tau_0_41']}, pred@0.38={r['pred_high_sens_tau_0_38']}")


if __name__ == '__main__':
    main()
