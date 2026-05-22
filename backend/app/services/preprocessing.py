import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

logger = logging.getLogger("neurograde.preprocessing")


def preprocess_volume(
    vol: np.ndarray, target: tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """Resize to target shape and z-score normalize (brain-masked)."""
    if vol.shape != target:
        sitk_img = sitk.GetImageFromArray(vol)
        old_size = np.array(sitk_img.GetSize())
        old_sp = np.array(sitk_img.GetSpacing())
        new_sp = old_sp * (old_size / np.array(target))
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(tuple(int(x) for x in target))
        resampler.SetOutputSpacing(new_sp.tolist())
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        sitk_img = resampler.Execute(sitk_img)
        vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    mask = vol > 0
    if mask.sum() > 0:
        m, s = vol[mask].mean(), vol[mask].std()
        if s > 1e-8:
            vol = (vol - m) / (s + 1e-8)
    vol[~mask] = 0.0
    return vol


def select_slices_entropy(volume_np: np.ndarray, k: int = 16) -> np.ndarray:
    """Select top-k depth slices by Shannon entropy for MIL bag construction."""
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


def load_and_preprocess_niftis(
    paths: dict[str, Path],
) -> torch.Tensor:
    """Load 4 modality NIfTI files, preprocess, and return (4, 128, 128, 128) tensor."""
    modality_order = ["t1", "t1ce", "t2", "flair"]
    channels = []
    for mod in modality_order:
        file_path = paths[mod]
        logger.info("Loading %s from %s", mod, file_path)
        img = sitk.ReadImage(str(file_path))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
        vol = preprocess_volume(vol)
        channels.append(vol)
    stacked = np.stack(channels, axis=0)  # (4, 128, 128, 128)
    return torch.from_numpy(stacked).float()
