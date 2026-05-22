import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from app.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet50_3d_fast.model import create_resnet50_3d  # noqa: E402
from models.swin_unetr_encoder import create_swin_unetr_classifier  # noqa: E402
from models.dual_stream_mil import create_dual_stream_mil  # noqa: E402

logger = logging.getLogger("neurograde.inference")

MIL_TOP_K = 16

CHECKPOINT_FILENAMES = (
    "best.pt",
    "best_ema.pt",
    "last.pt",
    "model.pt",
    "checkpoint.pt",
)

MODEL_SEARCH_ALIASES: dict[str, tuple[str, ...]] = {
    "ResNet50-3D": ("resnet50-3d", "resnet503d", "resnet50"),
    "SwinUNETR-3D": ("swinunetr-3d", "swinunetr3d", "swinunetr"),
    "DualStreamMIL-3D": ("dualstreammil-3d", "dualstreammil3d", "dualstreammil"),
}

# Python package subdirs inside project models/ — not checkpoint storage
ARCHITECTURE_SUBDIRS = frozenset({"resnet50_3d_fast", "__pycache__"})


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _first_existing_file(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _find_checkpoint_in_results(model_name: str, base_dir: Path, fold: int = 0) -> Path | None:
    """Standard training layout: base_dir/ModelName/runs/fold_N/run_*/checkpoints/*.pt"""
    runs_dir = base_dir / model_name / "runs" / f"fold_{fold}"
    if not runs_dir.is_dir():
        return None
    run_dirs = sorted(runs_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        ckpts_dir = run_dir / "checkpoints"
        if not ckpts_dir.is_dir():
            continue
        found = _first_existing_file(ckpts_dir, CHECKPOINT_FILENAMES)
        if found:
            return found
    return None


def _normalize_token(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def _model_name_in_path(model_name: str, path: Path) -> bool:
    tokens = {_normalize_token(model_name), *(_normalize_token(a) for a in MODEL_SEARCH_ALIASES[model_name])}
    haystack = _normalize_token(str(path))
    return any(token in haystack for token in tokens)


def _is_weight_candidate(model_name: str, path: Path, weights_root: Path) -> bool:
    if path.suffix.lower() not in (".pt", ".pth", ".ckpt"):
        return False
    try:
        rel = path.relative_to(weights_root)
    except ValueError:
        return False
    if rel.parts and rel.parts[0] in ARCHITECTURE_SUBDIRS:
        return False
    return _model_name_in_path(model_name, path)


def _describe_checkpoint_dir(directory: Path) -> str:
    if not directory.is_dir():
        return f"{directory} (missing)"
    weight_files = sorted(
        [*directory.rglob("*.pt"), *directory.rglob("*.pth"), *directory.rglob("*.ckpt")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not weight_files:
        return f"{directory} (exists, but no .pt/.pth/.ckpt files found)"
    shown = weight_files[:12]
    lines = "\n  ".join(str(p.resolve()) for p in shown)
    suffix = f"\n  ... and {len(weight_files) - len(shown)} more" if len(weight_files) > len(shown) else ""
    return f"{directory} — found {len(weight_files)} weight file(s):\n  {lines}{suffix}"


def _pick_best_weight_file(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    for preferred in CHECKPOINT_FILENAMES:
        for path in ranked:
            if path.name == preferred:
                return path.resolve()
    return ranked[0].resolve()


def _find_checkpoint_in_model_dir(model_name: str, weights_dir: Path) -> Path | None:
    """
    Fallback layouts under project models/ folder (weights, not Python architecture code):
      Layout B: models/ResNet50-3D.pt
      Layout A: models/ResNet50-3D/best.pt (and other CHECKPOINT_FILENAMES)
      Layout C: any matching *.pt under models/ (recursive, skips resnet50_3d_fast/)
    """
    if not weights_dir.is_dir():
        return None

    layout_b = (weights_dir / f"{model_name}.pt").resolve()
    if layout_b.is_file():
        return layout_b

    model_subdir = weights_dir / model_name
    if model_subdir.is_dir():
        found = _first_existing_file(model_subdir, CHECKPOINT_FILENAMES)
        if found:
            return found
        checkpoints_sub = model_subdir / "checkpoints"
        if checkpoints_sub.is_dir():
            found = _first_existing_file(checkpoints_sub, CHECKPOINT_FILENAMES)
            if found:
                return found

    recursive_matches = [
        p.resolve()
        for p in [*weights_dir.rglob("*.pt"), *weights_dir.rglob("*.pth"), *weights_dir.rglob("*.ckpt")]
        if _is_weight_candidate(model_name, p, weights_dir)
    ]
    return _pick_best_weight_file(recursive_matches)


def _find_checkpoint(
    model_name: str,
    results_dir: Path,
    fallback_dir: Path | None,
    fold: int = 0,
) -> tuple[Path | None, str | None]:
    """Return (checkpoint path, source label) where source is 'results' or 'models'."""
    ckpt = _find_checkpoint_in_results(model_name, results_dir, fold)
    if ckpt is not None:
        return ckpt, "results"

    if fallback_dir is not None:
        ckpt = _find_checkpoint_in_model_dir(model_name, fallback_dir)
        if ckpt is not None:
            return ckpt, "models"

    return None, None


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _load_resnet(ckpt_path: Path, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = create_resnet50_3d(num_classes=2, in_channels=4, dropout=0.4)
    sd = _strip_module_prefix(ck.get("model_state_dict", ck))
    model.load_state_dict(sd)
    return model.to(device).eval()


def _load_swin(ckpt_path: Path, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = create_swin_unetr_classifier(
        num_classes=2,
        in_channels=4,
        img_size=(128, 128, 128),
        feature_size=48,
        use_checkpoint=False,
        dropout=0.3,
    )
    sd = _strip_module_prefix(ck.get("model_state_dict", ck))
    model.load_state_dict(sd)
    return model.to(device).eval()


def _load_mil(ckpt_path: Path, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = create_dual_stream_mil(
        num_classes=2,
        instance_encoder_backbone="resnet18",
        instance_encoder_input_size=224,
        attention_type="gated",
        fusion_method="concat",
        dropout=0.5,
        use_hidden_layer=True,
    )
    sd = _strip_module_prefix(ck.get("model_state_dict", ck))
    model.load_state_dict(sd)
    return model.to(device).eval()


class ModelRegistry:
    """Loads and holds all three models + meta-learner coefficients."""

    def __init__(self) -> None:
        self.device = _resolve_device(settings.DEVICE)
        self.resnet: nn.Module | None = None
        self.swin: nn.Module | None = None
        self.mil: nn.Module | None = None
        self.metrics: dict | None = None
        self.calibrator: dict | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def can_predict(self) -> bool:
        return self.resnet is not None and self.swin is not None and self.mil is not None

    def _load_model_checkpoint(
        self,
        model_name: str,
        results_dir: Path,
        fallback_dir: Path | None,
        loader,
        attr: str,
    ) -> None:
        ckpt, source = _find_checkpoint(model_name, results_dir, fallback_dir)
        if ckpt is None:
            logger.warning(
                "%s checkpoint not found — skipping\n"
                "  results: %s\n"
                "  models fallback: %s",
                model_name,
                _describe_checkpoint_dir(results_dir),
                _describe_checkpoint_dir(fallback_dir) if fallback_dir else "disabled",
            )
            return
        logger.info(
            "%s checkpoint loaded from [%s]: %s",
            model_name,
            source,
            ckpt,
        )
        setattr(self, attr, loader(ckpt, self.device))

    def load_all(self) -> None:
        results_dir = settings.MODEL_BASE_DIR.resolve()
        fallback_dir = settings.MODEL_FALLBACK_DIR.resolve()
        logger.info(
            "Loading models on %s (results=%s, models fallback=%s)",
            self.device,
            results_dir,
            fallback_dir,
        )

        self._load_model_checkpoint(
            "ResNet50-3D",
            results_dir,
            fallback_dir,
            _load_resnet,
            "resnet",
        )
        self._load_model_checkpoint(
            "SwinUNETR-3D",
            results_dir,
            fallback_dir,
            _load_swin,
            "swin",
        )
        self._load_model_checkpoint(
            "DualStreamMIL-3D",
            results_dir,
            fallback_dir,
            _load_mil,
            "mil",
        )

        metrics_path = settings.METRICS_PATH
        if metrics_path.exists():
            with open(metrics_path) as f:
                self.metrics = json.load(f)
            logger.info("Loaded meta-learner metrics from %s", metrics_path)
        else:
            logger.warning("Meta-learner metrics not found at %s", metrics_path)

        cal_path = settings.CALIBRATOR_PATH
        if cal_path.exists():
            self.calibrator = joblib.load(cal_path)
            logger.info("Loaded Platt calibrator from %s", cal_path)
        else:
            logger.warning("Calibrator not found at %s", cal_path)

        self._loaded = True
        if not self.can_predict:
            logger.warning(
                "Ensemble neural models are NOT ready — predictions will fail until checkpoints "
                "are placed under results/ or models/ (see warnings above)."
            )
        else:
            logger.info("All three neural checkpoints loaded successfully.")

    def predict_resnet(self, vol: torch.Tensor) -> float:
        if self.resnet is None:
            raise RuntimeError("ResNet50-3D not loaded")
        with torch.no_grad(), autocast(enabled=(self.device.type == "cuda")):
            logits = self.resnet(vol.unsqueeze(0).to(self.device))
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].cpu())

    def predict_swin(self, vol: torch.Tensor) -> float:
        if self.swin is None:
            raise RuntimeError("SwinUNETR-3D not loaded")
        with torch.no_grad(), autocast(enabled=(self.device.type == "cuda")):
            logits = self.swin(vol.unsqueeze(0).to(self.device))
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].cpu())

    def predict_mil(self, vol: torch.Tensor) -> float:
        if self.mil is None:
            raise RuntimeError("DualStreamMIL-3D not loaded")
        from app.services.preprocessing import select_slices_entropy

        vol_np = vol.numpy()
        bag_np = select_slices_entropy(vol_np, k=MIL_TOP_K)
        bag = torch.from_numpy(bag_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad(), autocast(enabled=(self.device.type == "cuda")):
            logits = self.mil(bag, temperature=1.0, return_interpretability=False)
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].cpu())

    def get_coefficients(self) -> dict:
        if self.metrics is None:
            raise RuntimeError("Meta-learner metrics not loaded")
        c = self.metrics["model_coefficients"]
        return {
            "resnet": c["hgg_prob_resnet"],
            "swinunetr": c["hgg_prob_swin"],
            "mil": c["hgg_prob_mil"],
            "intercept": self.metrics["model_intercept"],
        }

    def get_device_name(self) -> str:
        return self.device.type

    def get_gpu_name(self) -> str | None:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return None
