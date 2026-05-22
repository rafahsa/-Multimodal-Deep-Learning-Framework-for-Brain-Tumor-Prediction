import numpy as np


THRESHOLD_BALANCED = 0.41
THRESHOLD_HIGH_SENSITIVITY = 0.38


def baseline_ensemble(
    p_resnet: float, p_swin: float, p_mil: float, metrics: dict
) -> tuple[float, float]:
    """Compute uncalibrated ensemble probability.

    Returns (uncalibrated_probability, raw_logit).
    """
    c = metrics["model_coefficients"]
    intercept = metrics["model_intercept"]
    logit = (
        intercept
        + c["hgg_prob_resnet"] * p_resnet
        + c["hgg_prob_swin"] * p_swin
        + c["hgg_prob_mil"] * p_mil
    )
    p_uncal = float(1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500))))
    return p_uncal, float(logit)


def apply_platt(calibrator: dict | None, p: float) -> float:
    """Apply Platt scaling calibration if available."""
    if calibrator is None or calibrator.get("type") != "platt":
        return p
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    log_odds = np.log(clipped / (1 - clipped))
    return float(
        calibrator["model"].predict_proba(np.array([[log_odds]]))[0, 1]
    )


def classify_at_thresholds(calibrated_p: float) -> dict:
    """Classify at both operating points."""
    return {
        "balanced": {
            "value": THRESHOLD_BALANCED,
            "classification": "HGG" if calibrated_p >= THRESHOLD_BALANCED else "LGG",
            "mode_name": "Balanced Screening",
        },
        "high_sensitivity": {
            "value": THRESHOLD_HIGH_SENSITIVITY,
            "classification": "HGG" if calibrated_p >= THRESHOLD_HIGH_SENSITIVITY else "LGG",
            "mode_name": "High-Sensitivity Triage",
        },
    }
