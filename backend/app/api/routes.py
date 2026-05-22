from __future__ import annotations

import logging
import time

import torch
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.api.schemas import (
    ErrorResponse,
    HealthResponse,
    MetaLearnerCoefficients,
    ModelProbabilities,
    PredictionResponse,
)
from app.services.calibration import apply_platt, baseline_ensemble, classify_at_thresholds
from app.services.preprocessing import load_and_preprocess_niftis
from app.utils.file_handling import (
    cleanup_tempdir,
    save_upload_to_tempdir,
    validate_nifti_extension,
)

logger = logging.getLogger("neurograde.api")
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
    patient_label: str | None = Form(None),
):
    registry = request.app.state.registry

    if not registry.is_loaded:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="models_not_ready",
                message="The prediction service is still loading models. Please wait and try again in 30 seconds.",
                retry_after_seconds=30,
            ).model_dump(),
        )

    if not registry.can_predict:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="checkpoints_missing",
                message=(
                    "Neural network checkpoints were not found. The API cannot run inference "
                    "until ResNet50-3D, SwinUNETR-3D, and DualStreamMIL-3D weights are available."
                ),
                suggestion=(
                    "Place checkpoints under models/ResNet50-3D/best.pt (and Swin, MIL), or "
                    "models/ResNet50-3D.pt, or the full results/.../checkpoints/ tree. "
                    "Restart the backend and check startup logs for "
                    "'checkpoint loaded from [models]' lines."
                ),
            ).model_dump(),
        )

    files = {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair}
    for modality, upload in files.items():
        if not upload.filename or not validate_nifti_extension(upload.filename):
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error="invalid_file",
                    message=f"File '{upload.filename or 'unknown'}' is not a valid NIfTI volume.",
                    field=modality,
                    suggestion="Ensure the file is a valid NIfTI (.nii or .nii.gz) brain MRI volume. "
                    "Files exported from FreeSurfer, FSL, or 3D Slicer are typically compatible.",
                ).model_dump(),
            )

    tmpdir = None
    try:
        start = time.perf_counter()

        tmpdir, paths = await save_upload_to_tempdir(files)

        vol = load_and_preprocess_niftis(paths)

        p_resnet = registry.predict_resnet(vol)
        p_swin = registry.predict_swin(vol)
        p_mil = registry.predict_mil(vol)

        p_uncal, logit = baseline_ensemble(p_resnet, p_swin, p_mil, registry.metrics)
        p_cal = apply_platt(registry.calibrator, p_uncal)

        thresholds = classify_at_thresholds(p_cal)
        coefficients = registry.get_coefficients()

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        label = patient_label
        if not label:
            base = (t1.filename or "scan").rsplit(".", 1)[0]
            if base.endswith("_T1"):
                base = base[:-3]
            label = base

        return PredictionResponse(
            patient_label=label,
            calibrated_probability=round(p_cal, 6),
            uncalibrated_probability=round(p_uncal, 6),
            model_probabilities=ModelProbabilities(
                resnet=round(p_resnet, 6),
                swinunetr=round(p_swin, 6),
                mil=round(p_mil, 6),
            ),
            ensemble_logit=round(logit, 6),
            meta_learner_coefficients=MetaLearnerCoefficients(**coefficients),
            thresholds=thresholds,
            processing_duration_ms=elapsed_ms,
            device_used=registry.get_device_name(),
        )

    except Exception as exc:
        logger.exception("Inference failed")
        import SimpleITK as sitk  # noqa: F811

        if "sitk" in type(exc).__module__ or "ITK" in str(exc):
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error="invalid_file",
                    message=f"One of the uploaded files could not be parsed as a NIfTI image: {exc}",
                    suggestion="Ensure all uploaded files are valid NIfTI (.nii or .nii.gz) brain MRI volumes.",
                ).model_dump(),
            )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="inference_error",
                message=f"Model inference failed: {exc}",
                suggestion="Try again. If the error persists, ensure the uploaded files are standard brain MRI volumes with typical dimensions.",
            ).model_dump(),
        )
    finally:
        if tmpdir:
            cleanup_tempdir(tmpdir)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    registry = request.app.state.registry

    if not registry.is_loaded:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="loading",
                models_loaded=False,
                device="unknown",
                message="Models are still loading. Estimated time: 30 seconds.",
            ).model_dump(),
        )

    if not registry.can_predict:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="degraded",
                models_loaded=False,
                device=registry.get_device_name(),
                message=(
                    "Checkpoints missing for one or more base models. "
                    "Place .pt files under models/ or results/ and restart."
                ),
            ).model_dump(),
        )

    return HealthResponse(
        status="healthy",
        models_loaded=True,
        device=registry.get_device_name(),
        gpu_name=registry.get_gpu_name(),
    )
