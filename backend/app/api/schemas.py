from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ModelProbabilities(BaseModel):
    resnet: float = Field(..., ge=0.0, le=1.0)
    swinunetr: float = Field(..., ge=0.0, le=1.0)
    mil: float = Field(..., ge=0.0, le=1.0)


class MetaLearnerCoefficients(BaseModel):
    resnet: float
    swinunetr: float
    mil: float
    intercept: float


class ThresholdResult(BaseModel):
    value: float
    classification: str
    mode_name: str


class Thresholds(BaseModel):
    balanced: ThresholdResult
    high_sensitivity: ThresholdResult


class PredictionResponse(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_label: str
    calibrated_probability: float = Field(..., ge=0.0, le=1.0)
    uncalibrated_probability: float = Field(..., ge=0.0, le=1.0)
    model_probabilities: ModelProbabilities
    ensemble_logit: float
    meta_learner_coefficients: MetaLearnerCoefficients
    thresholds: Thresholds
    processing_duration_ms: int = Field(..., ge=0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    device_used: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    suggestion: str | None = None
    field: str | None = None
    max_size_mb: int | None = None
    actual_size_mb: int | None = None
    retry_after_seconds: int | None = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    gpu_name: str | None = None
    version: str = "1.0.0"
    message: str | None = None
