import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    DEVICE: str = os.getenv("DEVICE", "auto")
    MODEL_BASE_DIR: Path = Path(os.getenv("MODEL_BASE_DIR", str(PROJECT_ROOT / "results")))
    MODEL_FALLBACK_DIR: Path = Path(
        os.getenv("MODEL_FALLBACK_DIR", str(PROJECT_ROOT / "models"))
    )
    METRICS_PATH: Path = Path(
        os.getenv(
            "METRICS_PATH",
            str(PROJECT_ROOT / "ensemble" / "results" / "meta_learner_metrics.json"),
        )
    )
    CALIBRATOR_PATH: Path = Path(
        os.getenv(
            "CALIBRATOR_PATH",
            str(
                PROJECT_ROOT
                / "ensemble"
                / "results"
                / "calibration"
                / "2026-02-07_22-29-29_platt_seed42"
                / "calibrator_platt.joblib"
            ),
        )
    )
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    ]
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "500"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
