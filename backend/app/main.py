import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.inference import ModelRegistry

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("neurograde")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model registry...")
    registry = ModelRegistry()
    registry.load_all()
    app.state.registry = registry
    logger.info("Models ready on device=%s", registry.device)
    yield
    logger.info("Shutting down — releasing models")


app = FastAPI(
    title="NeuroGrade API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BODY = settings.MAX_UPLOAD_MB * 1024 * 1024


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY:
        return JSONResponse(
            status_code=413,
            content={
                "error": "file_too_large",
                "message": f"Total upload size exceeds the {settings.MAX_UPLOAD_MB} MB limit.",
                "max_size_mb": settings.MAX_UPLOAD_MB,
            },
        )
    return await call_next(request)


from app.api.routes import router  # noqa: E402

app.include_router(router, prefix="/api")
