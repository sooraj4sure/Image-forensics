"""
FastAPI serving layer (brief §11):
    GET /health
    POST /predict  — prediction + confidence only, no Grad-CAM (fast path)
    POST /analyze  — full forensic report incl. Grad-CAM heatmap (base64 PNG)

Model is loaded ONCE at startup via FastAPI's lifespan context, not
per-request. Uploads are validated for type/size (config.serving) before
being handed to the pipeline.
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from src.config import load_config
from src.inference.pipeline import ForensicsPipeline

_pipeline: ForensicsPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    cfg = load_config()
    try:
        _pipeline = ForensicsPipeline(config_name="full_fusion", cfg=cfg)
        print("Model loaded successfully at startup.")
    except FileNotFoundError as e:
        # Let the app start (so /health still works, useful in CI/dev before
        # a model is trained) but /predict and /analyze will 503 until fixed.
        print(f"WARNING: model not loaded at startup: {e}")
        _pipeline = None
    yield


app = FastAPI(
    title="ImageForensics AI API",
    description=(
        "Classifies an image as REAL or AI_GENERATED with interpretable "
        "forensic evidence. Predictions are statistical estimates, NOT "
        "definitive proof of an image's origin — see the `disclaimer` "
        "field in every response."
    ),
    lifespan=lifespan,
)


def _validate_upload(file: UploadFile, contents: bytes, cfg) -> Image.Image:
    max_bytes = cfg.serving.max_upload_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(contents)} bytes exceeds {max_bytes} byte limit.",
        )

    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if file.filename and "." in file.filename else ""
    if ext not in cfg.serving.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension {ext!r}. Allowed: {cfg.serving.allowed_extensions}",
        )

    try:
        return Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid/readable image.")


def _require_pipeline() -> ForensicsPipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — train a checkpoint first (src.training.train_ablation) and restart the API.",
        )
    return _pipeline


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _pipeline is not None,
        "model_version": _pipeline.model_version if _pipeline else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Fast path: prediction + confidence, no Grad-CAM."""
    pipeline = _require_pipeline()
    cfg = load_config()
    contents = await file.read()
    pil_image = _validate_upload(file, contents, cfg)

    report, _ = pipeline.analyze(pil_image, include_gradcam=False)
    return JSONResponse(content=report.to_dict())


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Full forensic report including a base64-encoded Grad-CAM overlay PNG."""
    pipeline = _require_pipeline()
    cfg = load_config()
    contents = await file.read()
    pil_image = _validate_upload(file, contents, cfg)

    report, overlay = pipeline.analyze(pil_image, include_gradcam=True)

    response = report.to_dict()
    if overlay is not None:
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        response["gradcam_overlay_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        response["gradcam_overlay_base64"] = None

    return JSONResponse(content=response)
