"""
Tests for api/main.py using FastAPI's TestClient. Uses a temp untrained
checkpoint (same pattern as tests/test_pipeline.py) so these tests don't
depend on real trained weights or real data — they test the API's request/
response handling (validation, status codes, schema), not prediction quality.
"""

import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main
from src.config import load_config
from src.inference.pipeline import ForensicsPipeline
from src.models.fusion import build_fusion_model


@pytest.fixture
def client_with_model(tmp_path, monkeypatch):
    cfg = load_config()
    cfg = cfg.model_copy(deep=True)
    cfg.paths.runs_dir = str(tmp_path)
    cfg.model.pretrained = False

    checkpoint_dir = tmp_path / "ablation_full_fusion"
    checkpoint_dir.mkdir(parents=True)
    model = build_fusion_model("full_fusion", cfg.model)
    torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")

    # Bypass the lifespan startup (which reads the real config path) by
    # setting the module-level pipeline directly with our temp-config one.
    api_main._pipeline = ForensicsPipeline(config_name="full_fusion", cfg=cfg)
    monkeypatch.setattr(api_main, "load_config", lambda: cfg)

    with TestClient(api_main.app) as client:
        yield client

    api_main._pipeline = None


@pytest.fixture
def client_no_model():
    api_main._pipeline = None
    with TestClient(api_main.app) as client:
        yield client


def make_image_bytes(size=(64, 64), fmt="JPEG"):
    img = Image.new("RGB", size, color=(120, 60, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def test_health_with_model_loaded(client_with_model):
    resp = client_with_model.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_without_model_loaded(client_no_model):
    resp = client_no_model.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is False


def test_predict_without_model_returns_503(client_no_model):
    resp = client_no_model.post("/predict", files={"file": ("test.jpg", make_image_bytes(), "image/jpeg")})
    assert resp.status_code == 503


def test_predict_returns_well_formed_report(client_with_model):
    resp = client_with_model.post("/predict", files={"file": ("test.jpg", make_image_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in ("REAL", "AI_GENERATED")
    assert "disclaimer" in body
    assert "NOT definitive proof" in body["disclaimer"]
    assert "gradcam_overlay_base64" not in body  # /predict is the no-gradcam fast path


def test_analyze_includes_gradcam_overlay(client_with_model):
    resp = client_with_model.post("/analyze", files={"file": ("test.jpg", make_image_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["gradcam_overlay_base64"] is not None
    assert len(body["gradcam_overlay_base64"]) > 0


def test_rejects_disallowed_extension(client_with_model):
    fake_file = io.BytesIO(b"not an image")
    resp = client_with_model.post("/predict", files={"file": ("malware.exe", fake_file, "application/octet-stream")})
    assert resp.status_code == 415


def test_rejects_corrupt_image_with_allowed_extension(client_with_model):
    fake_file = io.BytesIO(b"this is not actually image data")
    resp = client_with_model.post("/predict", files={"file": ("fake.jpg", fake_file, "image/jpeg")})
    assert resp.status_code == 422


def test_rejects_oversized_upload(client_with_model, monkeypatch):
    # Shrink the limit for this test rather than uploading a real 10MB+ file
    client_with_model.app.dependency_overrides = {}
    big_image = make_image_bytes(size=(2000, 2000))
    content = big_image.read()

    # Monkeypatch the cfg's max_upload_mb very low via the already-injected cfg
    api_main._pipeline.cfg.serving.max_upload_mb = 0.001
    resp = client_with_model.post(
        "/predict", files={"file": ("big.jpg", io.BytesIO(content), "image/jpeg")}
    )
    assert resp.status_code == 413
