"""Tests for src/deployment/package_model.py."""

import json

import pytest
import torch

from src.config import load_config
from src.deployment.package_model import package_model
from src.models.fusion import build_fusion_model


@pytest.fixture
def cfg_with_trained_checkpoint(tmp_path):
    cfg = load_config()
    cfg = cfg.model_copy(deep=True)
    cfg.paths.runs_dir = str(tmp_path / "runs")
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.model.pretrained = False

    run_dir = tmp_path / "runs" / "ablation_full_fusion"
    run_dir.mkdir(parents=True)
    model = build_fusion_model("full_fusion", cfg.model)
    torch.save(model.state_dict(), run_dir / "best_model.pt")

    return cfg


def test_package_model_copies_checkpoint(cfg_with_trained_checkpoint, monkeypatch):
    monkeypatch.setattr("src.deployment.package_model.load_config", lambda: cfg_with_trained_checkpoint)
    package_model("full_fusion")

    dest = cfg_with_trained_checkpoint.paths.resolve("models_dir") / "production" / "best_model.pt"
    assert dest.exists()


def test_package_model_copies_temperature_json_when_present(cfg_with_trained_checkpoint, monkeypatch):
    run_dir = cfg_with_trained_checkpoint.paths.resolve("runs_dir") / "ablation_full_fusion"
    with open(run_dir / "temperature.json", "w") as f:
        json.dump({"temperature": 1.5, "ece_before": 0.1, "ece_after": 0.05, "n_val_samples": 10}, f)

    monkeypatch.setattr("src.deployment.package_model.load_config", lambda: cfg_with_trained_checkpoint)
    package_model("full_fusion")

    dest_temp = cfg_with_trained_checkpoint.paths.resolve("models_dir") / "production" / "temperature.json"
    assert dest_temp.exists()
    with open(dest_temp) as f:
        assert json.load(f)["temperature"] == 1.5


def test_package_model_raises_when_no_checkpoint(tmp_path, monkeypatch):
    cfg = load_config()
    cfg = cfg.model_copy(deep=True)
    cfg.paths.runs_dir = str(tmp_path / "empty_runs")

    monkeypatch.setattr("src.deployment.package_model.load_config", lambda: cfg)
    with pytest.raises(FileNotFoundError, match="No trained checkpoint"):
        package_model("full_fusion")


def test_packaged_model_is_loadable_by_pipeline_via_env_override(cfg_with_trained_checkpoint, monkeypatch):
    # End-to-end: package, then verify ForensicsPipeline can actually load
    # from the packaged location via IMAGEFORENSICS_MODEL_DIR.
    from PIL import Image

    from src.inference.pipeline import ForensicsPipeline

    monkeypatch.setattr("src.deployment.package_model.load_config", lambda: cfg_with_trained_checkpoint)
    package_model("full_fusion")

    dest_dir = cfg_with_trained_checkpoint.paths.resolve("models_dir") / "production"
    monkeypatch.setenv("IMAGEFORENSICS_MODEL_DIR", str(dest_dir))

    pipeline = ForensicsPipeline(config_name="full_fusion", cfg=cfg_with_trained_checkpoint)
    report, _ = pipeline.analyze(Image.new("RGB", (64, 64), color=(1, 2, 3)))
    assert report.predicted_class in ("REAL", "AI_GENERATED")
