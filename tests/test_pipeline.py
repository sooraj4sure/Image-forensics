"""
Integration test for src/inference/pipeline.py. Builds a throwaway
untrained checkpoint in a tmp_path runs dir (pretrained=False, since this
sandbox can't reach download.pytorch.org) and verifies analyze() runs
end-to-end and returns a well-formed report + heatmap. Not testing
prediction QUALITY (meaningless for an untrained model) — testing that
the pipeline's plumbing (preprocessing, all 3 branches, calibration
fallback, Grad-CAM, metadata, disclaimers) actually works together.
"""

import numpy as np
import torch
from PIL import Image

from src.config import load_config
from src.inference.pipeline import ForensicsPipeline
from src.models.fusion import build_fusion_model


def build_temp_pipeline(tmp_path, config_name="full_fusion"):
    cfg = load_config()
    cfg = cfg.model_copy(deep=True)
    cfg.paths.runs_dir = str(tmp_path)
    cfg.model.pretrained = False  # sandbox can't download ImageNet weights

    checkpoint_dir = tmp_path / f"ablation_{config_name}"
    checkpoint_dir.mkdir(parents=True)
    model = build_fusion_model(config_name, cfg.model)
    torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")

    return ForensicsPipeline(config_name=config_name, cfg=cfg)


def test_pipeline_analyze_returns_well_formed_report(tmp_path):
    pipeline = build_temp_pipeline(tmp_path)
    img = Image.new("RGB", (100, 100), color=(120, 80, 200))

    report, overlay = pipeline.analyze(img)

    assert report.predicted_class in ("REAL", "AI_GENERATED")
    assert 0.0 <= report.raw_confidence <= 1.0
    assert 0.0 <= report.calibrated_confidence <= 1.0
    assert 0.0 <= report.probability_ai_generated <= 1.0
    assert "rgb" in report.branch_signal_norms
    assert "fft" in report.branch_signal_norms
    assert "residual" in report.branch_signal_norms
    assert "NOT definitive proof" in report.disclaimer
    assert "does not show regions proven" in report.gradcam_caption
    assert overlay is not None
    assert overlay.shape == (pipeline.cfg.data.image_size, pipeline.cfg.data.image_size, 3)
    assert overlay.dtype == np.uint8


def test_pipeline_reports_uncalibrated_when_no_temperature_fitted(tmp_path):
    pipeline = build_temp_pipeline(tmp_path)
    img = Image.new("RGB", (64, 64), color=(30, 30, 30))
    report, _ = pipeline.analyze(img)
    assert pipeline.temperature_fitted is False
    assert "calibration_note" in report.metadata


def test_pipeline_metadata_never_flags_missing_exif_as_suspicious(tmp_path):
    pipeline = build_temp_pipeline(tmp_path)
    img = Image.new("RGB", (64, 64), color=(10, 10, 10))  # no EXIF
    report, _ = pipeline.analyze(img)
    assert report.metadata["has_exif"] is False
    assert "NOT evidence of AI generation" in report.metadata["note"]


def test_pipeline_rgb_only_config_has_no_gradcam_skip_but_still_works(tmp_path):
    # rgb_only still HAS an rgb branch, so gradcam should still be generated.
    pipeline = build_temp_pipeline(tmp_path, config_name="rgb_only")
    img = Image.new("RGB", (64, 64), color=(200, 50, 50))
    report, overlay = pipeline.analyze(img)
    assert overlay is not None
    assert report.branch_signal_norms == {"rgb": report.branch_signal_norms["rgb"]}


def test_pipeline_report_to_dict_is_json_serializable():
    import json

    from src.inference.pipeline import ForensicReport

    report = ForensicReport(
        predicted_class="REAL",
        raw_confidence=0.8,
        calibrated_confidence=0.7,
        probability_ai_generated=0.3,
        branch_signal_norms={"rgb": 1.0},
        metadata={"has_exif": False},
        model_version="test@0",
    )
    json.dumps(report.to_dict())  # should not raise
