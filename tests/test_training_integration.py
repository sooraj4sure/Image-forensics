"""
Integration tests turning the manual end-to-end smoke tests from Stages
3-6 into permanent automated coverage. Uses tiny synthetic data (a
gradient=REAL vs noise=AI toy task, same as the original manual smoke
tests) and 1 epoch everywhere to keep runtime short — these test that the
CLI-driven training/calibration/robustness/unseen-generator SCRIPTS run
correctly end-to-end, not that they produce good models.
"""

import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.data.build_metadata import main as build_metadata_main
from src.evaluation.fit_calibration import fit_calibration
from src.evaluation.run_robustness_suite import run_robustness_suite
from src.evaluation.unseen_generator import run_unseen_generator_eval
from src.training.train import train_baseline
from src.training.train_ablation import run_ablation_study


def make_synthetic_raw_data(raw_dir, n_real=16, n_ai_per_gen=12, size=32):
    (raw_dir / "real").mkdir(parents=True, exist_ok=True)
    for gen in ("stub_gen_a", "stub_gen_b", "stub_gen_c"):
        (raw_dir / "ai" / gen).mkdir(parents=True, exist_ok=True)

    for i in range(n_real):
        arr = np.tile(np.linspace(0, 255, size), (size, 1)).astype("uint8")
        Image.fromarray(np.stack([arr] * 3, axis=-1)).save(raw_dir / "real" / f"real_{i}.jpg")

    rng = np.random.default_rng(0)
    for gen in ("stub_gen_a", "stub_gen_b", "stub_gen_c"):
        for i in range(n_ai_per_gen):
            arr = (rng.random((size, size, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(raw_dir / "ai" / gen / f"{gen}_{i}.jpg")


@pytest.fixture
def tiny_cfg(tmp_path, monkeypatch):
    """A full AppConfig pointed entirely at tmp_path, scaled down for fast tests."""
    cfg = load_config()
    cfg = cfg.model_copy(deep=True)

    raw_dir = tmp_path / "raw"
    make_synthetic_raw_data(raw_dir)

    cfg.paths.data_raw = str(raw_dir)
    cfg.paths.data_metadata = str(tmp_path / "metadata.csv")
    cfg.paths.runs_dir = str(tmp_path / "runs")
    cfg.data.held_out_generator = "stub_gen_c"
    cfg.data.val_split = 0.2
    cfg.data.test_split = 0.2
    cfg.data.batch_size = 4
    cfg.data.num_workers = 0
    cfg.model.pretrained = False  # sandbox can't reach download.pytorch.org
    cfg.training.epochs = 1
    cfg.training.early_stopping_patience = 5
    cfg.ablation.configs = ["full_fusion"]  # keep the ablation test fast

    monkeypatch.setattr("src.data.build_metadata.load_config", lambda: cfg)
    build_metadata_main()

    return cfg


def test_train_baseline_runs_end_to_end(tiny_cfg, monkeypatch):
    monkeypatch.setattr("src.training.train.load_config", lambda: tiny_cfg)
    checkpoint_path = train_baseline(run_name="test_baseline")
    assert checkpoint_path.exists()
    history_path = checkpoint_path.parent / "history.json"
    assert history_path.exists()


def test_ablation_study_runs_end_to_end(tiny_cfg, monkeypatch):
    monkeypatch.setattr("src.training.train_ablation.load_config", lambda: tiny_cfg)
    results = run_ablation_study()
    assert len(results) == 1
    assert results[0]["config"] == "full_fusion"
    assert "test_f1" in results[0]

    comparison_path = tiny_cfg.paths.resolve("runs_dir") / "ablation_comparison.md"
    assert comparison_path.exists()
    assert "full_fusion" in comparison_path.read_text()


def test_calibration_and_unseen_generator_and_robustness_pipeline(tiny_cfg, monkeypatch):
    # Chain: train full_fusion -> fit calibration -> unseen-generator eval -> robustness suite.
    # Mirrors the exact manual smoke-test sequence run during Stages 5-6.
    monkeypatch.setattr("src.training.train_ablation.load_config", lambda: tiny_cfg)
    run_ablation_study()

    monkeypatch.setattr("src.evaluation.fit_calibration.load_config", lambda: tiny_cfg)
    calib_result = fit_calibration("full_fusion")
    assert "temperature" in calib_result
    temp_path = tiny_cfg.paths.resolve("runs_dir") / "ablation_full_fusion" / "temperature.json"
    assert temp_path.exists()

    monkeypatch.setattr("src.evaluation.unseen_generator.load_config", lambda: tiny_cfg)
    unseen_result = run_unseen_generator_eval("full_fusion")
    assert unseen_result["held_out_generator"] == "stub_gen_c"
    assert 0.0 <= unseen_result["unseen_generator_detection_rate"] <= 1.0

    monkeypatch.setattr("src.evaluation.run_robustness_suite.load_config", lambda: tiny_cfg)
    robustness_result = run_robustness_suite()
    assert "clean" in robustness_result
    assert len(robustness_result) == 8  # clean + 7 degradation cases
