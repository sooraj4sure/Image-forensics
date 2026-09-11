"""Sanity tests for the config loader (Stage 1)."""

from src.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg.project.name == "image-forensics-ai"
    assert cfg.model.backbone == "resnet18"
    assert cfg.data.image_size == 224


def test_paths_resolve_absolute():
    cfg = load_config()
    models_path = cfg.paths.resolve("models_dir")
    assert models_path.is_absolute()
    assert models_path.name == "models"


def test_ablation_configs_present():
    cfg = load_config()
    assert set(cfg.ablation.configs) == {
        "rgb_only",
        "rgb_fft",
        "rgb_residual",
        "full_fusion",
    }
