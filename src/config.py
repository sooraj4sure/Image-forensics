"""
Central config loader for ImageForensics AI.

Every module in src/, api/, and app/ should get paths and hyperparameters
by calling `load_config()` from here — never by hardcoding values inline.
This keeps configs/config.yaml as the single source of truth (brief §11:
"Config via files/env vars, not hardcoded paths/hyperparameters/secrets").

Usage:
    from src.config import load_config
    cfg = load_config()
    print(cfg.model.backbone)          # "resnet18"
    print(cfg.paths.resolve("models_dir"))  # absolute Path to models/
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field

# Repo root = two levels up from this file (src/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


class ProjectConfig(BaseModel):
    name: str
    seed: int


class PathsConfig(BaseModel):
    data_raw: str
    data_processed: str
    data_metadata: str
    models_dir: str
    runs_dir: str

    def resolve(self, field_name: str) -> Path:
        """Return an absolute Path for a given paths.* field, rooted at REPO_ROOT."""
        rel = getattr(self, field_name)
        return (REPO_ROOT / rel).resolve()


class DataConfig(BaseModel):
    image_size: int
    batch_size: int
    num_workers: int
    val_split: float
    test_split: float
    dataset_name: Optional[str] = None
    dataset_source_url: Optional[str] = None
    dataset_license: Optional[str] = None
    generator_families: List[str] = Field(default_factory=list)
    held_out_generator: Optional[str] = None


class ModelConfig(BaseModel):
    backbone: str
    pretrained: bool
    num_classes: int
    dropout: float


class TrainingConfig(BaseModel):
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    early_stopping_patience: int
    amp: bool


class AblationConfig(BaseModel):
    configs: List[str]


class CalibrationConfig(BaseModel):
    method: str
    n_bins_ece: int


class RobustnessConfig(BaseModel):
    jpeg_quality_levels: List[int]
    resize_scale: float
    crop_fraction: float
    blur_kernel: int
    noise_std: float


class ServingConfig(BaseModel):
    api_host: str
    api_port: int
    max_upload_mb: int
    allowed_extensions: List[str]


class AppConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    ablation: AblationConfig
    calibration: CalibrationConfig
    robustness: RobustnessConfig
    serving: ServingConfig


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load and validate configs/config.yaml.

    An explicit IMAGEFORENSICS_CONFIG env var overrides the default path,
    which is useful for tests or alternate experiment configs. Cached so
    repeated calls across modules don't re-read/re-parse the file.
    """
    path = Path(
        config_path
        or os.environ.get("IMAGEFORENSICS_CONFIG", "")
        or DEFAULT_CONFIG_PATH
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            f"Expected configs/config.yaml at repo root, or set IMAGEFORENSICS_CONFIG."
        )
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)


if __name__ == "__main__":
    # Quick manual sanity check: `python -m src.config`
    cfg = load_config()
    print(f"Loaded config for project: {cfg.project.name}")
    print(f"Seed: {cfg.project.seed}")
    print(f"Models dir (resolved): {cfg.paths.resolve('models_dir')}")
    print(f"Backbone: {cfg.model.backbone}, image_size: {cfg.data.image_size}")
