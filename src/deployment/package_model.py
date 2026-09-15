"""
Packages the trained full_fusion checkpoint (+ calibration, if fitted)
from runs/ablation_full_fusion/ into models/production/ — the fixed path
a deployed environment (HF Spaces) points at via the IMAGEFORENSICS_MODEL_DIR
env var (see src/inference/pipeline.py).

Deliberately a separate, explicit step rather than training writing
directly to models/production/ — packaging for deployment should be a
conscious decision after you've reviewed the ablation results and picked
which run to actually ship, not an automatic side effect of training.

Run as: ./.venv/bin/python -m src.deployment.package_model --config full_fusion
"""

from __future__ import annotations

import argparse
import shutil

from src.config import load_config


def package_model(config_name: str = "full_fusion") -> None:
    cfg = load_config()
    source_dir = cfg.paths.resolve("runs_dir") / f"ablation_{config_name}"
    checkpoint_path = source_dir / "best_model.pt"
    temperature_path = source_dir / "temperature.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint at {checkpoint_path}. "
            f"Run src.training.train_ablation first."
        )

    dest_dir = cfg.paths.resolve("models_dir") / "production"
    dest_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, dest_dir / "best_model.pt")
    print(f"Copied {checkpoint_path} -> {dest_dir / 'best_model.pt'}")

    size_mb = (dest_dir / "best_model.pt").stat().st_size / (1024 ** 2)
    print(f"Model size: {size_mb:.1f} MB", end="")
    if size_mb > 100:
        print(" — WARNING: exceeds the brief's 'well under 100MB' target for CPU-tier hosting.")
    else:
        print(" (within the CPU-tier hosting target)")

    if temperature_path.exists():
        shutil.copy2(temperature_path, dest_dir / "temperature.json")
        print(f"Copied {temperature_path} -> {dest_dir / 'temperature.json'}")
    else:
        print(
            "WARNING: no temperature.json found — the deployed app will report "
            "raw (uncalibrated) softmax. Run src.evaluation.fit_calibration first "
            "if you want calibrated confidence in production."
        )

    print(
        f"\nDone. To deploy: commit {dest_dir}/ (git-lfs tracks *.pt automatically, "
        f"see .gitattributes), then follow DEPLOYMENT.md."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_fusion", choices=["rgb_only", "rgb_fft", "rgb_residual", "full_fusion"])
    args = parser.parse_args()
    package_model(args.config)
