"""
Fits temperature scaling on the VALIDATION split (brief §6: calibration
must not leak test-set information) for a given ablation config, reports
ECE before/after, and writes runs/ablation_<config>/temperature.json —
which src.inference.pipeline.ForensicsPipeline looks for at startup.

Run as: ./.venv/bin/python -m src.evaluation.fit_calibration --config full_fusion
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.multibranch_dataset import MultiBranchForensicsDataset
from src.evaluation.calibration import TemperatureScaler, compute_ece_from_predictions
from src.models.fusion import build_fusion_model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_val_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            inputs = {k: batch[k].to(device) for k in ("rgb", "fft", "residual")}
            logits = model(inputs)
            all_logits.append(logits)
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def ece_for_logits(logits: torch.Tensor, labels: torch.Tensor, n_bins: int) -> float:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        confidence, preds = probs.max(dim=1)
    correct = (preds == labels).float().cpu().numpy()
    ece, _ = compute_ece_from_predictions(correct, confidence.cpu().numpy(), n_bins=n_bins)
    return ece


def fit_calibration(config_name: str) -> dict:
    cfg = load_config()
    device = get_device()
    metadata_path = cfg.paths.resolve("data_metadata")

    model = build_fusion_model(config_name, cfg.model).to(device)
    checkpoint_path = cfg.paths.resolve("runs_dir") / f"ablation_{config_name}" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint at {checkpoint_path}. Run src.training.train_ablation first."
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    val_ds = MultiBranchForensicsDataset(metadata_path, "val", cfg.data.image_size)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    logits, labels = collect_val_logits(model, val_loader, device)

    ece_before = ece_for_logits(logits, labels, cfg.calibration.n_bins_ece)

    scaler = TemperatureScaler().to(device)
    scaler.fit(logits, labels)
    ece_after = ece_for_logits(scaler(logits), labels, cfg.calibration.n_bins_ece)

    learned_temperature = float(scaler.temperature.item())
    print(
        f"[{config_name}] ECE before calibration: {ece_before:.4f} | "
        f"after (T={learned_temperature:.3f}): {ece_after:.4f}"
    )

    run_dir = cfg.paths.resolve("runs_dir") / f"ablation_{config_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "temperature": learned_temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "n_val_samples": len(labels),
    }
    with open(run_dir / "temperature.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {run_dir / 'temperature.json'}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_fusion", choices=["rgb_only", "rgb_fft", "rgb_residual", "full_fusion"])
    args = parser.parse_args()
    fit_calibration(args.config)
