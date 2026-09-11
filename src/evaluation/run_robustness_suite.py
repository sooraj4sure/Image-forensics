"""
Robustness suite runner (brief §7): evaluate the FINAL FUSION MODEL ONLY
under each degradation case, report accuracy/F1 delta vs. the clean test
baseline. Explicitly not re-run across all 4 ablation configs (out of
scope per §3/§7 — time budget protected for the ablation + unseen-
generator experiments instead).

Run as: ./.venv/bin/python -m src.evaluation.run_robustness_suite
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.multibranch_dataset import MultiBranchForensicsDataset
from src.evaluation.robustness import build_robustness_cases
from src.evaluation.unseen_generator import evaluate_split, get_device
from src.models.fusion import build_fusion_model

FINAL_CONFIG = "full_fusion"


def run_robustness_suite() -> dict:
    cfg = load_config()
    device = get_device()
    metadata_path = cfg.paths.resolve("data_metadata")

    model = build_fusion_model(FINAL_CONFIG, cfg.model).to(device)
    checkpoint_path = cfg.paths.resolve("runs_dir") / f"ablation_{FINAL_CONFIG}" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint found at {checkpoint_path}. "
            f"Run src.training.train_ablation first."
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    # Clean baseline
    clean_ds = MultiBranchForensicsDataset(metadata_path, "test", cfg.data.image_size)
    clean_loader = DataLoader(clean_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    print(f"Evaluating clean baseline ({len(clean_ds)} test samples)...")
    clean_metrics = evaluate_split(model, clean_loader, device)
    print(f"  clean: acc={clean_metrics['accuracy']:.4f} f1={clean_metrics['f1']:.4f}")

    results = {"clean": clean_metrics}
    cases = build_robustness_cases(cfg.robustness)

    for case in cases:
        degraded_ds = MultiBranchForensicsDataset(
            metadata_path, "test", cfg.data.image_size, pre_transform=case.transform
        )
        degraded_loader = DataLoader(
            degraded_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers
        )
        metrics = evaluate_split(model, degraded_loader, device)
        acc_delta = metrics["accuracy"] - clean_metrics["accuracy"]
        f1_delta = metrics["f1"] - clean_metrics["f1"]
        print(f"  {case.name}: acc={metrics['accuracy']:.4f} ({acc_delta:+.4f}) f1={metrics['f1']:.4f} ({f1_delta:+.4f})")
        results[case.name] = {**metrics, "accuracy_delta": acc_delta, "f1_delta": f1_delta}

    run_dir = cfg.paths.resolve("runs_dir") / f"ablation_{FINAL_CONFIG}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "robustness_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRobustness results written to {run_dir / 'robustness_result.json'}")

    return results


if __name__ == "__main__":
    run_robustness_suite()
