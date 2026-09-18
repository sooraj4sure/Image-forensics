"""
Ablation study runner (brief §6/§2.1): trains all 4 configs
(rgb_only, rgb_fft, rgb_residual, full_fusion) with an IDENTICAL protocol
(same splits, same epochs, same optimizer/scheduler settings, same eval
function) and produces one results table — this is what makes the
comparison valid rather than four separately-tuned experiments.

Run as: ./.venv/bin/python -m src.training.train_ablation
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.multibranch_dataset import MultiBranchForensicsDataset
from src.evaluation.metrics import compute_metrics
from src.models.fusion import ABLATION_CONFIGS, build_fusion_model
from src.utils import set_seed


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "rgb": batch["rgb"].to(device),
        "fft": batch["fft"].to(device),
        "residual": batch["residual"].to(device),
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            labels = batch["label"].to(device)
            inputs = move_batch_to_device(batch, device)

            if train:
                optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    return avg_loss, metrics


def train_one_config(config_name: str, cfg, train_loader, val_loader, test_loader, device) -> dict:
    print(f"\n=== Training config: {config_name} ({ABLATION_CONFIGS[config_name]}) ===")
    set_seed(cfg.project.seed)  # re-seed identically before each config's training

    run_dir = cfg.paths.resolve("runs_dir") / f"ablation_{config_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    existing_result_path = run_dir / "test_result.json"
    existing_checkpoint_path = run_dir / "best_model.pt"
    if existing_result_path.exists() and existing_checkpoint_path.exists():
        with open(existing_result_path) as f:
            cached_result = json.load(f)
        print(f"=== Skipping {config_name} — already trained (found {existing_result_path}) ===")
        print(f"  cached: test_f1={cached_result['test_f1']:.4f} test_roc_auc={cached_result['test_roc_auc']:.4f}")
        return cached_result

    model = build_fusion_model(config_name, cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    best_model_path = run_dir / "best_model.pt"
    history = []

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"  [{config_name}] epoch {epoch}/{cfg.training.epochs} ({elapsed:.1f}s) | "
            f"train_f1={train_metrics.f1:.4f} val_f1={val_metrics.f1:.4f} val_auc={val_metrics.roc_auc:.4f}"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             "train_f1": train_metrics.f1, "val_f1": val_metrics.f1, "val_auc": val_metrics.roc_auc}
        )

        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.training.early_stopping_patience:
                print(f"  [{config_name}] early stopping at epoch {epoch}")
                break

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on TEST split using the best checkpoint — same
    # eval protocol (compute_metrics) for every config, per the brief.
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_loss, test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

    result = {
        "config": config_name,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_metrics.accuracy,
        "test_precision": test_metrics.precision,
        "test_recall": test_metrics.recall,
        "test_f1": test_metrics.f1,
        "test_roc_auc": test_metrics.roc_auc,
    }
    with open(run_dir / "test_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def format_comparison_table(results: list[dict]) -> str:
    header = "| Config | Test Acc | Test Prec | Test Recall | Test F1 | Test ROC-AUC |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = ""
    for r in results:
        rows += (
            f"| {r['config']} | {r['test_accuracy']:.4f} | {r['test_precision']:.4f} | "
            f"{r['test_recall']:.4f} | {r['test_f1']:.4f} | {r['test_roc_auc']:.4f} |\n"
        )
    return header + rows


def run_ablation_study() -> list[dict]:
    cfg = load_config()
    device = get_device()
    metadata_path = cfg.paths.resolve("data_metadata")

    train_ds = MultiBranchForensicsDataset(metadata_path, "train", cfg.data.image_size)
    val_ds = MultiBranchForensicsDataset(metadata_path, "val", cfg.data.image_size)
    test_ds = MultiBranchForensicsDataset(metadata_path, "test", cfg.data.image_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    results = []
    for config_name in cfg.ablation.configs:
        result = train_one_config(config_name, cfg, train_loader, val_loader, test_loader, device)
        results.append(result)

    table = format_comparison_table(results)
    print("\n" + table)

    comparison_path = cfg.paths.resolve("runs_dir") / "ablation_comparison.md"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w") as f:
        f.write("# Ablation Study Results\n\n" + table)
    print(f"Comparison table written to {comparison_path}")

    return results


if __name__ == "__main__":
    run_ablation_study()