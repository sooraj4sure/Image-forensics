"""
Baseline training loop (Stage 3): fine-tune ResNet-18 on the RGB-only
config. Later ablation configs (Stage 4) reuse this same loop by swapping
in a different model — kept generic on purpose.

Run as: ./.venv/bin/python -m src.training.train
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
from src.data.dataset import ForensicsDataset
from src.evaluation.metrics import compute_metrics
from src.models.backbone import build_model
from src.utils import set_seed


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
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


def train_baseline(run_name: str = "baseline_rgb_only") -> Path:
    cfg = load_config()
    set_seed(cfg.project.seed)
    device = get_device()

    metadata_path = cfg.paths.resolve("data_metadata")
    train_ds = ForensicsDataset(metadata_path, "train", cfg.data.image_size)
    val_ds = ForensicsDataset(metadata_path, "val", cfg.data.image_size)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )

    model = build_model(cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    run_dir = cfg.paths.resolve("runs_dir") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    best_model_path = run_dir / "best_model.pt"

    print(f"Training on device: {device}")
    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{cfg.training.epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} train_f1={train_metrics.f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_metrics.f1:.4f} val_auc={val_metrics.roc_auc:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_f1": train_metrics.f1,
                "val_loss": val_loss,
                "val_f1": val_metrics.f1,
                "val_auc": val_metrics.roc_auc,
            }
        )

        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no val_f1 improvement for {cfg.training.early_stopping_patience} epochs)")
                break

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Best val_f1: {best_val_f1:.4f}. Model saved to {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    train_baseline()
