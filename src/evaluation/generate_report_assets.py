"""
Generates report assets for the README (brief §2.4 "honest failure
analysis" and §6 "reliability diagram"):
    - one REAL false positive (real image predicted AI_GENERATED)
    - one REAL false negative (AI image predicted REAL)
      each with its Grad-CAM heatmap overlay
    - a calibration reliability diagram (before/after temperature scaling)

Searches the actual test set for genuine misclassifications rather than
constructing examples — if no false positive or false negative exists in
the test set (a very strong model), this is reported honestly rather than
silently skipped.

Run as: ./.venv/bin/python -m src.evaluation.generate_report_assets --config full_fusion
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import load_config
from src.evaluation.calibration import TemperatureScaler, compute_ece_from_predictions
from src.inference.pipeline import ForensicsPipeline
from src.models.fusion import build_fusion_model


def find_failure_examples(cfg, config_name: str) -> dict:
    """
    Scans the test split for the first real false positive and first real
    false negative, returning their image paths (or None if the model has
    zero errors of that type on this test set).
    """
    metadata_path = cfg.paths.resolve("data_metadata")
    df = pd.read_csv(metadata_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    pipeline = ForensicsPipeline(config_name=config_name, cfg=cfg)

    false_positive_path = None
    false_negative_path = None

    for _, row in test_df.iterrows():
        if false_positive_path and false_negative_path:
            break
        true_class = row["class"]
        img = Image.open(row["path"]).convert("RGB")
        report, overlay = pipeline.analyze(img, include_gradcam=True)

        if true_class == "REAL" and report.predicted_class == "AI_GENERATED" and not false_positive_path:
            false_positive_path = row["path"]
            save_failure_example(img, overlay, report, "false_positive", cfg)
        elif true_class == "AI_GENERATED" and report.predicted_class == "REAL" and not false_negative_path:
            false_negative_path = row["path"]
            save_failure_example(img, overlay, report, "false_negative", cfg)

    return {"false_positive": false_positive_path, "false_negative": false_negative_path}


def save_failure_example(img: Image.Image, overlay: np.ndarray, report, name: str, cfg) -> None:
    screenshots_dir = cfg.paths.resolve("models_dir").parent / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img.resize((cfg.data.image_size, cfg.data.image_size)))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM (predicted {report.predicted_class}, {report.calibrated_confidence:.1%})")
    axes[1].axis("off")
    fig.suptitle(name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(screenshots_dir / f"{name}.png", dpi=120)
    plt.close(fig)
    print(f"Saved {screenshots_dir / f'{name}.png'}")


def plot_reliability_diagram(cfg, config_name: str) -> None:
    """Renders reliability diagrams before and after temperature scaling, side by side."""
    from torch.utils.data import DataLoader

    from src.data.multibranch_dataset import MultiBranchForensicsDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_path = cfg.paths.resolve("data_metadata")

    model = build_fusion_model(config_name, cfg.model).to(device)
    checkpoint_path = cfg.paths.resolve("runs_dir") / f"ablation_{config_name}" / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    test_ds = MultiBranchForensicsDataset(metadata_path, "test", cfg.data.image_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model.eval()
    all_labels, all_preds, all_probs, all_logits = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"].to(device)
            inputs = {k: batch[k].to(device) for k in ("rgb", "fft", "residual")}
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            confidence, preds = probs.max(dim=1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(confidence.cpu())
            all_logits.append(logits.cpu())

    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_preds)
    y_conf_raw = torch.cat(all_probs).numpy()
    logits_all = torch.cat(all_logits)
    correct = (y_pred == y_true).float().numpy()

    ece_before, bins_before = compute_ece_from_predictions(correct, y_conf_raw, n_bins=cfg.calibration.n_bins_ece)

    temp_path = cfg.paths.resolve("runs_dir") / f"ablation_{config_name}" / "temperature.json"
    with open(temp_path) as f:
        temperature = json.load(f)["temperature"]
    scaler = TemperatureScaler()
    with torch.no_grad():
        scaler.temperature.fill_(temperature)
        scaled_logits = scaler(logits_all)
        scaled_probs = torch.softmax(scaled_logits, dim=1)
        y_conf_scaled = scaled_probs.max(dim=1).values.numpy()
    ece_after, bins_after = compute_ece_from_predictions(correct, y_conf_scaled, n_bins=cfg.calibration.n_bins_ece)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, bins, ece, title in [
        (axes[0], bins_before, ece_before, f"Before calibration (ECE={ece_before:.4f})"),
        (axes[1], bins_after, ece_after, f"After calibration, T={temperature:.3f} (ECE={ece_after:.4f})"),
    ]:
        confidences = [b.confidence for b in bins if b.count > 0]
        accuracies = [b.accuracy for b in bins if b.count > 0]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.bar(confidences, accuracies, width=0.05, alpha=0.7, edgecolor="black", label="Model")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    screenshots_dir = cfg.paths.resolve("models_dir").parent / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(screenshots_dir / "reliability_diagram.png", dpi=120)
    plt.close(fig)
    print(f"Saved {screenshots_dir / 'reliability_diagram.png'}")


def main(config_name: str = "full_fusion") -> None:
    cfg = load_config()

    print("Searching test set for a real false positive and false negative...")
    failures = find_failure_examples(cfg, config_name)
    if not failures["false_positive"]:
        print("WARNING: no false positive found in the test set — model had zero REAL->AI errors here.")
    if not failures["false_negative"]:
        print("WARNING: no false negative found in the test set — model had zero AI->REAL errors here.")

    print("\nRendering reliability diagram...")
    plot_reliability_diagram(cfg, config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_fusion")
    args = parser.parse_args()
    main(args.config)
