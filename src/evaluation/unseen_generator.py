"""
Unseen-generator generalization evaluation (brief §8): evaluate a trained
model on both the in-distribution "test" split and the "unseen_test" split
(the held-out generator, never seen during training), and report the drop.
This is called out in the brief as the project's strongest claim, so the
comparison must use the exact same eval protocol (compute_metrics) as
everywhere else — WHERE that protocol is valid.

Important asymmetry, found while first running this end-to-end: the
"unseen_test" split contains ONLY the held-out generator's AI_GENERATED
images (by construction — see src/data/splits.py), never REAL images. That
means ROC-AUC, precision, and F1 are undefined there (they require both
classes in y_true). The only well-defined comparison is: what fraction of
held-out-generator AI images does the model correctly flag as AI_GENERATED
(i.e. recall on the AI_GENERATED class) — and since unseen_test is 100%
that class, this equals the split's raw accuracy. The fair "drop" to
report is therefore this unseen-generator detection rate vs. the
in-distribution test set's AI_GENERATED-class recall specifically (NOT
overall in-distribution accuracy, which is diluted by REAL images and
would understate the real comparison).

Run as: ./.venv/bin/python -m src.evaluation.unseen_generator --config full_fusion
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.multibranch_dataset import MultiBranchForensicsDataset
from src.evaluation.metrics import compute_metrics
from src.models.fusion import build_fusion_model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_split(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, y_prob) for a loader, no metric computation."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            inputs = {k: batch[k].to(device) for k in ("rgb", "fft", "residual")}
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)


def evaluate_split(model, loader, device) -> dict:
    """Full binary metrics — only valid when BOTH classes are present in the split."""
    y_true, y_pred, y_prob = predict_split(model, loader, device)
    return compute_metrics(y_true, y_pred, y_prob).to_dict()


def evaluate_ai_recall(model, loader, device) -> dict:
    """
    AI_GENERATED-class detection rate only — valid even on a single-class
    (AI-only) split. Used for the unseen_test split specifically.
    """
    y_true, y_pred, y_prob = predict_split(model, loader, device)
    if not np.all(y_true == 1):
        raise ValueError(
            "evaluate_ai_recall expects an AI_GENERATED-only split (unseen_test); "
            f"got classes {set(y_true.tolist())}. Use evaluate_split for mixed splits."
        )
    detection_rate = float((y_pred == 1).mean())
    return {"n_samples": int(len(y_true)), "ai_detection_rate": detection_rate}


def run_unseen_generator_eval(config_name: str) -> dict:
    cfg = load_config()
    device = get_device()
    metadata_path = cfg.paths.resolve("data_metadata")

    model = build_fusion_model(config_name, cfg.model).to(device)
    checkpoint_path = cfg.paths.resolve("runs_dir") / f"ablation_{config_name}" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint found at {checkpoint_path}. "
            f"Run src.training.train_ablation first."
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    in_dist_ds = MultiBranchForensicsDataset(metadata_path, "test", cfg.data.image_size)
    unseen_ds = MultiBranchForensicsDataset(metadata_path, "unseen_test", cfg.data.image_size)

    in_dist_loader = DataLoader(in_dist_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    unseen_loader = DataLoader(unseen_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    print(f"Evaluating {config_name} on in-distribution test set ({len(in_dist_ds)} samples)...")
    in_dist_metrics = evaluate_split(model, in_dist_loader, device)
    in_dist_ai_recall = in_dist_metrics["per_class_metrics"].get("AI_GENERATED", {}).get("accuracy")

    print(f"Evaluating {config_name} on unseen-generator test set ({len(unseen_ds)} samples)...")
    unseen_result = evaluate_ai_recall(model, unseen_loader, device)
    unseen_detection_rate = unseen_result["ai_detection_rate"]

    result = {
        "config": config_name,
        "held_out_generator": cfg.data.held_out_generator,
        "in_distribution_full_metrics": in_dist_metrics,
        "in_distribution_ai_recall": in_dist_ai_recall,
        "unseen_generator_detection_rate": unseen_detection_rate,
        "unseen_generator_n_samples": unseen_result["n_samples"],
        "ai_recall_drop": (in_dist_ai_recall - unseen_detection_rate) if in_dist_ai_recall is not None else None,
    }

    print(
        f"\nIn-distribution AI_GENERATED recall: {in_dist_ai_recall:.4f}\n"
        f"Unseen-generator ({cfg.data.held_out_generator}) detection rate: {unseen_detection_rate:.4f}\n"
        f"Drop: {result['ai_recall_drop']:.4f}"
    )

    run_dir = cfg.paths.resolve("runs_dir") / f"ablation_{config_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "unseen_generator_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_fusion", choices=["rgb_only", "rgb_fft", "rgb_residual", "full_fusion"])
    args = parser.parse_args()
    run_unseen_generator_eval(args.config)
