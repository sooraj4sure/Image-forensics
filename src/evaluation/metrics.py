"""
Shared metrics computation. Every stage that reports numbers (baseline,
ablation study, robustness testing) should call these functions rather
than recomputing metrics inline — keeps the eval protocol identical across
comparisons, which the brief requires ("same eval protocol for all four"
configs in the ablation study).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    n_samples: int
    per_class_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_samples": self.n_samples,
            "per_class_metrics": self.per_class_metrics,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> EvalResult:
    """
    Args:
        y_true: ground-truth labels, shape (n,), values in {0, 1}
        y_pred: predicted labels (argmax), shape (n,), values in {0, 1}
        y_prob: predicted probability of class 1 (AI_GENERATED), shape (n,)

    ROC-AUC requires both classes present in y_true; raises a clear error
    otherwise rather than silently returning NaN (e.g. if a debug run is
    accidentally given a single-class subset).
    """
    if len(set(y_true.tolist())) < 2:
        raise ValueError(
            "ROC-AUC requires both classes present in y_true; got only "
            f"{set(y_true.tolist())}. Check the split/subset being evaluated."
        )

    accuracy = float((y_true == y_pred).mean())
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    per_class = {}
    for cls, name in [(0, "REAL"), (1, "AI_GENERATED")]:
        mask = y_true == cls
        if mask.sum() > 0:
            per_class[name] = {
                "n": int(mask.sum()),
                "accuracy": float((y_pred[mask] == y_true[mask]).mean()),
            }

    return EvalResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm,
        n_samples=len(y_true),
        per_class_metrics=per_class,
    )


def measure_inference_time(model, sample_input, device, n_warmup: int = 3, n_runs: int = 20) -> float:
    """
    Mean single-sample inference latency in milliseconds, on `device`.
    Warmup runs are excluded from the timing to avoid measuring lazy
    initialization (cudnn autotune, first-call overhead, etc.).
    """
    import torch

    model.eval()
    sample_input = sample_input.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000.0


def model_size_mb(model) -> float:
    """Size of model parameters on disk, in MB (float32 assumed)."""
    n_params = sum(p.numel() for p in model.parameters())
    return (n_params * 4) / (1024 ** 2)
