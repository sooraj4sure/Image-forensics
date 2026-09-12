"""
Confidence calibration (brief §2.3/§6): "reliability diagram + Expected
Calibration Error (ECE), with temperature scaling if the raw softmax is
overconfident." Built here, ahead of the inference pipeline, because the
pipeline's "Calibrated Probability" step (brief §1) depends on a fitted
temperature — inference without this would just be reporting raw
(likely overconfident) softmax as if it meant something calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ReliabilityBin:
    bin_lower: float
    bin_upper: float
    confidence: float  # mean predicted probability in this bin
    accuracy: float  # fraction correct in this bin
    count: int


def compute_ece_from_predictions(
    correct: np.ndarray, confidence: np.ndarray, n_bins: int = 15
) -> tuple[float, list[ReliabilityBin]]:
    """
    Args:
        correct: shape (n,), 1.0 if the model's prediction was correct, else 0.0
        confidence: shape (n,), the model's confidence in its OWN prediction
            (i.e. max(softmax_probs), not P(class=1)) — this is the standard
            ECE definition (Guo et al. 2017).
        n_bins: number of equal-width bins over [0, 1]

    Returns:
        (ece, bins): ece = sum over bins of (count/n) * |accuracy - confidence|
    """
    assert correct.shape == confidence.shape
    n = len(correct)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include the right edge only in the last bin, standard convention
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)

        count = int(mask.sum())
        if count == 0:
            bins.append(ReliabilityBin(lo, hi, confidence=0.0, accuracy=0.0, count=0))
            continue

        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        bins.append(ReliabilityBin(lo, hi, confidence=bin_confidence, accuracy=bin_accuracy, count=count))
        ece += (count / n) * abs(bin_accuracy - bin_confidence)

    return ece, bins


class TemperatureScaler(nn.Module):
    """
    Single learned scalar T applied as logits / T before softmax. T > 1
    softens (de-confidences) an overconfident model; T < 1 sharpens it.
    Fit on a held-out set (val, NOT test) by minimizing NLL — see fit().
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        """
        Args:
            logits: (n, num_classes) raw model logits on the VALIDATION set
                (must not be test — fitting on test would leak test info
                into what's reported as a test-set calibration result)
            labels: (n,) ground truth

        Returns:
            final NLL loss value (for logging/debugging).
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            final_loss = criterion(self.forward(logits), labels).item()
        return final_loss
