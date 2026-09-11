"""
Tests for src/evaluation/unseen_generator.py's single-class handling.
Uses a fake model + fake dataloader (plain tensors) rather than real
images/checkpoints — this tests the METRICS LOGIC, not the model.
"""

import numpy as np
import pytest
import torch

from src.evaluation.unseen_generator import evaluate_ai_recall, predict_split


class FakeModel(torch.nn.Module):
    """Returns fixed logits regardless of input — lets us control predictions exactly."""

    def __init__(self, fixed_preds: list[int]):
        super().__init__()
        self.fixed_preds = fixed_preds
        self.call_count = 0
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))  # so .eval()/.to() work

    def forward(self, batch):
        n = batch["rgb"].shape[0]
        preds = self.fixed_preds[self.call_count : self.call_count + n]
        self.call_count += n
        logits = torch.zeros(n, 2)
        for i, p in enumerate(preds):
            logits[i, p] = 10.0  # confidently predict class p
        return logits


def make_fake_batch(labels: list[int], n_residual_features=14):
    n = len(labels)
    return {
        "rgb": torch.randn(n, 3, 8, 8),
        "fft": torch.randn(n, 1, 8, 8),
        "residual": torch.randn(n, n_residual_features),
        "label": torch.tensor(labels),
    }


def test_evaluate_ai_recall_all_correct():
    model = FakeModel(fixed_preds=[1, 1, 1, 1])  # always predicts AI_GENERATED
    loader = [make_fake_batch([1, 1, 1, 1])]  # all true labels are AI_GENERATED
    device = torch.device("cpu")

    result = evaluate_ai_recall(model, loader, device)
    assert result["n_samples"] == 4
    assert result["ai_detection_rate"] == 1.0


def test_evaluate_ai_recall_partial_detection():
    model = FakeModel(fixed_preds=[1, 0, 1, 0])  # misses 2 of 4
    loader = [make_fake_batch([1, 1, 1, 1])]
    device = torch.device("cpu")

    result = evaluate_ai_recall(model, loader, device)
    assert result["ai_detection_rate"] == 0.5


def test_evaluate_ai_recall_raises_on_mixed_classes():
    # This is the exact bug found during the Stage 5 smoke test:
    # evaluate_ai_recall must reject mixed-class input rather than silently
    # computing a misleading number, and evaluate_split (full compute_metrics)
    # must be used instead for mixed splits.
    model = FakeModel(fixed_preds=[1, 0])
    loader = [make_fake_batch([1, 0])]  # mixed: one REAL, one AI_GENERATED
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="AI_GENERATED-only split"):
        evaluate_ai_recall(model, loader, device)


def test_predict_split_returns_correct_shapes():
    model = FakeModel(fixed_preds=[1, 0, 1])
    loader = [make_fake_batch([1, 1, 0])]
    device = torch.device("cpu")

    y_true, y_pred, y_prob = predict_split(model, loader, device)
    assert y_true.shape == (3,)
    assert y_pred.shape == (3,)
    assert y_prob.shape == (3,)
    assert np.array_equal(y_true, np.array([1, 1, 0]))
    assert np.array_equal(y_pred, np.array([1, 0, 1]))
