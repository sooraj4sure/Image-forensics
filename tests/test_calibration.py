"""Tests for src/evaluation/calibration.py."""

import numpy as np
import torch

from src.evaluation.calibration import TemperatureScaler, compute_ece_from_predictions


def test_perfectly_calibrated_has_zero_ece():
    # 100 predictions all at confidence 0.7, with exactly 70 correct ->
    # accuracy == confidence in that bin -> ECE should be ~0.
    rng = np.random.default_rng(0)
    n = 100
    confidence = np.full(n, 0.7)
    correct = np.zeros(n)
    correct_idx = rng.choice(n, size=70, replace=False)
    correct[correct_idx] = 1.0

    ece, bins = compute_ece_from_predictions(correct, confidence, n_bins=10)
    assert ece < 1e-6


def test_overconfident_model_has_high_ece():
    # Confidence always 0.99 but only 50% correct -> badly miscalibrated.
    n = 100
    confidence = np.full(n, 0.99)
    correct = np.zeros(n)
    correct[:50] = 1.0

    ece, bins = compute_ece_from_predictions(correct, confidence, n_bins=10)
    assert ece > 0.4  # should be close to |0.5 - 0.99| = 0.49


def test_empty_bins_have_zero_count_not_nan():
    # All confidence values in one narrow range -> most bins empty.
    n = 20
    confidence = np.full(n, 0.55)
    correct = np.ones(n)

    ece, bins = compute_ece_from_predictions(correct, confidence, n_bins=10)
    empty_bins = [b for b in bins if b.count == 0]
    assert len(empty_bins) > 0
    for b in empty_bins:
        assert not np.isnan(b.confidence)
        assert not np.isnan(b.accuracy)


def test_temperature_scaling_reduces_ece_on_overconfident_logits():
    # Build a synthetic overconfident classifier: logits scaled up hugely
    # relative to how separable the classes actually are.
    torch.manual_seed(0)
    n = 200
    labels = torch.randint(0, 2, (n,))
    # true separation is modest, but we scale logits up artificially to
    # simulate an overconfident (under-regularized) network.
    base_logits = torch.randn(n, 2) * 0.5
    base_logits[torch.arange(n), labels] += 1.0  # nudge toward correct class
    overconfident_logits = base_logits * 10.0  # artificially sharpen softmax

    def ece_of(logits):
        probs = torch.softmax(logits, dim=1)
        confidence, preds = probs.max(dim=1)
        correct = (preds == labels).float().numpy()
        ece, _ = compute_ece_from_predictions(correct, confidence.detach().numpy(), n_bins=10)
        return ece

    ece_before = ece_of(overconfident_logits)

    scaler = TemperatureScaler()
    scaler.fit(overconfident_logits, labels)
    ece_after = ece_of(scaler(overconfident_logits))

    assert ece_after < ece_before
    assert scaler.temperature.item() > 1.0  # should have learned to soften, not sharpen
