"""Tests for src/evaluation/metrics.py using hand-checkable synthetic predictions."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, model_size_mb


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.05, 0.9, 0.95])

    result = compute_metrics(y_true, y_pred, y_prob)
    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.roc_auc == 1.0
    assert result.confusion_matrix.tolist() == [[2, 0], [0, 2]]


def test_all_wrong_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.95, 0.1, 0.05])

    result = compute_metrics(y_true, y_pred, y_prob)
    assert result.accuracy == 0.0
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_per_class_metrics_present():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.8, 0.9])

    result = compute_metrics(y_true, y_pred, y_prob)
    assert result.per_class_metrics["REAL"]["n"] == 3
    assert result.per_class_metrics["AI_GENERATED"]["n"] == 2
    # 2 of 3 REAL correctly predicted (2/3), 2 of 2 AI_GENERATED correct (1.0)
    assert abs(result.per_class_metrics["REAL"]["accuracy"] - (2 / 3)) < 1e-9
    assert result.per_class_metrics["AI_GENERATED"]["accuracy"] == 1.0


def test_raises_on_single_class():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.6])

    with pytest.raises(ValueError, match="ROC-AUC requires both classes"):
        compute_metrics(y_true, y_pred, y_prob)


def test_to_dict_json_serializable():
    import json

    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_prob = np.array([0.1, 0.9])
    result = compute_metrics(y_true, y_pred, y_prob)
    json.dumps(result.to_dict())  # should not raise


def test_model_size_mb_matches_known_param_count():
    class FakeModel:
        def parameters(self):
            import torch

            # 1000 float32 params -> 1000*4 bytes = 3906.25 / 1024 KB... compute in MB
            return [torch.zeros(1000)]

    size = model_size_mb(FakeModel())
    expected = (1000 * 4) / (1024 ** 2)
    assert abs(size - expected) < 1e-9
