"""Tests for src/models/backbone.py."""

import pytest
import torch

from src.config import load_config
from src.models.backbone import ResNetBinaryClassifier, build_model


def test_forward_output_shape():
    model = ResNetBinaryClassifier(pretrained=False, num_classes=2, dropout=0.3)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_dropout_present_in_head():
    model = ResNetBinaryClassifier(pretrained=False, dropout=0.5)
    assert isinstance(model.backbone.fc[0], torch.nn.Dropout)
    assert model.backbone.fc[0].p == 0.5


def test_build_model_from_config():
    cfg = load_config()
    cfg = cfg.model_copy(deep=True)
    cfg.model.pretrained = False
    model = build_model(cfg.model)
    assert isinstance(model, ResNetBinaryClassifier)
    x = torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size)
    logits = model(x)
    assert logits.shape == (2, cfg.model.num_classes)


def test_build_model_rejects_unsupported_backbone():
    cfg = load_config()
    cfg = cfg.model_copy(deep=True)
    cfg.model.backbone = "resnet50"
    with pytest.raises(NotImplementedError, match="Only resnet18"):
        build_model(cfg.model)
