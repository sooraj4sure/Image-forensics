"""
Baseline RGB-only model (Stage 3): fine-tuned ResNet-18 binary classifier.

This is deliberately the SIMPLEST config in the eventual 4-way ablation
(brief §6/§2.1: "RGB-only vs RGB+FFT/DCT vs RGB+Noise/Residual vs Full
Fusion"). The frequency and noise/residual branches, and the fusion module
that combines them, are separate additions in Stage 4 — kept out of this
file so RGB-only stays a clean, independently-testable baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNetBinaryClassifier(nn.Module):
    """
    Pretrained ResNet-18 with its final fc layer replaced for binary
    classification (REAL=0, AI_GENERATED=1), plus a dropout layer before
    the head (config: model.dropout) to reduce overfitting on a modestly
    sized dataset.
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (batch, num_classes). Apply softmax outside if needed."""
        return self.backbone(x)


def build_model(model_cfg) -> ResNetBinaryClassifier:
    """Construct a model from the `model` section of configs/config.yaml."""
    if model_cfg.backbone != "resnet18":
        raise NotImplementedError(
            f"Only resnet18 is implemented so far; got backbone={model_cfg.backbone!r}"
        )
    return ResNetBinaryClassifier(
        pretrained=model_cfg.pretrained,
        num_classes=model_cfg.num_classes,
        dropout=model_cfg.dropout,
    )
