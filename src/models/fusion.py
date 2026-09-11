"""
Fusion architecture for the ablation study (brief §6): exactly 4 configs
built from the same three branches by turning branches on/off, so the
comparison is apples-to-apples (same RGB backbone, same classifier head
shape logic, only the fused input differs):
    - rgb_only:      RGB branch only
    - rgb_fft:        RGB + frequency branch
    - rgb_residual:    RGB + noise/residual branch
    - full_fusion:      RGB + frequency + residual branch
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.features.residual import FEATURE_NAMES

RGB_FEATURE_DIM = 512       # resnet18 avgpool output dim
FFT_FEATURE_DIM = 64        # frequency branch CNN output dim
RESIDUAL_INPUT_DIM = len(FEATURE_NAMES)
RESIDUAL_FEATURE_DIM = 32   # residual branch MLP output dim


class RGBBranch(nn.Module):
    """ResNet-18 trunk (no final fc) — outputs a 512-d feature vector."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.trunk = nn.Sequential(*list(backbone.children())[:-1])  # drop fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.trunk(x)  # (B, 512, 1, 1)
        return feats.flatten(1)  # (B, 512)


class FrequencyBranch(nn.Module):
    """
    Small CNN over the single-channel FFT log-magnitude spectrum.
    Deliberately shallow (this is a supplementary signal, not the main
    backbone) — the point is to test whether frequency information adds
    anything over RGB, not to build a second full-size network.
    """

    def __init__(self, out_dim: int = FFT_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x).flatten(1)
        return self.fc(feats)


class ResidualBranch(nn.Module):
    """Small MLP over the fixed-length engineered residual/noise feature vector."""

    def __init__(self, in_dim: int = RESIDUAL_INPUT_DIM, out_dim: int = RESIDUAL_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


ABLATION_CONFIGS = {
    "rgb_only": {"use_rgb": True, "use_fft": False, "use_residual": False},
    "rgb_fft": {"use_rgb": True, "use_fft": True, "use_residual": False},
    "rgb_residual": {"use_rgb": True, "use_fft": False, "use_residual": True},
    "full_fusion": {"use_rgb": True, "use_fft": True, "use_residual": True},
}


class FusionModel(nn.Module):
    """
    Args:
        use_rgb / use_fft / use_residual: which branches are active. At
            least one must be True. Set via ABLATION_CONFIGS[config_name]
            or directly.
        num_classes, dropout, pretrained: as in the Stage 3 baseline model.
    """

    def __init__(
        self,
        use_rgb: bool = True,
        use_fft: bool = False,
        use_residual: bool = False,
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        if not (use_rgb or use_fft or use_residual):
            raise ValueError("At least one branch must be active.")

        self.use_rgb = use_rgb
        self.use_fft = use_fft
        self.use_residual = use_residual

        fusion_dim = 0
        if use_rgb:
            self.rgb_branch = RGBBranch(pretrained=pretrained)
            fusion_dim += RGB_FEATURE_DIM
        if use_fft:
            self.fft_branch = FrequencyBranch()
            fusion_dim += FFT_FEATURE_DIM
        if use_residual:
            self.residual_branch = ResidualBranch()
            fusion_dim += RESIDUAL_FEATURE_DIM

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys among "rgb", "fft", "residual" (as produced
                by MultiBranchForensicsDataset) — only the keys for active
                branches are actually read.
        """
        parts = []
        if self.use_rgb:
            parts.append(self.rgb_branch(batch["rgb"]))
        if self.use_fft:
            parts.append(self.fft_branch(batch["fft"]))
        if self.use_residual:
            parts.append(self.residual_branch(batch["residual"]))
        fused = torch.cat(parts, dim=1)
        return self.classifier(fused)


def build_fusion_model(config_name: str, model_cfg) -> FusionModel:
    """Construct a FusionModel for one of the 4 named ablation configs."""
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation config {config_name!r}. Valid: {list(ABLATION_CONFIGS)}"
        )
    branches = ABLATION_CONFIGS[config_name]
    return FusionModel(
        **branches,
        num_classes=model_cfg.num_classes,
        dropout=model_cfg.dropout,
        pretrained=model_cfg.pretrained,
    )
