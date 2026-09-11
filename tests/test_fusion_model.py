"""
Tests for src/models/fusion.py. Uses pretrained=False everywhere — this
sandbox can't reach download.pytorch.org, and these tests only need to
verify architecture correctness (shapes, gradient flow), not real weights.
"""

import torch

from src.models.fusion import ABLATION_CONFIGS, FusionModel


def make_dummy_batch(batch_size=4, image_size=32):
    return {
        "rgb": torch.randn(batch_size, 3, image_size, image_size),
        "fft": torch.randn(batch_size, 1, image_size, image_size),
        "residual": torch.randn(batch_size, 14),
    }


def test_all_four_ablation_configs_forward_correctly():
    batch = make_dummy_batch()
    for name, branches in ABLATION_CONFIGS.items():
        model = FusionModel(**branches, num_classes=2, pretrained=False)
        logits = model(batch)
        assert logits.shape == (4, 2), f"config {name} produced wrong output shape"


def test_rgb_only_matches_expected_branch_flags():
    assert ABLATION_CONFIGS["rgb_only"] == {"use_rgb": True, "use_fft": False, "use_residual": False}
    assert ABLATION_CONFIGS["full_fusion"] == {"use_rgb": True, "use_fft": True, "use_residual": True}


def test_raises_when_no_branch_active():
    import pytest

    with pytest.raises(ValueError, match="At least one branch"):
        FusionModel(use_rgb=False, use_fft=False, use_residual=False)


def test_gradients_flow_through_all_active_branches_full_fusion():
    model = FusionModel(use_rgb=True, use_fft=True, use_residual=True, pretrained=False)
    batch = make_dummy_batch()
    logits = model(batch)
    loss = logits.sum()
    loss.backward()

    # Every branch should have received a gradient (not None, not all-zero)
    rgb_grad = next(model.rgb_branch.parameters()).grad
    fft_grad = next(model.fft_branch.parameters()).grad
    residual_grad = next(model.residual_branch.parameters()).grad

    assert rgb_grad is not None and rgb_grad.abs().sum() > 0
    assert fft_grad is not None and fft_grad.abs().sum() > 0
    assert residual_grad is not None and residual_grad.abs().sum() > 0


def test_fusion_dim_scales_with_active_branches():
    rgb_only = FusionModel(use_rgb=True, use_fft=False, use_residual=False, pretrained=False)
    full = FusionModel(use_rgb=True, use_fft=True, use_residual=True, pretrained=False)

    # classifier's input dim (first Linear layer) should be larger for full_fusion
    rgb_only_in_dim = rgb_only.classifier[1].in_features
    full_in_dim = full.classifier[1].in_features
    assert full_in_dim > rgb_only_in_dim
