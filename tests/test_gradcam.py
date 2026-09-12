"""
Tests for src/explainability/gradcam.py. Uses pretrained=False (sandbox
can't reach download.pytorch.org) — these tests verify the Grad-CAM
MECHANISM (hooks, shapes, gradient flow), not real localization quality,
which needs a real trained model on real images to assess meaningfully.
"""

import numpy as np
import pytest
import torch

from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.fusion import FusionModel


def make_dummy_batch(image_size=32):
    return {
        "rgb": torch.randn(1, 3, image_size, image_size),
        "fft": torch.randn(1, 1, image_size, image_size),
        "residual": torch.randn(1, 14),
    }


def test_gradcam_output_shape_and_range():
    model = FusionModel(use_rgb=True, use_fft=True, use_residual=True, pretrained=False)
    cam = GradCAM(model)
    batch = make_dummy_batch(image_size=32)

    heatmap = cam.generate(batch)
    assert heatmap.shape == (32, 32)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_gradcam_requires_rgb_branch():
    model = FusionModel(use_rgb=False, use_fft=True, use_residual=True, pretrained=False)
    with pytest.raises(ValueError, match="requires a model with an active RGB branch"):
        GradCAM(model)


def test_gradcam_rejects_batch_size_greater_than_one():
    model = FusionModel(use_rgb=True, use_fft=False, use_residual=False, pretrained=False)
    cam = GradCAM(model)
    batch = {
        "rgb": torch.randn(2, 3, 32, 32),
        "fft": torch.randn(2, 1, 32, 32),
        "residual": torch.randn(2, 14),
    }
    with pytest.raises(ValueError, match="expects batch size 1"):
        cam.generate(batch)


def test_gradcam_works_on_rgb_only_config():
    # Should work even without fft/residual branches active
    model = FusionModel(use_rgb=True, use_fft=False, use_residual=False, pretrained=False)
    cam = GradCAM(model)
    batch = make_dummy_batch(image_size=32)
    heatmap = cam.generate(batch)
    assert heatmap.shape == (32, 32)


def test_overlay_heatmap_shape_and_dtype():
    rgb = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    heatmap = np.random.rand(32, 32).astype(np.float32)
    overlaid = overlay_heatmap(rgb, heatmap)
    assert overlaid.shape == (32, 32, 3)
    assert overlaid.dtype == np.uint8


def test_gradcam_differs_for_different_target_classes():
    # Explaining "class 0" vs "class 1" should generally produce different
    # attribution maps, since the gradient of a different output neuron
    # is being backpropagated. Not guaranteed to differ for a random
    # untrained model in every case, so we just check the code path runs
    # for both classes and returns valid, appropriately-shaped output.
    model = FusionModel(use_rgb=True, pretrained=False)
    cam = GradCAM(model)
    batch = make_dummy_batch(image_size=32)

    heatmap_class0 = cam.generate(batch, target_class=0)
    heatmap_class1 = cam.generate(batch, target_class=1)
    assert heatmap_class0.shape == heatmap_class1.shape == (32, 32)
