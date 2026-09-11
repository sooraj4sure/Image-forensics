"""Tests for src/features/residual.py using synthetic images with known noise/edge properties."""

import numpy as np

from src.features.residual import FEATURE_NAMES, compute_residual_features


def test_output_shape_and_length():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    features = compute_residual_features(img)
    assert features.shape == (len(FEATURE_NAMES),)
    assert features.dtype == np.float32


def test_smooth_image_has_low_high_freq_energy():
    # Smooth gradient: minimal high-frequency content.
    smooth = np.tile(np.linspace(0, 255, 64), (64, 1))
    smooth_rgb = np.stack([smooth] * 3, axis=-1).astype(np.uint8)
    features = compute_residual_features(smooth_rgb)
    energy_idx = FEATURE_NAMES.index("high_freq_energy")
    assert features[energy_idx] < 5.0  # near-zero for a smooth gradient


def test_noisy_image_has_higher_high_freq_energy_than_smooth():
    rng = np.random.default_rng(0)
    smooth = np.tile(np.linspace(0, 255, 64), (64, 1))
    smooth_rgb = np.stack([smooth] * 3, axis=-1).astype(np.uint8)

    noisy_rgb = (rng.random((64, 64, 3)) * 255).astype(np.uint8)

    smooth_features = compute_residual_features(smooth_rgb)
    noisy_features = compute_residual_features(noisy_rgb)

    energy_idx = FEATURE_NAMES.index("high_freq_energy")
    assert noisy_features[energy_idx] > smooth_features[energy_idx]


def test_features_are_finite():
    img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
    features = compute_residual_features(img)
    assert np.all(np.isfinite(features))


def test_correlation_features_bounded():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    features = compute_residual_features(img)
    for name in ["corr_rg", "corr_gb", "corr_rb"]:
        val = features[FEATURE_NAMES.index(name)]
        assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6


def test_handles_float_input_by_casting():
    img_float = np.random.rand(40, 40, 3) * 255.0  # float, not uint8
    features = compute_residual_features(img_float)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))
