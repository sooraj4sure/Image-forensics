"""Tests for src/features/frequency.py using synthetic images with known spectral properties."""

import numpy as np

from src.features.frequency import dct_magnitude, fft_log_magnitude, to_grayscale


def test_fft_output_shape_and_range():
    img = np.random.rand(64, 64).astype(np.float32)
    spectrum = fft_log_magnitude(img)
    assert spectrum.shape == (64, 64)
    assert spectrum.min() >= 0.0
    assert spectrum.max() <= 1.0 + 1e-6


def test_fft_constant_image_dc_spike():
    # A flat image puts ALL energy into the single DC bin; every other
    # frequency bin is exactly zero. So after normalization: center (DC,
    # after fftshift) = 1.0, everywhere else = 0.0. This is the opposite of
    # a degenerate/uniform spectrum — it's maximally concentrated.
    img = np.full((32, 32), 128.0, dtype=np.float32)
    spectrum = fft_log_magnitude(img)
    assert spectrum.shape == (32, 32)
    center = spectrum[16, 16]
    assert center == 1.0
    # every other bin should be 0
    spectrum_no_center = spectrum.copy()
    spectrum_no_center[16, 16] = 0.0
    assert np.all(spectrum_no_center == 0.0)


def test_fft_truly_degenerate_zero_image_returns_zero_safely():
    # An all-zero image: FFT magnitude is exactly zero everywhere, including
    # DC -> log1p(0)=0 everywhere -> hi-lo=0 -> hits the safe zero-fallback.
    img = np.zeros((32, 32), dtype=np.float32)
    spectrum = fft_log_magnitude(img)
    assert spectrum.shape == (32, 32)
    assert np.all(spectrum == 0.0)


def test_fft_periodic_pattern_produces_sharp_peak():
    # A pure sine wave pattern should produce a small number of dominant
    # frequency peaks, i.e. the normalized spectrum should NOT be
    # uniformly spread — most energy concentrated in a few bins.
    x = np.linspace(0, 4 * np.pi, 64)
    img = np.tile(np.sin(x), (64, 1)).astype(np.float32)
    spectrum = fft_log_magnitude(img)
    # top 1% of pixels should carry a disproportionate share of total energy
    flat = spectrum.ravel()
    threshold = np.percentile(flat, 99)
    top_energy_share = flat[flat >= threshold].sum() / (flat.sum() + 1e-8)
    assert top_energy_share > 0.02  # concentrated, not uniform


def test_dct_output_shape_and_range():
    img = np.random.rand(48, 48).astype(np.float32)
    spectrum = dct_magnitude(img)
    assert spectrum.shape == (48, 48)
    assert spectrum.min() >= 0.0
    assert spectrum.max() <= 1.0 + 1e-6


def test_to_grayscale_shape_and_known_value():
    # Pure red image: luma = 0.299 * 255
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[..., 0] = 255.0
    gray = to_grayscale(img)
    assert gray.shape == (10, 10)
    assert abs(gray[0, 0] - 0.299 * 255) < 1e-3
