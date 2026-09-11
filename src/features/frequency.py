"""
Frequency-domain representation for the frequency branch (brief §6:
"FFT and/or DCT representation; visualize real vs AI spectra; measure
whether it actually adds signal over RGB-only").

Using FFT log-magnitude here (not raw DCT coefficients) because it's easy
to feed into a small CNN as a single-channel "image" the same way the RGB
branch consumes RGB — keeps the two branches architecturally parallel.
DCT is noted as an easy swap-in alternative (see dct_magnitude below) if
the ablation experiments in Stage 4 show FFT isn't adding signal and DCT
might do better; brief explicitly says to measure this, not assume.
"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dct


def fft_log_magnitude(gray_image: np.ndarray) -> np.ndarray:
    """
    Args:
        gray_image: 2D float array (H, W), values in [0, 1] or [0, 255] — scale
            doesn't matter, only relative structure does after log-scaling.

    Returns:
        2D float array (H, W), log-scaled FFT magnitude spectrum, zero-centered
        (DC component at the center via fftshift), normalized to [0, 1].

    AI-generated images from upsampling-based generators (GANs, some diffusion
    decoders) often show characteristic periodic artifacts in the frequency
    domain that aren't visible in the spatial/RGB domain — that's the signal
    this branch is trying to give the model access to.
    """
    f = np.fft.fft2(gray_image)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)
    # log1p avoids log(0); the DC component otherwise dwarfs everything else
    log_magnitude = np.log1p(magnitude)

    lo, hi = log_magnitude.min(), log_magnitude.max()
    if hi - lo < 1e-8:
        # Degenerate case (e.g. a flat/constant image) — avoid divide-by-zero
        return np.zeros_like(log_magnitude, dtype=np.float32)
    normalized = (log_magnitude - lo) / (hi - lo)
    return normalized.astype(np.float32)


def dct_magnitude(gray_image: np.ndarray) -> np.ndarray:
    """
    2D DCT-II magnitude, log-scaled and normalized to [0, 1], same
    conventions as fft_log_magnitude. Provided as an alternative
    representation to compare against FFT — not used by default.
    """
    d = dct(dct(gray_image, axis=0, norm="ortho"), axis=1, norm="ortho")
    log_magnitude = np.log1p(np.abs(d))
    lo, hi = log_magnitude.min(), log_magnitude.max()
    if hi - lo < 1e-8:
        return np.zeros_like(log_magnitude, dtype=np.float32)
    return ((log_magnitude - lo) / (hi - lo)).astype(np.float32)


def to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """rgb_image: (H, W, 3) float array. Standard luma weighting."""
    return (
        0.299 * rgb_image[..., 0] + 0.587 * rgb_image[..., 1] + 0.114 * rgb_image[..., 2]
    ).astype(np.float32)
