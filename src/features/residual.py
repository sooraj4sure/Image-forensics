"""
Noise/residual branch (brief §6): "high-frequency residual (original −
low-pass), features such as high-freq energy, local variance, noise stats,
edge/texture stats, channel relationships. Justify each feature; measure
whether it helps."

Each feature below is justified individually rather than dumped in as an
unexplained vector, since the brief is explicit that features need
justification, not just inclusion.
"""

from __future__ import annotations

import cv2
import numpy as np

FEATURE_NAMES = [
    "residual_mean_abs_r", "residual_mean_abs_g", "residual_mean_abs_b",
    "residual_std_r", "residual_std_g", "residual_std_b",
    "high_freq_energy",
    "local_variance_mean", "local_variance_std",
    "edge_density",
    "corr_rg", "corr_gb", "corr_rb",
    "laplacian_variance",
]


def compute_residual_features(rgb_image: np.ndarray) -> np.ndarray:
    """
    Args:
        rgb_image: (H, W, 3) uint8 or float array, values in [0, 255].

    Returns:
        1D float32 array, length len(FEATURE_NAMES), in the fixed order above.

    Feature rationale:
    - residual_mean_abs / residual_std (per channel): the core "noise print"
      signal. Real camera sensor noise has a fairly consistent statistical
      profile; many generators either under- or over-produce high-frequency
      residual relative to real photos, so both the magnitude and spread of
      the residual are informative, not just one summary stat.
    - high_freq_energy: sum of squared residual — a single scalar summary of
      how much high-frequency content survives after low-pass removal.
      Complements the per-channel mean/std with an overall-image view.
    - local_variance_mean/std: texture uniformity. Real photos have naturally
      uneven local variance (varied textures — skin, foliage, fabric); some
      generators produce unnaturally uniform or unnaturally patchy local
      variance across the image.
    - edge_density: fraction of pixels flagged as edges (Canny). Generators
      sometimes over-smooth or over-sharpen edges relative to real photos.
    - corr_rg/gb/rb: cross-channel correlation of the *residual* (not the
      raw image). Real sensor noise has characteristic cross-channel
      correlation from the camera's color filter array / demosaicing;
      synthetic images generated per-channel or through different pipelines
      can have different cross-channel residual correlation.
    - laplacian_variance: a classic blur/sharpness proxy (variance of the
      Laplacian). Included because it's cheap, well-established in image
      forensics, and captures a different aspect (overall sharpness) than
      the localized edge_density measure.
    """
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Low-pass via Gaussian blur; residual = original - low-pass, per channel.
    low_pass = cv2.GaussianBlur(rgb_image, (5, 5), sigmaX=1.0)
    residual = rgb_image.astype(np.float32) - low_pass.astype(np.float32)

    residual_mean_abs = np.abs(residual).mean(axis=(0, 1))  # (3,)
    residual_std = residual.std(axis=(0, 1))  # (3,)
    high_freq_energy = float(np.mean(residual ** 2))

    # Local variance: variance within small windows, computed via
    # mean(x^2) - mean(x)^2 with box filters (fast, avoids per-pixel loops).
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(7, 7))
    mean_sq = cv2.boxFilter(gray_f ** 2, ddepth=-1, ksize=(7, 7))
    local_var = np.clip(mean_sq - mean ** 2, a_min=0, a_max=None)
    local_variance_mean = float(local_var.mean())
    local_variance_std = float(local_var.std())

    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edge_density = float((edges > 0).mean())

    r_res, g_res, b_res = residual[..., 0].ravel(), residual[..., 1].ravel(), residual[..., 2].ravel()

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-8 or b.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    corr_rg = safe_corr(r_res, g_res)
    corr_gb = safe_corr(g_res, b_res)
    corr_rb = safe_corr(r_res, b_res)

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    features = np.array(
        [
            residual_mean_abs[0], residual_mean_abs[1], residual_mean_abs[2],
            residual_std[0], residual_std[1], residual_std[2],
            high_freq_energy,
            local_variance_mean, local_variance_std,
            edge_density,
            corr_rg, corr_gb, corr_rb,
            laplacian_variance,
        ],
        dtype=np.float32,
    )
    assert len(features) == len(FEATURE_NAMES)
    return features
