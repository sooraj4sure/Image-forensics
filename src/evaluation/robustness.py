"""
Robustness testing (brief §7, trimmed scope): on the FINAL FUSION MODEL
ONLY (not re-run across all 4 ablation configs — explicitly out of scope
per §3/§7), measure accuracy/F1 delta under:
    - JPEG compression at 3 quality levels
    - 1 resize scale
    - 1 crop setting
    - 1 blur level
    - 1 added-noise level

Each degradation is implemented as a PIL.Image -> PIL.Image transform,
applied via MultiBranchForensicsDataset's pre_transform hook so RGB/FFT/
residual branches all see the identically-degraded image.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter


def jpeg_compress(quality: int) -> Callable[[Image.Image], Image.Image]:
    """Re-encode through JPEG at the given quality and decode back."""

    def _transform(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return _transform


def resize_degrade(scale: float) -> Callable[[Image.Image], Image.Image]:
    """Downscale then upscale back to original size — simulates a low-res source."""

    def _transform(img: Image.Image) -> Image.Image:
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        return small.resize((w, h))

    return _transform


def center_crop(fraction: float) -> Callable[[Image.Image], Image.Image]:
    """Center-crop to `fraction` of each dimension, then resize back to original size."""

    def _transform(img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w, new_h = int(w * fraction), int(h * fraction)
        left, top = (w - new_w) // 2, (h - new_h) // 2
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h))

    return _transform


def gaussian_blur(kernel_size: int) -> Callable[[Image.Image], Image.Image]:
    """PIL's GaussianBlur radius is not identical to a kernel size, but kernel_size
    is used as a monotonic proxy for blur strength (radius = kernel_size / 2)."""

    def _transform(img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=kernel_size / 2))

    return _transform


def add_gaussian_noise(std: float) -> Callable[[Image.Image], Image.Image]:
    """Add zero-mean Gaussian noise, std expressed as a fraction of the [0,255] range."""

    def _transform(img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(loc=0.0, scale=std * 255.0, size=arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    return _transform


@dataclass
class RobustnessCase:
    name: str
    transform: Callable[[Image.Image], Image.Image]


def build_robustness_cases(robustness_cfg) -> list[RobustnessCase]:
    """Build the fixed, capped set of robustness cases from config.robustness."""
    cases = []
    for q in robustness_cfg.jpeg_quality_levels:
        cases.append(RobustnessCase(f"jpeg_q{q}", jpeg_compress(q)))
    cases.append(RobustnessCase(f"resize_{robustness_cfg.resize_scale}", resize_degrade(robustness_cfg.resize_scale)))
    cases.append(RobustnessCase(f"crop_{robustness_cfg.crop_fraction}", center_crop(robustness_cfg.crop_fraction)))
    cases.append(RobustnessCase(f"blur_k{robustness_cfg.blur_kernel}", gaussian_blur(robustness_cfg.blur_kernel)))
    cases.append(RobustnessCase(f"noise_{robustness_cfg.noise_std}", add_gaussian_noise(robustness_cfg.noise_std)))
    return cases
