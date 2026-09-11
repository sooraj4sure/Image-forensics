"""Tests for src/evaluation/robustness.py degradation transforms."""

import numpy as np
from PIL import Image

from src.evaluation.robustness import (
    add_gaussian_noise,
    build_robustness_cases,
    center_crop,
    gaussian_blur,
    jpeg_compress,
    resize_degrade,
)


def make_test_image(size=64):
    rng = np.random.default_rng(0)
    arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_jpeg_compress_preserves_size_and_changes_pixels():
    img = make_test_image()
    transform = jpeg_compress(quality=30)
    out = transform(img)
    assert out.size == img.size
    assert not np.array_equal(np.array(out), np.array(img))  # lossy compression should change pixels


def test_jpeg_lower_quality_causes_more_change():
    img = make_test_image()
    out_q85 = np.array(jpeg_compress(85)(img)).astype(np.float32)
    out_q30 = np.array(jpeg_compress(30)(img)).astype(np.float32)
    orig = np.array(img).astype(np.float32)

    diff_q85 = np.abs(out_q85 - orig).mean()
    diff_q30 = np.abs(out_q30 - orig).mean()
    assert diff_q30 > diff_q85  # lower quality = more distortion


def test_resize_degrade_preserves_final_size():
    img = make_test_image()
    transform = resize_degrade(scale=0.5)
    out = transform(img)
    assert out.size == img.size


def test_center_crop_preserves_final_size_and_changes_content():
    img = make_test_image()
    transform = center_crop(fraction=0.8)
    out = transform(img)
    assert out.size == img.size
    assert not np.array_equal(np.array(out), np.array(img))


def test_gaussian_blur_preserves_size_and_reduces_high_freq_variance():
    img = make_test_image()
    transform = gaussian_blur(kernel_size=5)
    out = transform(img)
    assert out.size == img.size
    # Blur should reduce local pixel-to-pixel variance (smoother image)
    orig_arr = np.array(img).astype(np.float32)
    blurred_arr = np.array(out).astype(np.float32)
    orig_grad = np.abs(np.diff(orig_arr, axis=0)).mean()
    blurred_grad = np.abs(np.diff(blurred_arr, axis=0)).mean()
    assert blurred_grad < orig_grad


def test_add_gaussian_noise_preserves_size_and_increases_variance():
    img = make_test_image()
    transform = add_gaussian_noise(std=0.1)
    out = transform(img)
    assert out.size == img.size
    orig_arr = np.array(img).astype(np.float32)
    noisy_arr = np.array(out).astype(np.float32)
    assert not np.array_equal(orig_arr, noisy_arr)


def test_build_robustness_cases_count_matches_config():
    class FakeRobustnessConfig:
        jpeg_quality_levels = [30, 60, 85]
        resize_scale = 0.5
        crop_fraction = 0.8
        blur_kernel = 5
        noise_std = 0.05

    cases = build_robustness_cases(FakeRobustnessConfig())
    # 3 jpeg levels + 1 resize + 1 crop + 1 blur + 1 noise = 7 total
    assert len(cases) == 7
    names = [c.name for c in cases]
    assert "jpeg_q30" in names and "jpeg_q60" in names and "jpeg_q85" in names
    assert "resize_0.5" in names
    assert "crop_0.8" in names
    assert "blur_k5" in names
    assert "noise_0.05" in names


def test_all_cases_produce_valid_images():
    class FakeRobustnessConfig:
        jpeg_quality_levels = [50]
        resize_scale = 0.5
        crop_fraction = 0.8
        blur_kernel = 3
        noise_std = 0.05

    img = make_test_image()
    cases = build_robustness_cases(FakeRobustnessConfig())
    for case in cases:
        out = case.transform(img)
        assert isinstance(out, Image.Image)
        assert out.size == img.size
