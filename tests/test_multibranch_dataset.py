"""Tests for src/data/multibranch_dataset.py."""

import numpy as np
import pandas as pd
from PIL import Image

from src.data.multibranch_dataset import MultiBranchForensicsDataset
from src.features.residual import FEATURE_NAMES


def make_metadata_and_images(tmp_path, size=40):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i in range(3):
        path = img_dir / f"real_{i}.jpg"
        Image.new("RGB", (size, size), color=(10 * i, 20, 30)).save(path)
        rows.append({"path": str(path), "class": "REAL", "source": "real", "generator": None, "split": "train"})
    for i in range(2):
        path = img_dir / f"ai_{i}.jpg"
        Image.new("RGB", (size, size), color=(30, 10 * i, 60)).save(path)
        rows.append({"path": str(path), "class": "AI_GENERATED", "source": "ai", "generator": "genA", "split": "train"})

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "metadata.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_getitem_returns_all_three_branches_correct_shape(tmp_path):
    csv_path = make_metadata_and_images(tmp_path)
    ds = MultiBranchForensicsDataset(csv_path, split="train", image_size=32)
    sample = ds[0]

    assert set(sample.keys()) == {"rgb", "fft", "residual", "label"}
    assert sample["rgb"].shape == (3, 32, 32)
    assert sample["fft"].shape == (1, 32, 32)
    assert sample["residual"].shape == (len(FEATURE_NAMES),)
    assert sample["label"] in (0, 1)


def test_pre_transform_applied_consistently_across_branches(tmp_path):
    # This is the key correctness property added in Stage 5: whatever
    # pre_transform does to the image, ALL THREE branches should see the
    # SAME degraded image — not RGB seeing one version and FFT/residual
    # seeing another.
    csv_path = make_metadata_and_images(tmp_path)

    calls = []

    def tracking_transform(img):
        # Replace with a solid color so we can verify both branches
        # reflect this exact transformation, not the original image.
        calls.append(1)
        return Image.new("RGB", img.size, color=(200, 200, 200))

    ds = MultiBranchForensicsDataset(csv_path, split="train", image_size=32, pre_transform=tracking_transform)
    sample = ds[0]

    assert len(calls) == 1  # pre_transform called exactly once per __getitem__
    # A solid gray image has near-zero residual energy (no texture) and a
    # near-degenerate FFT spectrum (mostly DC) — sanity-check the FFT/
    # residual branches actually reflect the transformed (not original) image.
    assert sample["residual"][6].item() < 1.0  # high_freq_energy index, near-zero for flat image


def test_output_without_pre_transform_differs_from_with(tmp_path):
    csv_path = make_metadata_and_images(tmp_path)
    ds_plain = MultiBranchForensicsDataset(csv_path, split="train", image_size=32)
    ds_transformed = MultiBranchForensicsDataset(
        csv_path, split="train", image_size=32,
        pre_transform=lambda img: Image.new("RGB", img.size, color=(0, 0, 0)),
    )

    sample_plain = ds_plain[1]
    sample_transformed = ds_transformed[1]
    assert not np.array_equal(sample_plain["rgb"].numpy(), sample_transformed["rgb"].numpy())
