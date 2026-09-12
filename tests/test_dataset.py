"""Tests for src/data/dataset.py — preprocessing pipeline (brief §11 requires
coverage for preprocessing specifically, not just feature extraction)."""

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import CLASS_TO_LABEL, ForensicsDataset, build_transforms


def make_metadata_and_images(tmp_path, n_real=4, n_ai=3):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i in range(n_real):
        path = img_dir / f"real_{i}.jpg"
        Image.new("RGB", (40, 40), color=(10 * i, 20, 30)).save(path)
        rows.append({"path": str(path), "class": "REAL", "source": "real", "generator": None, "split": "train" if i < n_real - 1 else "val"})
    for i in range(n_ai):
        path = img_dir / f"ai_{i}.jpg"
        Image.new("RGB", (40, 40), color=(30, 10 * i, 60)).save(path)
        rows.append({"path": str(path), "class": "AI_GENERATED", "source": "ai", "generator": "genA", "split": "train" if i < n_ai - 1 else "test"})

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "metadata.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_build_transforms_train_vs_eval_output_shape():
    train_t = build_transforms(image_size=64, train=True)
    eval_t = build_transforms(image_size=64, train=False)
    img = Image.new("RGB", (100, 100), color=(50, 60, 70))

    train_tensor = train_t(img)
    eval_tensor = eval_t(img)
    assert train_tensor.shape == (3, 64, 64)
    assert eval_tensor.shape == (3, 64, 64)
    assert isinstance(train_tensor, torch.Tensor)


def test_dataset_filters_by_split(tmp_path):
    csv_path = make_metadata_and_images(tmp_path, n_real=4, n_ai=3)
    train_ds = ForensicsDataset(csv_path, split="train", image_size=32)
    val_ds = ForensicsDataset(csv_path, split="val", image_size=32)
    test_ds = ForensicsDataset(csv_path, split="test", image_size=32)

    assert len(train_ds) == 5  # 3 real-train + 2 ai-train
    assert len(val_ds) == 1
    assert len(test_ds) == 1


def test_dataset_getitem_returns_correct_shape_and_label(tmp_path):
    csv_path = make_metadata_and_images(tmp_path)
    ds = ForensicsDataset(csv_path, split="train", image_size=32)
    image, label = ds[0]
    assert image.shape == (3, 32, 32)
    assert label in (0, 1)
    assert label == CLASS_TO_LABEL["REAL"]  # first row is REAL


def test_dataset_raises_on_missing_split(tmp_path):
    csv_path = make_metadata_and_images(tmp_path)
    with pytest.raises(ValueError, match="No rows found for split"):
        ForensicsDataset(csv_path, split="unseen_test", image_size=32)


def test_dataset_normalizes_to_expected_range(tmp_path):
    # After ImageNet normalization, values should NOT be in raw [0,1] or
    # [0,255] range anymore — a basic sanity check that normalize() ran.
    csv_path = make_metadata_and_images(tmp_path)
    ds = ForensicsDataset(csv_path, split="train", image_size=32)
    image, _ = ds[0]
    assert image.min() < 0 or image.max() > 1.5  # normalized, not raw [0,1]
