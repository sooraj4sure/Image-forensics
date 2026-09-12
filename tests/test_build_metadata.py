"""Tests for src/data/build_metadata.py's file-scanning and dedup logic."""

import pandas as pd
from PIL import Image

from src.data.build_metadata import deduplicate, scan_raw_data


def make_raw_dir(tmp_path):
    raw = tmp_path / "raw"
    (raw / "real").mkdir(parents=True)
    (raw / "ai" / "genA").mkdir(parents=True)
    (raw / "ai" / "genB").mkdir(parents=True)

    for i in range(3):
        Image.new("RGB", (16, 16)).save(raw / "real" / f"r{i}.jpg")
    for i in range(2):
        Image.new("RGB", (16, 16)).save(raw / "ai" / "genA" / f"a{i}.png")
    Image.new("RGB", (16, 16)).save(raw / "ai" / "genB" / "b0.webp")

    # Non-image files should be ignored
    (raw / "real" / "notes.txt").write_text("not an image")

    return raw


def test_scan_raw_data_finds_all_images_with_correct_labels(tmp_path):
    raw = make_raw_dir(tmp_path)
    df = scan_raw_data(raw)

    assert len(df) == 6  # 3 real + 2 genA + 1 genB, notes.txt excluded
    assert (df["class"] == "REAL").sum() == 3
    assert (df["class"] == "AI_GENERATED").sum() == 3
    assert set(df[df["source"] == "ai"]["generator"].unique()) == {"genA", "genB"}
    assert df[df["source"] == "real"]["generator"].isna().all()


def test_scan_raw_data_handles_missing_directories_gracefully(tmp_path):
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()
    df = scan_raw_data(empty_raw)
    assert df.empty
    assert list(df.columns) == ["path", "class", "source", "generator"]


def test_scan_raw_data_ignores_non_image_extensions(tmp_path):
    raw = make_raw_dir(tmp_path)
    df = scan_raw_data(raw)
    assert not any(p.endswith(".txt") for p in df["path"])


def test_deduplicate_removes_exact_duplicate_paths(tmp_path):
    raw = make_raw_dir(tmp_path)
    df = scan_raw_data(raw)
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    deduped = deduplicate(duplicated)
    assert len(deduped) == len(df)
