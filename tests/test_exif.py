"""Tests for src/inference/exif.py."""

import io

import piexif
from PIL import Image

from src.inference.exif import extract_metadata


def test_image_with_no_exif_returns_has_exif_false():
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    result = extract_metadata(img)
    assert result.has_exif is False
    assert result.camera_make is None
    assert result.camera_model is None
    assert "NOT evidence of AI generation" in result.note


def test_image_with_exif_extracts_fields():
    img = Image.new("RGB", (32, 32), color=(50, 50, 50))
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: "TestCameraCo",
            piexif.ImageIFD.Model: "TestModel X100",
            piexif.ImageIFD.Software: "TestEditor 1.0",
        }
    }
    exif_bytes = piexif.dump(exif_dict)

    buf = io.BytesIO()
    img.save(buf, format="jpeg", exif=exif_bytes)
    buf.seek(0)
    loaded = Image.open(buf)

    result = extract_metadata(loaded)
    assert result.has_exif is True
    assert result.camera_make == "TestCameraCo"
    assert result.camera_model == "TestModel X100"
    assert result.software == "TestEditor 1.0"


def test_to_dict_includes_disclaimer_note():
    img = Image.new("RGB", (16, 16))
    result = extract_metadata(img)
    d = result.to_dict()
    assert "note" in d
    assert "NOT evidence" in d["note"]
