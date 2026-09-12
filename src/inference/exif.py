"""
Metadata/EXIF extraction (brief §9): "Supporting signal only, shown
separately from the model's visual prediction... Never treat missing/
stripped metadata as evidence of AI generation — state this explicitly
in both the code comments and the UI copy."

This module only extracts and reports what's present; it never emits a
verdict or contributes to the model's prediction. The "never evidence of
AI generation" statement belongs in api/ and app/ UI copy too — repeating
it here as a comment is not sufficient on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image
from PIL.ExifTags import TAGS


@dataclass
class MetadataObservations:
    has_exif: bool
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    datetime_original: Optional[str] = None
    note: str = (
        "Metadata is supporting context only. Missing or stripped metadata is "
        "common for real photos shared via messaging apps, social media, or "
        "screenshots, and is NOT evidence of AI generation."
    )

    def to_dict(self) -> dict:
        return {
            "has_exif": self.has_exif,
            "camera_make": self.camera_make,
            "camera_model": self.camera_model,
            "software": self.software,
            "datetime_original": self.datetime_original,
            "note": self.note,
        }


def extract_metadata(pil_image: Image.Image) -> MetadataObservations:
    """
    Extract whatever EXIF fields are present. Returns has_exif=False (with
    all fields None) if the image has no EXIF block at all — a very common,
    unremarkable case, not flagged as suspicious anywhere in this function.
    """
    try:
        exif_data = pil_image._getexif()
    except AttributeError:
        exif_data = None

    if not exif_data:
        return MetadataObservations(has_exif=False)

    tag_map = {TAGS.get(tag_id, tag_id): value for tag_id, value in exif_data.items()}

    def clean(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="ignore")
            except Exception:
                return None
        value = str(value).strip().strip("\x00")
        return value or None

    return MetadataObservations(
        has_exif=True,
        camera_make=clean(tag_map.get("Make")),
        camera_model=clean(tag_map.get("Model")),
        software=clean(tag_map.get("Software")),
        datetime_original=clean(tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")),
    )
