"""
PyTorch Dataset for ImageForensics AI.

Reads data/metadata/metadata.csv (produced by src/data/build_metadata.py)
and serves (image_tensor, label) pairs for a given split. Kept separate
from build_metadata.py: this module never touches the filesystem beyond
reading the CSV + loading individual images referenced in it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_TO_LABEL = {"REAL": 0, "AI_GENERATED": 1}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}


def build_transforms(image_size: int, train: bool) -> Callable:
    """
    Standard ImageNet-style normalization (matches ResNet-18 pretraining).

    Augmentation is deliberately light and avoids anything that could erase
    the forensic artifacts this project is trying to detect (brief §6:
    "avoid augmentations that erase the very artifacts you're trying to
    detect, e.g. heavy blur/compression in training"). Horizontal flip and
    mild color jitter don't touch frequency/noise characteristics; we do
    NOT apply blur, heavy JPEG re-compression, or resize-degradation here —
    those are reserved for the dedicated robustness evaluation (§7), where
    degrading the input on purpose is the point.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ForensicsDataset(Dataset):
    """
    Args:
        metadata_path: path to metadata.csv (columns: path, class, source, generator, split)
        split: one of "train", "val", "test", "unseen_test" — filters metadata rows
        image_size: resize target (square)
        transform: optional override; defaults to build_transforms(image_size, train=(split=="train"))
    """

    def __init__(
        self,
        metadata_path: str | Path,
        split: str,
        image_size: int,
        transform: Optional[Callable] = None,
    ):
        df = pd.read_csv(metadata_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(
                f"No rows found for split={split!r} in {metadata_path}. "
                f"Available splits: {sorted(df['split'].unique().tolist())}"
            )
        self.transform = transform or build_transforms(
            image_size, train=(split == "train")
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = CLASS_TO_LABEL[row["class"]]
        return image, label
