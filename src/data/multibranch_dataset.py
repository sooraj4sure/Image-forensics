"""
Multi-branch dataset for the ablation study (Stage 4). Always computes all
three representations (RGB tensor, FFT log-magnitude, residual features) —
the FusionModel picks which ones it actually uses per ablation config. This
keeps one dataset class serving all 4 configs rather than 4 near-duplicate
dataset classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset import CLASS_TO_LABEL, build_transforms
from src.features.frequency import fft_log_magnitude, to_grayscale
from src.features.residual import compute_residual_features
import pandas as pd


class MultiBranchForensicsDataset(Dataset):
    """
    Returns a dict per sample:
        {
            "rgb": FloatTensor (3, H, W) — normalized, same as ForensicsDataset
            "fft": FloatTensor (1, H, W) — log-magnitude spectrum, [0,1], resized to match rgb
            "residual": FloatTensor (14,) — engineered residual/noise features
            "label": int (0=REAL, 1=AI_GENERATED)
        }
    """

    def __init__(
        self,
        metadata_path: str | Path,
        split: str,
        image_size: int,
        pre_transform: Optional[callable] = None,
    ):
        """
        Args:
            pre_transform: optional callable(PIL.Image) -> PIL.Image applied
                immediately after loading, before any branch-specific
                processing. Used by the robustness suite (Stage 5) to apply
                a degradation (JPEG compression, blur, etc.) consistently to
                what all three branches see — without this hook, RGB/FFT/
                residual branches could see three different "versions" of
                the same degradation, which would confound the results.
        """
        df = pd.read_csv(metadata_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(
                f"No rows found for split={split!r} in {metadata_path}. "
                f"Available splits: {sorted(df['split'].unique().tolist())}"
            )
        self.image_size = image_size
        self.rgb_transform = build_transforms(image_size, train=(split == "train"))
        self.pre_transform = pre_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pil_image = Image.open(row["path"]).convert("RGB")

        if self.pre_transform is not None:
            pil_image = self.pre_transform(pil_image)

        rgb_tensor = self.rgb_transform(pil_image)

        # For FFT/residual, work from a resized numpy array independent of
        # the RGB branch's own (possibly augmented) tensor — augmentation
        # like color jitter shouldn't leak into the forensic feature branches.
        resized = pil_image.resize((self.image_size, self.image_size))
        np_image = np.array(resized).astype(np.float32)

        gray = to_grayscale(np_image)
        fft_spectrum = fft_log_magnitude(gray)
        fft_tensor = torch.from_numpy(fft_spectrum).unsqueeze(0)  # (1, H, W)

        residual_feats = compute_residual_features(np_image)
        residual_tensor = torch.from_numpy(residual_feats)

        label = CLASS_TO_LABEL[row["class"]]

        return {
            "rgb": rgb_tensor,
            "fft": fft_tensor,
            "residual": residual_tensor,
            "label": label,
        }
