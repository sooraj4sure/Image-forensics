"""
Unified inference pipeline: Upload -> Preprocessing -> Multi-branch
Feature Extraction -> Fusion -> Classifier -> Calibrated Probability ->
Grad-CAM -> Forensic Report (brief §1). Both api/ (FastAPI) and app/
(Streamlit) call this single ForensicsPipeline rather than each
re-implementing prediction logic — keeps them consistent by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.config import AppConfig, load_config
from src.data.dataset import LABEL_TO_CLASS, build_transforms
from src.evaluation.calibration import TemperatureScaler
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.features.frequency import fft_log_magnitude, to_grayscale
from src.features.residual import compute_residual_features
from src.inference.exif import extract_metadata
from src.models.fusion import build_fusion_model

DISCLAIMER = (
    "This prediction is a statistical estimate from a machine learning model "
    "and is NOT definitive proof of an image's origin. Both false positives "
    "(real images flagged as AI) and false negatives (AI images flagged as "
    "real) are possible — treat this as one input among several, not a "
    "final verdict."
)

GRADCAM_CAPTION = (
    "This heatmap shows regions that influenced the model's prediction — "
    "it does not show regions proven to be AI-generated."
)


@dataclass
class ForensicReport:
    predicted_class: str  # "REAL" or "AI_GENERATED"
    raw_confidence: float  # max(softmax) before calibration
    calibrated_confidence: float  # max(softmax) after temperature scaling (== raw if no scaler loaded)
    probability_ai_generated: float  # calibrated P(AI_GENERATED), for callers who want the raw score
    branch_signal_norms: dict  # rough per-branch feature-magnitude diagnostic, NOT a calibrated sub-score
    metadata: dict
    model_version: str
    disclaimer: str = DISCLAIMER
    gradcam_caption: str = GRADCAM_CAPTION

    def to_dict(self) -> dict:
        d = {
            "predicted_class": self.predicted_class,
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "probability_ai_generated": self.probability_ai_generated,
            "branch_signal_norms": self.branch_signal_norms,
            "metadata": self.metadata,
            "model_version": self.model_version,
            "disclaimer": self.disclaimer,
            "gradcam_caption": self.gradcam_caption,
        }
        return d


class ForensicsPipeline:
    """
    Loads the model ONCE (brief §11: "load the model once, not per-request")
    and exposes a single `analyze()` entry point.
    """

    def __init__(self, config_name: str = "full_fusion", cfg: Optional[AppConfig] = None):
        self.cfg = cfg or load_config()
        self.config_name = config_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = self.cfg.paths.resolve("runs_dir") / f"ablation_{config_name}" / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No trained checkpoint at {checkpoint_path}. Train first with "
                f"src.training.train_ablation before starting the inference pipeline."
            )

        self.model = build_fusion_model(config_name, self.cfg.model).to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.model_version = f"{config_name}@{checkpoint_path.stat().st_mtime_ns}"

        # Temperature scaler: optional. If not fitted yet (Stage 6 note: run
        # a calibration-fitting script against the val set to produce this
        # file), fall back to an identity scaler (T=1, i.e. raw softmax)
        # rather than failing — calibration is a refinement, not a hard
        # dependency for the pipeline to function.
        self.temperature_scaler = TemperatureScaler().to(self.device)
        temp_path = self.cfg.paths.resolve("runs_dir") / f"ablation_{config_name}" / "temperature.json"
        if temp_path.exists():
            with open(temp_path) as f:
                t_value = json.load(f)["temperature"]
            with torch.no_grad():
                self.temperature_scaler.temperature.fill_(t_value)
        self.temperature_fitted = temp_path.exists()

        self.transform = build_transforms(self.cfg.data.image_size, train=False)
        self.gradcam = GradCAM(self.model) if self.model.use_rgb else None

    def _preprocess(self, pil_image: Image.Image) -> dict:
        rgb_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        resized = pil_image.resize((self.cfg.data.image_size, self.cfg.data.image_size))
        np_image = np.array(resized).astype(np.float32)

        fft_tensor = torch.from_numpy(fft_log_magnitude(to_grayscale(np_image))).unsqueeze(0).unsqueeze(0).to(self.device)
        residual_tensor = torch.from_numpy(compute_residual_features(np_image)).unsqueeze(0).to(self.device)

        return {"rgb": rgb_tensor, "fft": fft_tensor, "residual": residual_tensor}

    def analyze(self, pil_image: Image.Image, include_gradcam: bool = True) -> tuple[ForensicReport, Optional[np.ndarray]]:
        """
        Returns:
            (report, gradcam_overlay_or_None) — gradcam_overlay is an (H, W, 3)
            uint8 array suitable for direct display, or None if include_gradcam
            is False or the active config has no RGB branch.
        """
        pil_image = pil_image.convert("RGB")
        metadata = extract_metadata(pil_image)

        batch = self._preprocess(pil_image)

        with torch.no_grad():
            logits = self.model(batch)
            raw_probs = torch.softmax(logits, dim=1)
            raw_confidence, pred_idx = raw_probs.max(dim=1)

            calibrated_logits = self.temperature_scaler(logits)
            calibrated_probs = torch.softmax(calibrated_logits, dim=1)
            calibrated_confidence = calibrated_probs.max(dim=1).values

        pred_idx = int(pred_idx.item())
        predicted_class = LABEL_TO_CLASS[pred_idx]

        # Rough per-branch signal diagnostic: L2 norm of each active branch's
        # feature vector. This is NOT a calibrated sub-probability — it only
        # indicates how much each branch's representation "lit up" for this
        # image, useful for a UI breakdown but explicitly labeled as such.
        branch_norms = {}
        with torch.no_grad():
            if self.model.use_rgb:
                branch_norms["rgb"] = float(self.model.rgb_branch(batch["rgb"]).norm().item())
            if self.model.use_fft:
                branch_norms["fft"] = float(self.model.fft_branch(batch["fft"]).norm().item())
            if self.model.use_residual:
                branch_norms["residual"] = float(self.model.residual_branch(batch["residual"]).norm().item())

        report = ForensicReport(
            predicted_class=predicted_class,
            raw_confidence=float(raw_confidence.item()),
            calibrated_confidence=float(calibrated_confidence.item()),
            probability_ai_generated=float(calibrated_probs[0, 1].item()),
            branch_signal_norms=branch_norms,
            metadata=metadata.to_dict(),
            model_version=self.model_version,
        )
        if not self.temperature_fitted:
            report.metadata["calibration_note"] = (
                "No fitted temperature found — reporting raw (uncalibrated) softmax. "
                "Run the calibration-fitting step against the validation set first."
            )

        overlay = None
        if include_gradcam and self.gradcam is not None:
            heatmap = self.gradcam.generate({k: v.clone() for k, v in batch.items()}, target_class=pred_idx)
            original_resized = np.array(pil_image.resize((self.cfg.data.image_size, self.cfg.data.image_size)))
            overlay = overlay_heatmap(original_resized, heatmap)

        return report, overlay
