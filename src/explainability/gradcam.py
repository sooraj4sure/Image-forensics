"""
Grad-CAM (brief §10): "heatmap over the RGB branch, shown alongside the
original image and the prediction." Implemented manually via forward/
backward hooks rather than a library like pytorch-grad-cam, because those
libraries generally assume a single-tensor model input — FusionModel takes
a dict of {rgb, fft, residual} tensors, and the heatmap is only meaningful
over the RGB branch's spatial feature map (layer4 of the ResNet-18 trunk,
the last conv layer before global pooling).

UI/report copy using this MUST say the heatmap shows "regions that
influenced the model's prediction", not regions proven to be AI-generated
(brief §10) — that phrasing lives in api/ and app/, not here, but is noted
here too since it's easy to lose track of when wiring the heatmap into UI.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Args:
        model: a FusionModel instance with use_rgb=True (raises otherwise —
            there is no RGB branch to compute a Grad-CAM over without it)
        target_layer: the conv layer to hook. Defaults to
            model.rgb_branch.trunk[-2], which is ResNet-18's layer4 (the
            trunk's children are [..., layer4, avgpool] after fc removal
            in RGBBranch, so index -2 is layer4).
    """

    def __init__(self, model, target_layer=None):
        if not getattr(model, "use_rgb", False):
            raise ValueError("GradCAM requires a model with an active RGB branch.")
        self.model = model
        self.target_layer = target_layer or model.rgb_branch.trunk[-2]

        self._activations = None
        self._gradients = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, batch: dict, target_class: int | None = None) -> np.ndarray:
        """
        Args:
            batch: dict as consumed by FusionModel.forward — must include
                "rgb" and whatever other keys the model's active branches need.
                Batch size must be 1 (one heatmap per call).
            target_class: class index to explain. Defaults to the model's
                own predicted class (argmax of logits) — i.e. "why did the
                model predict what it predicted", not a fixed class.

        Returns:
            2D float32 array, shape (H, W) matching the RGB input's spatial
            size, values normalized to [0, 1]. 0 = no influence, 1 = most
            influential region for the target class.
        """
        self.model.eval()
        batch["rgb"].requires_grad_(True)

        logits = self.model(batch)
        if logits.shape[0] != 1:
            raise ValueError(f"GradCAM.generate expects batch size 1, got {logits.shape[0]}")

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        activations = self._activations[0]  # (C, h, w)
        gradients = self._gradients[0]  # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # (C,) — global-average-pooled gradients
        cam = torch.einsum("c,chw->hw", weights, activations)
        cam = F.relu(cam)  # only positive influence, standard Grad-CAM convention

        # Resize to match the RGB input's spatial size
        target_h, target_w = batch["rgb"].shape[-2], batch["rgb"].shape[-1]
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        cam = F.interpolate(cam, size=(target_h, target_w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            return np.zeros_like(cam, dtype=np.float32)
        return ((cam - cam_min) / (cam_max - cam_min)).astype(np.float32)


def overlay_heatmap(rgb_image_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Args:
        rgb_image_uint8: (H, W, 3) uint8 original image
        heatmap: (H, W) float array in [0, 1], from GradCAM.generate
        alpha: heatmap blend strength

    Returns:
        (H, W, 3) uint8 image with a red-scale heatmap overlaid.
    """
    import cv2

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = (1 - alpha) * rgb_image_uint8.astype(np.float32) + alpha * heatmap_color.astype(np.float32)
    return np.clip(overlaid, 0, 255).astype(np.uint8)
