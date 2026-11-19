"""
Predictor — inference + Grad-CAM explainability.

Usage:
    predictor = Predictor(device="cuda")
    result = predictor.predict(image_path)
    # result.mining_score      : float 0–1
    # result.tags              : List[str]  (all tags above threshold)
    # result.all_probs         : Dict[str, float]
    # result.heatmap_pil       : PIL.Image (Grad-CAM overlay)
    # result.alert_level       : "CRITICAL" | "WARNING" | "CLEAR"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.core.model import MiningDetector, build_model, load_checkpoint
from app.utils.config import (
    ALL_TAGS,
    DEFAULT_THRESHOLD,
    MINING_ALERT_THRESHOLD,
    MINING_TAG_INDICES,
)
from app.core.dataset import get_predict_transforms
from app.core.metrics import decision_scores
from app.utils.logger import get_logger

log = get_logger("predictor")


# ── Result dataclass ────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    image_path:   Path
    mining_score: float
    alert_level:  str                      # "CRITICAL" / "WARNING" / "CLEAR"
    tags:         List[str] = field(default_factory=list)
    all_probs:    Dict[str, float] = field(default_factory=dict)
    heatmap_pil:  Optional[Image.Image] = None

    def to_dict(self) -> dict:
        return {
            "image":        str(self.image_path),
            "mining_score": round(self.mining_score, 4),
            "alert_level":  self.alert_level,
            "tags":         self.tags,
            "probabilities": {k: round(v, 4) for k, v in self.all_probs.items()},
        }


def _alert_level(score: float) -> str:
    if score >= 0.65:
        return "CRITICAL"
    if score >= MINING_ALERT_THRESHOLD:
        return "WARNING"
    return "CLEAR"


# ── Grad-CAM ─────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for EfficientNet-B0.
    Hooks into the last convolutional block to produce a class activation map.
    """

    def __init__(self, model: MiningDetector, target_class: int):
        self.model        = model
        self.target_class = target_class
        self._features: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Hook the last conv block of EfficientNet-B0
        target_layer = model.net.features[-1]
        self._fwd_hook = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, _, __, output):
        self._features = output.detach()

    def _save_gradients(self, _, __, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Returns normalised heatmap (H, W), float32 in [0, 1]."""
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        score  = logits[0, self.target_class]
        score.backward()

        # Global average pool gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._features).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam).squeeze().cpu().numpy()

        # Normalise
        cam = cam - cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.astype(np.float32)


def apply_heatmap_overlay(
    original_pil: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend Grad-CAM heatmap with the original image."""
    h, w = original_pil.size[1], original_pil.size[0]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_np = np.array(original_pil.convert("RGB"))
    blended = (1 - alpha) * orig_np + alpha * heatmap
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# ── Predictor class ──────────────────────────────────────────────────────
class Predictor:
    """
    Wraps model loading + inference + Grad-CAM generation.
    Call predict(path) for a single image or batch_predict(paths) for many.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device    = device
        self.threshold = threshold
        self.transform = get_predict_transforms()

        self.model = build_model(device, pretrained=False)
        ckpt = load_checkpoint(self.model, checkpoint_path, device)
        self.transform = get_predict_transforms(ckpt.get('image_size', 224))
        self.tag_thresholds = np.asarray(ckpt.get('thresholds', [threshold] * len(ALL_TAGS)))
        if self.tag_thresholds.shape != (len(ALL_TAGS),):
            raise ValueError('Checkpoint must contain one decision threshold per tag.')
        self.use_threshold_scores = 'thresholds' in ckpt
        decision_scores(np.zeros(len(ALL_TAGS)), self.tag_thresholds)
        self.model.eval()
        log.info(f"Model loaded from epoch {ckpt.get('epoch', '?')} | "
                 f"Val F2: {ckpt.get('metrics', {}).get('val_f2', 'N/A')}")

    # ------------------------------------------------------------------
    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as source:
            img = source.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor, img

    # ------------------------------------------------------------------
    def predict(
        self,
        image_path: Path | str,
        generate_cam: bool = True,
    ) -> PredictionResult:
        image_path = Path(image_path)
        tensor, orig_pil = self._load_image(image_path)

        # ── Forward pass ─────────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(tensor)           # (1, 17)
            probs  = torch.sigmoid(logits)[0]     # (17,)

        probs_np = probs.cpu().numpy()
        tag_scores = decision_scores(probs_np, self.tag_thresholds) if self.use_threshold_scores else probs_np
        mining_score = float(tag_scores[MINING_TAG_INDICES].max())
        tags = [ALL_TAGS[i] for i, p in enumerate(probs_np) if p >= self.tag_thresholds[i]]
        all_probs = {ALL_TAGS[i]: float(probs_np[i]) for i in range(len(ALL_TAGS))}

        # ── Grad-CAM ─────────────────────────────────────────────────
        heatmap_pil = None
        if generate_cam:
            # Use the highest-scoring mining tag as CAM target
            mining_idx = int(np.argmax(tag_scores[MINING_TAG_INDICES]))
            target_class = MINING_TAG_INDICES[mining_idx]
            cam_gen = None
            try:
                cam_gen = GradCAM(self.model, target_class)
                cam = cam_gen.generate(tensor)
                heatmap_pil = apply_heatmap_overlay(orig_pil, cam)
            except Exception as e:
                log.warning(f"Grad-CAM failed: {e}")
            finally:
                if cam_gen is not None:
                    cam_gen.remove_hooks()

        return PredictionResult(
            image_path   = image_path,
            mining_score = mining_score,
            alert_level  = _alert_level(mining_score),
            tags         = tags,
            all_probs    = all_probs,
            heatmap_pil  = heatmap_pil,
        )

    # ------------------------------------------------------------------
    def batch_predict(
        self,
        paths: List[Path | str],
        generate_cam: bool = False,   # skip CAM for speed in batch mode
        progress_fn=None,
    ) -> List[PredictionResult]:
        results = []
        for i, p in enumerate(paths):
            try:
                r = self.predict(p, generate_cam=generate_cam)
                results.append(r)
            except Exception as e:
                log.error(f"Prediction failed for {p}: {e}")
            if progress_fn:
                progress_fn(i + 1, len(paths))
        return results
