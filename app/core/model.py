"""
Model — EfficientNet-B0 with a multi-label classification head.

Architecture:
  EfficientNet-B0 (pretrained ImageNet)
      └─ AdaptiveAvgPool2d (1280-dim feature vector)
      └─ Dropout(0.3)
      └─ Linear(1280, NUM_CLASSES)   ← raw logits
         (sigmoid applied during inference only)

Training loss: BCEWithLogitsLoss (numerically stable, no need for
separate sigmoid during training).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from app.utils.config import ALL_TAGS, CHECKPOINT_NAME, MODELS_DIR, NUM_CLASSES


# ── Model definition ────────────────────────────────────────────────────
class MiningDetector(nn.Module):
    """
    EfficientNet-B0 backbone with a multi-label head.

    Call .get_mining_score(logits) to obtain the mining probability [0,1].
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        in_features = base.classifier[1].in_features  # 1280

        # Replace classifier head
        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.net = base

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (B, NUM_CLASSES)."""
        return self.net(x)

    # ------------------------------------------------------------------
    @staticmethod
    def get_mining_score(logits: torch.Tensor) -> torch.Tensor:
        """
        Given raw logits (B, NUM_CLASSES), return mining probability (B,).
        Mining score = max(sigmoid(artisanal_mine), sigmoid(conventional_mine)).
        Tag indices 1 and 7 correspond to artisanal_mine / conventional_mine.
        """
        from app.utils.config import MINING_TAG_INDICES
        probs = torch.sigmoid(logits)
        mining_probs = probs[:, MINING_TAG_INDICES]   # (B, 2)
        return mining_probs.max(dim=1).values           # (B,)


# ── Save / Load helpers ─────────────────────────────────────────────────
def save_checkpoint(
    model: MiningDetector,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: Optional[Path] = None,
) -> Path:
    path = path or (MODELS_DIR / CHECKPOINT_NAME)
    torch.save({
        "tags":       ALL_TAGS,
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
    }, path)
    return path


def load_checkpoint(
    model: MiningDetector,
    path: Optional[Path] = None,
    device: str = "cpu",
) -> Dict:
    path = path or (MODELS_DIR / CHECKPOINT_NAME)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    from app.utils.config import ALL_TAGS
    if 'tags' in ckpt and ckpt['tags'] != ALL_TAGS:
        raise ValueError("Checkpoint label order does not match application labels.")
    model.load_state_dict(ckpt["state_dict"])
    return ckpt


def build_model(device: str = "cpu", pretrained: bool = True) -> MiningDetector:
    model = MiningDetector(pretrained=pretrained)
    model.to(device)
    return model
