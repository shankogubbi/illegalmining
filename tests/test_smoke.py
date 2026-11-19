"""
Smoke tests — verify the model, dataset logic, and predictor
can be imported and produce correct-shaped outputs.

Run with:
    python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest
from PIL import Image

from app.utils.config import ALL_TAGS, NUM_CLASSES, MINING_TAG_INDICES
from app.core.model import MiningDetector, build_model
from app.core.trainer import fbeta_score, mining_accuracy
from app.core.dataset import get_val_transforms
from app.core.dataset import build_dataloaders


# ── Model tests ──────────────────────────────────────────────────────────
class TestModel:
    def test_forward_shape(self):
        model = build_model("cpu", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES), \
            f"Expected (2, {NUM_CLASSES}), got {out.shape}"

    def test_mining_score_range(self):
        model = build_model("cpu", pretrained=False)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        scores = MiningDetector.get_mining_score(logits)
        assert scores.shape == (4,)
        assert (scores >= 0).all() and (scores <= 1).all(), \
            "Mining scores must be in [0, 1]"

    def test_num_classes(self):
        assert NUM_CLASSES == 17

    def test_mining_indices(self):
        assert ALL_TAGS[MINING_TAG_INDICES[0]] == "artisinal_mine"
        assert ALL_TAGS[MINING_TAG_INDICES[1]] == "conventional_mine"


# ── Metric tests ─────────────────────────────────────────────────────────
class TestMetrics:
    def test_fbeta_perfect(self):
        y = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        score = fbeta_score(y, y.astype(np.float32), beta=2)
        assert abs(score - 1.0) < 1e-5, f"Perfect prediction should give F2=1.0, got {score}"

    def test_fbeta_zero(self):
        y_true = np.array([[1, 0], [0, 1]], dtype=np.int32)
        y_pred = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        score = fbeta_score(y_true, y_pred, threshold=0.5)
        assert score < 0.1, f"Inverse predictions should give near-0 F2, got {score}"

    def test_mining_accuracy_perfect(self):
        # 2 samples, 17 classes — set mining indices to match perfectly
        y_true = np.zeros((2, NUM_CLASSES), dtype=np.int32)
        y_pred = np.zeros((2, NUM_CLASSES), dtype=np.float32)
        y_true[0, MINING_TAG_INDICES[0]] = 1
        y_pred[0, MINING_TAG_INDICES[0]] = 0.9
        acc = mining_accuracy(y_true, y_pred, threshold=0.4)
        assert acc == 1.0, f"Expected perfect mining acc, got {acc}"


# ── Transform tests ───────────────────────────────────────────────────────
class TestTransforms:
    def test_val_transform_output_shape(self):
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        transform = get_val_transforms(224)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224), \
            f"Expected (3, 224, 224), got {tensor.shape}"
        assert tensor.dtype == torch.float32

    def test_normalisation_range(self):
        img = Image.new("RGB", (256, 256), color=(127, 127, 127))
        transform = get_val_transforms(224)
        tensor = transform(img)
        # After ImageNet normalisation, values should be around 0
        assert tensor.abs().max() < 5.0, "Normalised tensor looks wrong"


def test_dataloaders_exclude_unreadable_images(tmp_path):
    import pandas as pd

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        Image.new("RGB", (32, 32), color=(100, 150, 200)).save(img_dir / f"train_{i}.jpg")
    (img_dir / "train_5.jpg").write_bytes(b"\0" * 64)
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({
        "image_name": [f"train_{i}" for i in range(6)],
        "tags": ["primary"] * 6,
    }).to_csv(csv_path, index=False)
    messages = []
    train, val, df = build_dataloaders(
        csv_path=csv_path, img_dir=img_dir, batch_size=2,
        val_split=0.2, log_fn=messages.append,
    )
    assert len(df) == 5
    assert len(train.dataset) + len(val.dataset) == 5
    assert any("train_5.jpg" in message for message in messages)
    for loader in (train, val):
        for images, labels in loader:
            assert images.shape[1:] == (3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
