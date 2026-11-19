"""
Trainer — full training loop with:
  - BCEWithLogitsLoss
  - AdamW + CosineAnnealingLR
  - Mining average precision, recall and F2, plus macro tag F2
  - Early stopping
  - Qt-signal-based progress reporting (no print statements in model code)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from app.core.model import MiningDetector, save_checkpoint
from app.core.metrics import mining_metrics
from app.utils.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHT_DECAY,
    LOGS_DIR,
    MINING_TAG_INDICES,
    MODELS_DIR,
    NUM_CLASSES,
)
from app.utils.logger import get_logger

log = get_logger("trainer")


# ── Metric helpers ──────────────────────────────────────────────────────
def fbeta_score(y_true: np.ndarray, y_pred: np.ndarray,
                beta: float = 2.0, threshold: float = DEFAULT_THRESHOLD) -> float:
    """Macro-averaged Fβ over classes; this is not the sample-averaged Kaggle F2."""
    # Boolean masks are essential here. Bitwise ``~`` on integer arrays turns
    # 0/1 into -1/-2, which produces negative FP/FN counts and invalid scores.
    true_bin = np.asarray(y_true, dtype=bool)
    pred_bin = np.asarray(y_pred) >= threshold
    tp = (pred_bin & true_bin).sum(axis=0)
    fp = (pred_bin & ~true_bin).sum(axis=0)
    fn = (~pred_bin & true_bin).sum(axis=0)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)

    beta2 = beta ** 2
    f_beta = ((1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8))
    return float(f_beta.mean())


def mining_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                    threshold: float = 0.4) -> float:
    """Binary accuracy for mining-related tags only."""
    mining_true = y_true[:, MINING_TAG_INDICES].max(axis=1)
    mining_pred_prob = y_pred[:, MINING_TAG_INDICES].max(axis=1)
    mining_pred = (mining_pred_prob >= threshold).astype(np.int32)
    return float((mining_pred == mining_true).mean())


# ── Trainer class ────────────────────────────────────────────────────────
class Trainer:
    """
    Orchestrates training. Designed for GUI integration:
    - All logging goes through the `log_fn` callback (string → None).
    - Epoch-level metrics emitted via `epoch_fn` callback (dict → None).
    - Supports graceful cancellation via the `.stop()` method.
    """

    def __init__(
        self,
        model: MiningDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        epochs: int = DEFAULT_EPOCHS,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        patience: int = DEFAULT_PATIENCE,
        log_fn: Optional[Callable[[str], None]] = None,
        epoch_fn: Optional[Callable[[Dict], None]] = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs
        self.patience     = patience
        self.log_fn       = log_fn or (lambda msg: log.info(msg))
        self.epoch_fn     = epoch_fn or (lambda d: None)

        self._stop_flag = False
        self.history: List[Dict] = []

        # ── Optimizer & Scheduler ──────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler    = GradScaler(enabled=(device == "cuda"))

    # ------------------------------------------------------------------
    def stop(self):
        """Signal the training loop to stop gracefully after the current batch."""
        self._stop_flag = True

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict:
        self.model.train(training)
        total_loss = 0.0
        processed_batches = 0
        processed_samples = 0
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch_idx, (imgs, labels) in enumerate(loader):
                if self._stop_flag:
                    break
                imgs   = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(enabled=(self.device == "cuda")):
                    logits = self.model(imgs)
                    loss   = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * len(imgs)
                processed_samples += len(imgs)
                processed_batches += 1
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        if not all_probs:
            raise ValueError("No batches evaluated; refusing to report fabricated zero metrics.")

        all_probs  = np.concatenate(all_probs,  axis=0).astype(np.float32)
        all_labels = np.concatenate(all_labels, axis=0).astype(np.int32)

        return {
            "loss":           total_loss / processed_samples,
            "f2":             fbeta_score(all_labels, all_probs),
            "mining_acc":     mining_accuracy(all_labels, all_probs),
            "mining":         mining_metrics(all_labels, all_probs),
        }

    # ------------------------------------------------------------------
    def fit(self) -> List[Dict]:
        """
        Run the full training loop.  Returns history list.
        Each element: {"epoch", "train_loss", "val_loss", "train_f2",
                       "val_f2", "train_mining_acc", "val_mining_acc"}
        """
        best_val_f2   = -1.0
        best_mining_ap = -1.0
        patience_left = self.patience
        history_path  = LOGS_DIR / "training_history.json"

        self.log_fn(f"▶ Training on device: {self.device}  |  "
                    f"Epochs: {self.epochs}  |  "
                    f"Train batches: {len(self.train_loader)}")

        for epoch in range(1, self.epochs + 1):
            if self._stop_flag:
                self.log_fn("⏹ Training stopped by user.")
                break

            t0 = time.time()
            train_metrics = self._run_epoch(self.train_loader, training=True)
            if self._stop_flag:
                self.log_fn("Training cancelled; incomplete epoch will not be saved.")
                break
            val_metrics   = self._run_epoch(self.val_loader,   training=False)
            if self._stop_flag:
                self.log_fn("Validation cancelled; incomplete epoch will not be saved.")
                break
            mining_ap = val_metrics['mining']['average_precision']
            if mining_ap is None:
                raise ValueError("Validation has no mining positives; cannot select a mining model.")
            self.scheduler.step()

            elapsed = time.time() - t0
            row = {
                "epoch":            epoch,
                "train_loss":       round(train_metrics["loss"],       4),
                "val_loss":         round(val_metrics["loss"],         4),
                "train_f2":         round(train_metrics["f2"],         4),
                "val_f2":           round(val_metrics["f2"],           4),
                "train_mining_acc": round(train_metrics["mining_acc"], 4),
                "val_mining_acc":   round(val_metrics["mining_acc"],   4),
                "lr":               round(self.scheduler.get_last_lr()[0], 6),
                "elapsed_s":        round(elapsed, 1),
                "val_mining":       val_metrics["mining"],
            }
            self.history.append(row)

            self.log_fn(
                f"Epoch {epoch:03d}/{self.epochs:03d}  |  "
                f"Train Loss: {row['train_loss']:.4f}  Val Loss: {row['val_loss']:.4f}  |  "
                f"Val F2: {row['val_f2']:.4f}  Mining Acc: {row['val_mining_acc']:.3f}  |  "
                f"LR: {row['lr']:.2e}  [{elapsed:.1f}s]"
            )
            self.epoch_fn(row)

            # ── Early stopping / checkpointing ─────────────────────────
            if mining_ap > best_mining_ap:
                best_mining_ap = mining_ap
                best_val_f2   = val_metrics["f2"]
                patience_left = self.patience
                ckpt_path = save_checkpoint(
                    self.model, self.optimizer, epoch,
                    {"val_f2": best_val_f2, "val_mining": val_metrics["mining"],
                     "train_image_names": self.train_loader.dataset.df.image_name.tolist(),
                     "val_image_names": self.val_loader.dataset.df.image_name.tolist(),
                     "f2_averaging": "macro_classes", "independent_test_passed": False},
                )
                self.log_fn(f"  ✅ Best model saved  (mining AP={best_mining_ap:.4f}, val macro F2={best_val_f2:.4f}) → {ckpt_path.name}")
            else:
                patience_left -= 1
                self.log_fn(f"  ⚠ No improvement. Patience left: {patience_left}/{self.patience}")
                if patience_left == 0:
                    self.log_fn("⏹ Early stopping triggered.")
                    break

        # Persist history
        if self.history:
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)
            self.log_fn(f"📊 Training history saved → {history_path}")
        return self.history
