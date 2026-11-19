"""
Train tab — hyperparameter controls, real-time loss/F2 chart, log console.
Runs training in a QThread and emits epoch metrics via signal for live plotting.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from app.core.dataset import build_dataloaders
from app.core.model import build_model
from app.core.trainer import Trainer
from app.utils.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_VAL_SPLIT,
    DEFAULT_WEIGHT_DECAY,
)
from app.utils.logger import get_logger

log = get_logger("train_tab")


# ── Training worker thread ───────────────────────────────────────────────
class TrainWorker(QThread):
    log_signal   = pyqtSignal(str)
    epoch_signal = pyqtSignal(dict)
    done_signal  = pyqtSignal(list)   # history list
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        epochs: int, batch_size: int, lr: float,
        weight_decay: float, val_split: float, patience: int,
    ):
        super().__init__()
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.weight_decay = weight_decay
        self.val_split    = val_split
        self.patience     = patience
        self._trainer: Trainer | None = None

    def run(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_signal.emit(f"⚙ Building dataloaders (batch={self.batch_size}) …")
            train_loader, val_loader, _ = build_dataloaders(
                val_split=self.val_split, batch_size=self.batch_size,
                log_fn=self.log_signal.emit,
            )
            self.log_signal.emit(
                f"   Train samples: {len(train_loader.dataset):,}  "
                f"Val samples: {len(val_loader.dataset):,}"
            )

            self.log_signal.emit("🧠 Building model (EfficientNet-B0) …")
            model = build_model(device)

            self._trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=self.epochs,
                lr=self.lr,
                weight_decay=self.weight_decay,
                patience=self.patience,
                log_fn=self.log_signal.emit,
                epoch_fn=self.epoch_signal.emit,
            )
            history = self._trainer.fit()
            self.done_signal.emit(history)
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        if self._trainer:
            self._trainer.stop()


# ── Live chart widget ────────────────────────────────────────────────────
class LiveChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), facecolor="#0D1117")
        super().__init__(self.fig)
        self.setParent(parent)
        self._ax_loss = self.fig.add_subplot(211)
        self._ax_f2   = self.fig.add_subplot(212)
        self._train_loss: List[float] = []
        self._val_loss:   List[float] = []
        self._train_f2:   List[float] = []
        self._val_f2:     List[float] = []
        self._style_axes()

    def _style_axes(self):
        for ax in [self._ax_loss, self._ax_f2]:
            ax.set_facecolor("#0D1117")
            ax.tick_params(colors="#C9D1D9", labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#30363D")
        self._ax_loss.set_title("Training & Validation Loss",
                                 color="#4FC3F7", fontsize=9, pad=4)
        self._ax_f2.set_title("Validation F2-Score (Kaggle metric)",
                               color="#4FC3F7", fontsize=9, pad=4)
        self._ax_loss.set_ylabel("BCE Loss", color="#C9D1D9", fontsize=8)
        self._ax_f2.set_ylabel("F2", color="#C9D1D9", fontsize=8)
        self._ax_f2.set_xlabel("Epoch", color="#C9D1D9", fontsize=8)
        self.fig.tight_layout(pad=1.5)

    def update_data(self, row: Dict):
        self._train_loss.append(row["train_loss"])
        self._val_loss.append(row["val_loss"])
        self._train_f2.append(row["train_f2"])
        self._val_f2.append(row["val_f2"])
        epochs = list(range(1, len(self._train_loss) + 1))

        self._ax_loss.clear()
        self._ax_f2.clear()
        self._style_axes()

        self._ax_loss.plot(epochs, self._train_loss, "o-", color="#4FC3F7",
                            linewidth=1.5, markersize=4, label="Train")
        self._ax_loss.plot(epochs, self._val_loss, "s--", color="#EF5350",
                            linewidth=1.5, markersize=4, label="Val")
        self._ax_loss.legend(fontsize=8, facecolor="#0D1117",
                              labelcolor="#C9D1D9", framealpha=0.8)

        self._ax_f2.plot(epochs, self._val_f2, "s-", color="#66BB6A",
                          linewidth=2, markersize=5, label="Val F2")
        if self._val_f2:
            best_ep = self._val_f2.index(max(self._val_f2)) + 1
            self._ax_f2.axvline(best_ep, color="#FFB300", linewidth=1,
                                 linestyle=":", alpha=0.7, label=f"Best epoch {best_ep}")
        self._ax_f2.legend(fontsize=8, facecolor="#0D1117",
                            labelcolor="#C9D1D9", framealpha=0.8)

        self.fig.tight_layout(pad=1.5)
        self.draw()

    def reset(self):
        self._train_loss.clear(); self._val_loss.clear()
        self._train_f2.clear();   self._val_f2.clear()
        self._ax_loss.clear(); self._ax_f2.clear()
        self._style_axes(); self.draw()


# ── Train tab widget ─────────────────────────────────────────────────────
class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: TrainWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        splitter = QSplitter()
        splitter.setHandleWidth(6)

        # ── Left panel — controls ──────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        hp_box = QGroupBox("⚙  Hyperparameters")
        form = QFormLayout(hp_box)
        form.setSpacing(8)

        self._spin_epochs = QSpinBox()
        self._spin_epochs.setRange(1, 200); self._spin_epochs.setValue(DEFAULT_EPOCHS)
        form.addRow("Epochs", self._spin_epochs)

        self._spin_batch = QSpinBox()
        self._spin_batch.setRange(4, 256); self._spin_batch.setValue(DEFAULT_BATCH_SIZE)
        self._spin_batch.setSingleStep(8)
        form.addRow("Batch Size", self._spin_batch)

        self._spin_lr = QDoubleSpinBox()
        self._spin_lr.setRange(1e-6, 0.1)
        self._spin_lr.setDecimals(6)
        self._spin_lr.setValue(DEFAULT_LR)
        self._spin_lr.setSingleStep(1e-4)
        form.addRow("Learning Rate", self._spin_lr)

        self._spin_wd = QDoubleSpinBox()
        self._spin_wd.setRange(0, 0.1)
        self._spin_wd.setDecimals(5)
        self._spin_wd.setValue(DEFAULT_WEIGHT_DECAY)
        form.addRow("Weight Decay", self._spin_wd)

        self._spin_val = QDoubleSpinBox()
        self._spin_val.setRange(0.05, 0.4)
        self._spin_val.setDecimals(2)
        self._spin_val.setValue(DEFAULT_VAL_SPLIT)
        self._spin_val.setSingleStep(0.05)
        form.addRow("Val Split", self._spin_val)

        self._spin_patience = QSpinBox()
        self._spin_patience.setRange(1, 20); self._spin_patience.setValue(DEFAULT_PATIENCE)
        form.addRow("Early Stop Patience", self._spin_patience)

        left_layout.addWidget(hp_box)

        # Status
        status_box = QGroupBox("📊  Current Epoch")
        status_form = QFormLayout(status_box)
        self._lbl_epoch       = QLabel("—")
        self._lbl_train_loss  = QLabel("—")
        self._lbl_val_loss    = QLabel("—")
        self._lbl_val_f2      = QLabel("—")
        self._lbl_mining_acc  = QLabel("—")
        self._lbl_lr_cur      = QLabel("—")
        for row_label, w in [
            ("Epoch",        self._lbl_epoch),
            ("Train Loss",   self._lbl_train_loss),
            ("Val Loss",     self._lbl_val_loss),
            ("Val F2",       self._lbl_val_f2),
            ("Mining Acc",   self._lbl_mining_acc),
            ("Current LR",   self._lbl_lr_cur),
        ]:
            status_form.addRow(row_label, w)
        left_layout.addWidget(status_box)
        left_layout.addStretch()

        # Buttons
        self._btn_start = QPushButton("▶  Start Training")
        self._btn_start.setFixedHeight(40)
        self._btn_start.setStyleSheet(
            "QPushButton{background:#2E7D32;color:white;font-weight:bold;"
            "border-radius:6px;}QPushButton:hover{background:#388E3C;}"
            "QPushButton:disabled{background:#37474F;color:#78909C;}"
        )
        self._btn_start.clicked.connect(self._start_training)

        self._btn_stop = QPushButton("⏹  Stop")
        self._btn_stop.setFixedHeight(40)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(
            "QPushButton{background:#B71C1C;color:white;font-weight:bold;"
            "border-radius:6px;}QPushButton:hover{background:#C62828;}"
            "QPushButton:disabled{background:#37474F;color:#78909C;}"
        )
        self._btn_stop.clicked.connect(self._stop_training)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # ── Right panel — chart + log ──────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._chart = LiveChart()
        right_layout.addWidget(self._chart, stretch=3)

        log_box = QGroupBox("📋  Training Log")
        log_layout = QVBoxLayout(log_box)
        self._console = QPlainTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumBlockCount(2000)
        self._console.setStyleSheet(
            "QPlainTextEdit{background:#0D1117;color:#C9D1D9;"
            "font-family:Consolas,monospace;font-size:11px;border-radius:6px;}"
        )
        log_layout.addWidget(self._console)
        right_layout.addWidget(log_box, stretch=2)
        splitter.addWidget(right)
        splitter.setSizes([280, 720])

        root.addWidget(splitter)

    # ------------------------------------------------------------------
    def _start_training(self):
        from app.utils.config import TRAIN_CSV
        if not TRAIN_CSV.exists():
            self._console.appendPlainText(
                "❌ Dataset not found. Please download it in the Setup tab first."
            )
            return

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._chart.reset()
        self._console.clear()

        self._worker = TrainWorker(
            epochs       = self._spin_epochs.value(),
            batch_size   = self._spin_batch.value(),
            lr           = self._spin_lr.value(),
            weight_decay = self._spin_wd.value(),
            val_split    = self._spin_val.value(),
            patience     = self._spin_patience.value(),
        )
        self._worker.log_signal.connect(self._console.appendPlainText)
        self._worker.epoch_signal.connect(self._on_epoch)
        self._worker.done_signal.connect(self._training_done)
        self._worker.error_signal.connect(self._training_error)
        self._worker.start()

    # ------------------------------------------------------------------
    def _on_epoch(self, row: Dict):
        self._chart.update_data(row)
        self._lbl_epoch.setText(str(row["epoch"]))
        self._lbl_train_loss.setText(f"{row['train_loss']:.4f}")
        self._lbl_val_loss.setText(f"{row['val_loss']:.4f}")
        self._lbl_val_f2.setText(f"{row['val_f2']:.4f}")
        self._lbl_mining_acc.setText(f"{row['val_mining_acc']*100:.1f}%")
        self._lbl_lr_cur.setText(f"{row['lr']:.2e}")

    # ------------------------------------------------------------------
    def _stop_training(self):
        if self._worker:
            self._worker.stop()
        self._btn_stop.setEnabled(False)

    # ------------------------------------------------------------------
    def _training_done(self, history):
        self._reset_buttons()
        self._console.appendPlainText(
            f"\n🎉 Training complete! {len(history)} epochs finished. "
            "Check Dashboard tab for detailed metrics."
        )

    # ------------------------------------------------------------------
    def _training_error(self, msg: str):
        self._reset_buttons()
        self._console.appendPlainText(f"\n❌ Training error: {msg}")

    # ------------------------------------------------------------------
    def _reset_buttons(self):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
