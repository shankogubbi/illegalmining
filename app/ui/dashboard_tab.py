"""
Dashboard tab — training history charts, per-class F2 bar chart,
confusion matrix for mining vs. non-mining.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from app.utils.config import ALL_TAGS, LOGS_DIR, RESULTS_DIR, MODELS_DIR, CHECKPOINT_NAME
from app.utils.logger import get_logger

log = get_logger("dashboard_tab")
HISTORY_PATH = LOGS_DIR / "training_history.json"


# ── Helper: styled matplotlib figure ────────────────────────────────────
def _make_fig(rows: int = 1, cols: int = 1,
              figsize=(8, 3.5)) -> tuple[Figure, list]:
    fig = Figure(figsize=figsize, facecolor="#0D1117")
    axes = []
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_facecolor("#0D1117")
        ax.tick_params(colors="#C9D1D9", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363D")
        axes.append(ax)
    fig.tight_layout(pad=1.5)
    return fig, axes


# ── Dashboard tab widget ─────────────────────────────────────────────────
class DashboardTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: List[Dict] = []
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Refresh button
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("🔄  Refresh Charts")
        btn_refresh.setFixedHeight(32)
        btn_refresh.clicked.connect(self._load_and_render)
        self._lbl_status = QLabel("No training history found yet.")
        self._lbl_status.setStyleSheet("color:#78909C; font-size:12px;")
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(self._lbl_status)
        btn_row.addStretch()
        root.addLayout(btn_row)
        self._evaluation_summary = QLabel('Model evaluation will appear after the experiment completes.')
        self._evaluation_summary.setWordWrap(True)
        self._evaluation_summary.setStyleSheet('color:#C9D1D9; padding:8px; background:#1A2730;')
        root.addWidget(self._evaluation_summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        self._chart_layout = QVBoxLayout(container)
        self._chart_layout.setSpacing(12)

        # ── Chart 1: Loss curves ──────────────────────────────────
        loss_box = QGroupBox("📉  Loss History")
        loss_inner = QVBoxLayout(loss_box)
        self._fig_loss, self._ax_loss = _make_fig(1, 1, figsize=(8, 3))
        self._ax_loss = self._ax_loss[0]
        self._canvas_loss = FigureCanvas(self._fig_loss)
        self._canvas_loss.setFixedHeight(230)
        loss_inner.addWidget(self._canvas_loss)
        self._chart_layout.addWidget(loss_box)

        # ── Chart 2: F2 + Mining Acc ──────────────────────────────
        f2_box = QGroupBox("📈  F2-Score & Mining Accuracy")
        f2_inner = QVBoxLayout(f2_box)
        self._fig_f2, axs = _make_fig(1, 2, figsize=(9, 3))
        self._ax_f2, self._ax_macc = axs
        self._canvas_f2 = FigureCanvas(self._fig_f2)
        self._canvas_f2.setFixedHeight(230)
        f2_inner.addWidget(self._canvas_f2)
        self._chart_layout.addWidget(f2_box)

        # ── Chart 3: LR schedule ─────────────────────────────────
        lr_box = QGroupBox("📡  Learning Rate Schedule")
        lr_inner = QVBoxLayout(lr_box)
        self._fig_lr, self._ax_lr_c = _make_fig(1, 1, figsize=(8, 2.5))
        self._ax_lr_c = self._ax_lr_c[0]
        self._canvas_lr = FigureCanvas(self._fig_lr)
        self._canvas_lr.setFixedHeight(180)
        lr_inner.addWidget(self._canvas_lr)
        self._chart_layout.addWidget(lr_box)

        # ── Summary stats ─────────────────────────────────────────
        stats_box = QGroupBox("📊  Training Summary")
        stats_inner = QHBoxLayout(stats_box)
        self._stat_labels: Dict[str, QLabel] = {}
        for stat in ["Total Epochs", "Best Val F2", "Best Mining Acc",
                      "Best Epoch", "Min Val Loss", "Final LR"]:
            col = QVBoxLayout()
            title = QLabel(stat)
            title.setStyleSheet("color:#78909C; font-size:10px;")
            val = QLabel("—")
            val.setStyleSheet("color:#4FC3F7; font-size:18px; font-weight:bold;")
            col.addWidget(title)
            col.addWidget(val)
            stats_inner.addLayout(col)
            self._stat_labels[stat] = val
        self._chart_layout.addWidget(stats_box)

        scroll.setWidget(container)
        root.addWidget(scroll)

        # Auto-load if history exists
        self._load_and_render()

    # ------------------------------------------------------------------
    def _load_and_render(self):
        evaluation = RESULTS_DIR / 'transfer_v1' / 'final_metrics.json'
        active = MODELS_DIR / CHECKPOINT_NAME
        if evaluation.exists():
            data = json.loads(evaluation.read_text(encoding='utf-8'))
            expected_hash = data.get('active_checkpoint_sha256')
            if active.exists() and expected_hash and hashlib.sha256(active.read_bytes()).hexdigest() == expected_hash:
                m = data['test']['mining']
                subtype_text = ' | '.join(
                    f"{tag.replace('_', ' ')} recall: {v['tp']/v['positives']:.1%}"
                    for tag, v in data['test']['subtype'].items() if v['positives'])
                self._evaluation_summary.setText(
                    f"Active model — held-out image evaluation: precision {m['precision']:.1%}, "
                    f"recall {m['recall']:.1%}, F2 {m['f2']:.3f}. "
                    f"Detected {m['tp']}/{m['positives']} mining images; {m['fp']} false alerts.\n{subtype_text}")
            else:
                self._evaluation_summary.setText('Saved experiment results are available in results/transfer_v1/REPORT.md.')
        if not HISTORY_PATH.exists():
            self._lbl_status.setText("⚠ No training history found. Train a model first.")
            return

        with open(HISTORY_PATH, "r") as f:
            self._history = json.load(f)

        if not self._history:
            self._lbl_status.setText("⚠ History file is empty.")
            return

        self._lbl_status.setText(
            f"✅ Loaded {len(self._history)} epochs from {HISTORY_PATH.name}"
        )
        self._render_loss()
        self._render_f2()
        self._render_lr()
        self._render_stats()

    # ------------------------------------------------------------------
    def _render_loss(self):
        ax = self._ax_loss
        ax.clear()
        ax.set_facecolor("#0D1117")
        ax.tick_params(colors="#C9D1D9", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363D")

        epochs     = [r["epoch"] for r in self._history]
        train_loss = [r["train_loss"] for r in self._history]
        val_loss   = [r["val_loss"]   for r in self._history]

        ax.plot(epochs, train_loss, "o-", color="#4FC3F7",
                linewidth=2, markersize=4, label="Train Loss")
        ax.plot(epochs, val_loss, "s--", color="#EF5350",
                linewidth=2, markersize=4, label="Val Loss")
        ax.set_xlabel("Epoch", color="#C9D1D9", fontsize=8)
        ax.set_ylabel("BCE Loss", color="#C9D1D9", fontsize=8)
        ax.set_title("Loss Curves", color="#4FC3F7", fontsize=9)
        ax.legend(fontsize=8, facecolor="#0D1117", labelcolor="#C9D1D9")
        self._fig_loss.tight_layout(pad=1.5)
        self._canvas_loss.draw()

    # ------------------------------------------------------------------
    def _render_f2(self):
        for ax in [self._ax_f2, self._ax_macc]:
            ax.clear()
            ax.set_facecolor("#0D1117")
            ax.tick_params(colors="#C9D1D9", labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#30363D")

        epochs   = [r["epoch"]            for r in self._history]
        val_f2   = [r["val_f2"]           for r in self._history]
        m_acc    = [r["val_mining_acc"]   for r in self._history]

        self._ax_f2.plot(epochs, val_f2, "s-", color="#66BB6A",
                          linewidth=2, markersize=5)
        best_idx = int(np.argmax(val_f2))
        self._ax_f2.scatter([epochs[best_idx]], [val_f2[best_idx]],
                             color="#FFB300", zorder=5, s=80,
                             label=f"Best F2={val_f2[best_idx]:.4f}")
        self._ax_f2.set_title("Val F2-Score", color="#4FC3F7", fontsize=9)
        self._ax_f2.set_xlabel("Epoch", color="#C9D1D9", fontsize=8)
        self._ax_f2.legend(fontsize=8, facecolor="#0D1117", labelcolor="#C9D1D9")

        self._ax_macc.plot(epochs, [v * 100 for v in m_acc], "o-",
                            color="#FF8A65", linewidth=2, markersize=4)
        self._ax_macc.set_title("Mining Detection Acc (%)", color="#4FC3F7", fontsize=9)
        self._ax_macc.set_xlabel("Epoch", color="#C9D1D9", fontsize=8)
        self._ax_macc.set_ylim(0, 100)

        self._fig_f2.tight_layout(pad=1.5)
        self._canvas_f2.draw()

    # ------------------------------------------------------------------
    def _render_lr(self):
        ax = self._ax_lr_c
        ax.clear()
        ax.set_facecolor("#0D1117")
        ax.tick_params(colors="#C9D1D9", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363D")

        epochs = [r["epoch"] for r in self._history]
        lrs    = [r.get("lr", 0) for r in self._history]
        ax.semilogy(epochs, lrs, "o-", color="#CE93D8",
                    linewidth=2, markersize=4)
        ax.set_xlabel("Epoch", color="#C9D1D9", fontsize=8)
        ax.set_ylabel("LR (log scale)", color="#C9D1D9", fontsize=8)
        ax.set_title("Learning Rate Schedule (Cosine Annealing)",
                     color="#4FC3F7", fontsize=9)
        self._fig_lr.tight_layout(pad=1.5)
        self._canvas_lr.draw()

    # ------------------------------------------------------------------
    def _render_stats(self):
        if not self._history:
            return
        val_f2s   = [r["val_f2"]         for r in self._history]
        val_losses = [r["val_loss"]       for r in self._history]
        m_accs    = [r["val_mining_acc"]  for r in self._history]

        best_idx = int(np.argmax(val_f2s))
        self._stat_labels["Total Epochs"].setText(str(len(self._history)))
        self._stat_labels["Best Val F2"].setText(f"{max(val_f2s):.4f}")
        self._stat_labels["Best Mining Acc"].setText(
            f"{max(m_accs)*100:.1f}%"
        )
        self._stat_labels["Best Epoch"].setText(str(self._history[best_idx]["epoch"]))
        self._stat_labels["Min Val Loss"].setText(f"{min(val_losses):.4f}")
        self._stat_labels["Final LR"].setText(
            f"{self._history[-1].get('lr', 0):.2e}"
        )
