"""
Predict tab — single image + batch folder inference with Grad-CAM overlay,
tag probability bars, and CSV export.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from app.utils.config import ALL_TAGS, MINING_ALERT_THRESHOLD, MODELS_DIR, RESULTS_DIR
from app.utils.logger import get_logger

log = get_logger("predict_tab")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ── Worker thread ─────────────────────────────────────────────────────────
class PredictWorker(QThread):
    result_signal   = pyqtSignal(object)    # single PredictionResult
    progress_signal = pyqtSignal(int, int)  # current, total
    error_signal    = pyqtSignal(str)
    done_signal     = pyqtSignal(list)      # List[PredictionResult]

    def __init__(self, paths: List[Path], single: bool = True):
        super().__init__()
        self.paths  = paths
        self.single = single

    def run(self):
        try:
            from app.core.predictor import Predictor
            device = "cuda"
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

            predictor = Predictor(device=device)
            results = []

            for i, p in enumerate(self.paths):
                try:
                    r = predictor.predict(p, generate_cam=(self.single or i == 0))
                    results.append(r)
                    self.result_signal.emit(r)
                except Exception as e:
                    log.error(f"Error predicting {p}: {e}")
                    if self.single:
                        self.error_signal.emit(f"Could not analyse {p.name}: {e}")
                self.progress_signal.emit(i + 1, len(self.paths))

            self.done_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))


# ── Drag-and-drop image label ─────────────────────────────────────────────
class DropImageLabel(QLabel):
    image_dropped = pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 240)
        self.setStyleSheet(
            "QLabel { border: 2px dashed #37474F; border-radius: 8px; "
            "color: #607D8B; background: #0D1117; font-size: 13px; }"
        )
        self.setText("📂  Drop an image here\nor click Browse")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            p = Path(urls[0].toLocalFile())
            if p.suffix.lower() in IMG_EXTS:
                self.image_dropped.emit(p)

    def show_image(self, pil_image):
        from PIL.ImageQt import toqimage
        qimg = toqimage(pil_image.convert("RGB"))
        pix = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pix)


# ── Probability bar chart ─────────────────────────────────────────────────
class ProbChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 3.5), facecolor="#0D1117")
        super().__init__(self.fig)
        self.setParent(parent)
        self._ax = self.fig.add_subplot(111)
        self._style()

    def _style(self):
        self._ax.set_facecolor("#0D1117")
        self._ax.tick_params(colors="#C9D1D9", labelsize=7)
        for sp in self._ax.spines.values():
            sp.set_color("#30363D")
        self.fig.tight_layout(pad=0.8)

    def set_probabilities(self, all_probs: dict):
        """Render probabilities without shadowing QWidget.update()."""
        self._ax.clear()
        self._style()
        tags  = list(all_probs.keys())
        probs = list(all_probs.values())
        colors = []
        for t, p in zip(tags, probs):
            if "mine" in t:
                colors.append("#EF5350" if p > MINING_ALERT_THRESHOLD else "#FF8A65")
            else:
                colors.append("#4FC3F7" if p > 0.5 else "#37474F")

        bars = self._ax.barh(tags, probs, color=colors, height=0.6)
        self._ax.set_xlim(0, 1)
        self._ax.set_xlabel("Uncalibrated model score", color="#C9D1D9", fontsize=8)
        self._ax.set_title("Tag Scores", color="#4FC3F7", fontsize=9)
        self._ax.axvline(0.5, color="#78909C", linewidth=0.8, linestyle="--")
        self.fig.tight_layout(pad=0.8)
        self.draw()


# ── Predict tab widget ────────────────────────────────────────────────────
class PredictTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: PredictWorker | None = None
        self._batch_results: list = []
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Toolbar ────────────────────────────────────────────────
        tb = QHBoxLayout()
        self._btn_browse = QPushButton("📂 Browse Image")
        self._btn_browse.clicked.connect(self._browse_single)
        self._btn_batch  = QPushButton("📁 Batch Scan Folder")
        self._btn_batch.clicked.connect(self._browse_batch)
        self._btn_export = QPushButton("💾 Export CSV")
        self._btn_export.clicked.connect(self._export_csv)
        self._btn_export.setEnabled(False)

        for btn in [self._btn_browse, self._btn_batch, self._btn_export]:
            btn.setFixedHeight(34)
            tb.addWidget(btn)

        self._prog = QProgressBar()
        self._prog.setVisible(False)
        self._prog.setFixedHeight(18)
        tb.addWidget(self._prog)
        root.addLayout(tb)

        # ── Alert banner ───────────────────────────────────────────
        self._alert_banner = QLabel("  Load an image to analyse it")
        self._alert_banner.setFixedHeight(40)
        self._alert_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._alert_banner.setStyleSheet(
            "QLabel { background:#1A2730; color:#78909C; font-size:14px; "
            "font-weight:bold; border-radius:6px; }"
        )
        root.addWidget(self._alert_banner)
        reliability_note = QLabel(
            "Research model: scores are uncalibrated. Low scores do not rule out mining. "
            "Images alone do not establish legality; human review is required."
        )
        reliability_note.setWordWrap(True)
        root.addWidget(reliability_note)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: original + heatmap ───────────────────────────────
        img_panel = QWidget()
        img_layout = QVBoxLayout(img_panel)
        img_layout.setContentsMargins(0, 0, 0, 0)

        self._lbl_orig = DropImageLabel()
        self._lbl_orig.image_dropped.connect(self._run_prediction)
        img_layout.addWidget(self._lbl_orig, stretch=1)

        cam_box = QGroupBox("🔥 Grad-CAM Heatmap")
        cam_layout = QVBoxLayout(cam_box)
        self._lbl_cam = QLabel("— Heatmap will appear here —")
        self._lbl_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_cam.setMinimumHeight(180)
        self._lbl_cam.setStyleSheet(
            "QLabel { border-radius:6px; background:#0D1117; color:#607D8B; }"
        )
        cam_layout.addWidget(self._lbl_cam)
        img_layout.addWidget(cam_box, stretch=1)
        splitter.addWidget(img_panel)

        # ── Right: prob chart + batch table ───────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._prob_chart = ProbChart()
        right_layout.addWidget(self._prob_chart, stretch=2)

        # Mining score large display
        score_box = QGroupBox("⛏  Mining Analysis")
        score_layout = QFormLayout(score_box)
        self._lbl_score     = QLabel("—")
        self._lbl_score.setStyleSheet("font-size:20px; font-weight:bold; color:#EF5350;")
        self._lbl_alert     = QLabel("—")
        self._lbl_tags      = QLabel("—")
        self._lbl_tags.setWordWrap(True)
        score_layout.addRow("Mining Score", self._lbl_score)
        score_layout.addRow("Alert Level",  self._lbl_alert)
        score_layout.addRow("Active Tags",  self._lbl_tags)
        right_layout.addWidget(score_box, stretch=1)

        # Batch results table
        batch_box = QGroupBox("📋 Batch Results")
        batch_layout = QVBoxLayout(batch_box)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Image", "Mining Score", "Alert", "Tags", "Top Prob"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setStyleSheet(
            "QTableWidget { background:#0D1117; color:#C9D1D9; "
            "gridline-color:#30363D; border: none; }"
            "QHeaderView::section { background:#1E2D3D; color:#4FC3F7; }"
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        batch_layout.addWidget(self._table)
        right_layout.addWidget(batch_box, stretch=2)
        splitter.addWidget(right)
        splitter.setSizes([420, 580])
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    def _browse_single(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)"
        )
        if path:
            self._run_prediction(Path(path))

    # ------------------------------------------------------------------
    def _browse_batch(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        paths = [p for p in Path(folder).rglob("*")
                 if p.suffix.lower() in IMG_EXTS]
        if not paths:
            self._alert_banner.setText("⚠ No images found in that folder.")
            return
        self._run_prediction_batch(paths)

    # ------------------------------------------------------------------
    def _check_model(self) -> bool:
        model_path = MODELS_DIR / "best_model.pth"
        if not model_path.exists():
            self._alert_banner.setStyleSheet(
                "QLabel { background:#3B2F00; color:#FFB300; font-size:14px; "
                "font-weight:bold; border-radius:6px; }"
            )
            self._alert_banner.setText(
                "⚠  No trained model found. Please train the model in the Train tab first."
            )
            return False
        return True

    # ------------------------------------------------------------------
    def _run_prediction(self, path: Path):
        if self._worker is not None and self._worker.isRunning():
            return
        if not self._check_model():
            return

        # Show original image immediately
        from PIL import Image
        try:
            with Image.open(path) as source:
                img = source.convert("RGB")
            self._lbl_orig.show_image(img)
        except Exception as e:
            self._on_error(f"Could not open {path.name}: {e}")
            return

        self._set_busy(True)
        self._worker = PredictWorker([path], single=True)
        self._worker.result_signal.connect(self._on_single_result)
        self._worker.error_signal.connect(self._on_error)
        self._worker.done_signal.connect(lambda _: self._set_busy(False))
        self._worker.start()

    # ------------------------------------------------------------------
    def _run_prediction_batch(self, paths: List[Path]):
        if self._worker is not None and self._worker.isRunning():
            return
        if not self._check_model():
            return
        self._batch_results.clear()
        self._table.setRowCount(0)
        self._set_busy(True)
        self._prog.setVisible(True)
        self._prog.setRange(0, len(paths))

        self._worker = PredictWorker(paths, single=False)
        self._worker.result_signal.connect(self._on_batch_result)
        self._worker.progress_signal.connect(
            lambda cur, tot: self._prog.setValue(cur)
        )
        self._worker.error_signal.connect(self._on_error)
        self._worker.done_signal.connect(self._on_batch_done)
        self._worker.start()

    # ------------------------------------------------------------------
    def _on_single_result(self, result):
        self._update_alert_banner(result)
        self._prob_chart.set_probabilities(result.all_probs)
        self._lbl_score.setText(f"{result.mining_score*100:.1f}%")
        self._lbl_alert.setText(result.alert_level)
        self._lbl_tags.setText(", ".join(result.tags) if result.tags else "none")

        if result.heatmap_pil:
            self._lbl_cam.show_image(result.heatmap_pil) if hasattr(
                self._lbl_cam, "show_image"
            ) else self._show_cam(result.heatmap_pil)

    def _show_cam(self, pil_image):
        from PIL.ImageQt import toqimage
        qimg = toqimage(pil_image.convert("RGB"))
        pix = QPixmap.fromImage(qimg).scaled(
            self._lbl_cam.width(), self._lbl_cam.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._lbl_cam.setPixmap(pix)

    # ------------------------------------------------------------------
    def _on_batch_result(self, result):
        self._batch_results.append(result)
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(result.image_path.name))
        score_item = QTableWidgetItem(f"{result.mining_score*100:.1f}%")
        if result.alert_level == "CRITICAL":
            score_item.setForeground(Qt.GlobalColor.red)
        elif result.alert_level == "WARNING":
            score_item.setForeground(Qt.GlobalColor.yellow)
        self._table.setItem(row, 1, score_item)
        self._table.setItem(row, 2, QTableWidgetItem(result.alert_level))
        self._table.setItem(row, 3, QTableWidgetItem(", ".join(result.tags[:4])))
        top_tag = max(result.all_probs, key=result.all_probs.get)
        self._table.setItem(row, 4, QTableWidgetItem(
            f"{top_tag}: {result.all_probs[top_tag]*100:.0f}%"
        ))

    # ------------------------------------------------------------------
    def _on_batch_done(self, results):
        self._set_busy(False)
        self._prog.setVisible(False)
        self._btn_export.setEnabled(bool(results))
        n_mining = sum(1 for r in results if r.alert_level != "CLEAR")
        self._alert_banner.setText(
            f"✅ Batch complete — {len(results)} images | "
            f"⛏ Mining detected: {n_mining}"
        )
        self._alert_banner.setStyleSheet(
            "QLabel { background:#1B3B1E; color:#66BB6A; font-size:13px; "
            "font-weight:bold; border-radius:6px; }"
        )

    # ------------------------------------------------------------------
    def _on_error(self, msg: str):
        self._set_busy(False)
        self._alert_banner.setText(f"❌ Error: {msg}")
        self._alert_banner.setStyleSheet(
            "QLabel { background:#3B1C1C; color:#EF5350; font-size:13px; "
            "font-weight:bold; border-radius:6px; }"
        )

    # ------------------------------------------------------------------
    def _update_alert_banner(self, result):
        styles = {
            "CRITICAL": ("background:#7B1C1C; color:#FF5252;", "🚨 High Mining Score — Review Required"),
            "WARNING":  ("background:#3B2F00; color:#FFD740;", "⚠  WARNING  — Possible Mining Activity"),
            "CLEAR":    ("background:#1A2730; color:#C9D1D9;", "Low Mining Score — Mining Not Ruled Out"),
        }
        style, text = styles[result.alert_level]
        self._alert_banner.setStyleSheet(
            f"QLabel {{ {style} font-size:15px; font-weight:bold; border-radius:6px; }}"
        )
        self._alert_banner.setText(
            f"{text}   |   Score: {result.mining_score*100:.1f}%"
        )

    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool):
        self._btn_browse.setEnabled(not busy)
        self._btn_batch.setEnabled(not busy)

    # ------------------------------------------------------------------
    def _export_csv(self):
        if not self._batch_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", str(RESULTS_DIR / "results.csv"),
            "CSV Files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image", "mining_score", "alert_level", "tags",
                *[f"prob_{t}" for t in ALL_TAGS]
            ])
            writer.writeheader()
            for r in self._batch_results:
                row = {
                    "image":        str(r.image_path),
                    "mining_score": round(r.mining_score, 4),
                    "alert_level":  r.alert_level,
                    "tags":         " ".join(r.tags),
                    **{f"prob_{t}": round(r.all_probs.get(t, 0), 4) for t in ALL_TAGS},
                }
                writer.writerow(row)
        self._alert_banner.setText(f"💾 Exported → {path}")
