"""
Setup & Download tab — Kaggle credentials input, dataset download,
environment status check.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import torch
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.core.downloader import DatasetDownloader
from app.utils.config import MODELS_DIR, TRAIN_CSV, TRAIN_IMG_DIR
from app.utils.logger import get_logger

log = get_logger("setup_tab")


# ── Background worker ────────────────────────────────────────────────────
class DownloadWorker(QThread):
    log_signal  = pyqtSignal(str)
    done_signal = pyqtSignal(bool)

    def __init__(self, username: str, api_key: str):
        super().__init__()
        self.username = username
        self.api_key  = api_key

    def run(self):
        dl = DatasetDownloader(username=self.username, api_key=self.api_key)
        ok = dl.download(log_fn=self.log_signal.emit)
        self.done_signal.emit(ok)


# ── Tab widget ───────────────────────────────────────────────────────────
class SetupTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: DownloadWorker | None = None
        self._build_ui()
        self._refresh_status()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(20, 20, 20, 20)

        # ── Environment status ──────────────────────────────────────
        env_box = QGroupBox("🖥  Environment Status")
        env_layout = QFormLayout(env_box)
        self._lbl_python  = QLabel()
        self._lbl_torch   = QLabel()
        self._lbl_gpu     = QLabel()
        self._lbl_dataset = QLabel()
        self._lbl_model   = QLabel()
        for row_label, widget in [
            ("Python",  self._lbl_python),
            ("PyTorch", self._lbl_torch),
            ("GPU",     self._lbl_gpu),
            ("Dataset", self._lbl_dataset),
            ("Model",   self._lbl_model),
        ]:
            env_layout.addRow(row_label, widget)

        btn_refresh = QPushButton("🔄 Refresh Status")
        btn_refresh.clicked.connect(self._refresh_status)
        env_layout.addRow("", btn_refresh)
        root.addWidget(env_box)

        # ── Kaggle Credentials ──────────────────────────────────────
        cred_box = QGroupBox("🔑  Kaggle API Credentials")
        cred_layout = QFormLayout(cred_box)

        self._txt_username = QLineEdit()
        self._txt_username.setPlaceholderText("your-kaggle-username")
        cred_layout.addRow("Username", self._txt_username)

        self._txt_apikey = QLineEdit()
        self._txt_apikey.setPlaceholderText("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self._txt_apikey.setEchoMode(QLineEdit.EchoMode.Password)
        cred_layout.addRow("API Key", self._txt_apikey)

        tip = QLabel(
            '💡 Get your key: kaggle.com → Account → <b>Create New API Token</b>. '
            'Then accept the competition rules at '
            '<a href="https://www.kaggle.com/c/planet-understanding-the-amazon-from-space">'
            'this link</a> before downloading.'
        )
        tip.setOpenExternalLinks(True)
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #80DEEA; font-size: 11px;")
        cred_layout.addRow("", tip)
        root.addWidget(cred_box)

        # ── Download controls ───────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_download = QPushButton("⬇  Download Dataset  (~1.5 GB)")
        self._btn_download.setFixedHeight(40)
        self._btn_download.setStyleSheet(
            "QPushButton { background:#1565C0; color:white; font-weight:bold; "
            "border-radius:6px; } QPushButton:hover { background:#1976D2; }"
            "QPushButton:disabled { background:#37474F; color:#78909C; }"
        )
        self._btn_download.clicked.connect(self._start_download)

        self._btn_cancel = QPushButton("✕ Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setFixedHeight(40)
        self._btn_cancel.clicked.connect(self._cancel_download)

        btn_row.addWidget(self._btn_download, 3)
        btn_row.addWidget(self._btn_cancel, 1)
        root.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # ── Log console ─────────────────────────────────────────────
        log_box = QGroupBox("📋  Download Log")
        log_layout = QVBoxLayout(log_box)
        self._console = QPlainTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumBlockCount(500)
        self._console.setStyleSheet(
            "QPlainTextEdit { background:#0D1117; color:#C9D1D9; "
            "font-family: Consolas, monospace; font-size: 11px; border-radius:6px; }"
        )
        self._console.setFixedHeight(200)
        log_layout.addWidget(self._console)
        root.addWidget(log_box)
        root.addStretch()

        self._pre_fill_from_env()

    # ------------------------------------------------------------------
    def _pre_fill_from_env(self):
        self._txt_username.setText(os.environ.get("KAGGLE_USERNAME", ""))
        self._txt_apikey.setText(os.environ.get("KAGGLE_KEY", ""))

    # ------------------------------------------------------------------
    def _refresh_status(self):
        def ok(txt):  return f'<span style="color:#66BB6A;">✅ {txt}</span>'
        def warn(txt): return f'<span style="color:#FFB300;">⚠ {txt}</span>'
        def err(txt):  return f'<span style="color:#EF5350;">❌ {txt}</span>'

        self._lbl_python.setText(
            ok(f"{platform.python_version()} ({sys.executable})")
        )
        self._lbl_torch.setText(ok(f"PyTorch {torch.__version__}"))

        if torch.cuda.is_available():
            self._lbl_gpu.setText(ok(f"{torch.cuda.get_device_name(0)} "
                                      f"(CUDA {torch.version.cuda})"))
        else:
            self._lbl_gpu.setText(warn("No GPU detected — CPU training will be slow"))

        if TRAIN_CSV.exists() and TRAIN_IMG_DIR.exists():
            n_imgs = len(list(TRAIN_IMG_DIR.glob("*.jpg")))
            self._lbl_dataset.setText(ok(f"{n_imgs:,} images found"))
        else:
            self._lbl_dataset.setText(err("Dataset not found — please download"))

        model_path = MODELS_DIR / "best_model.pth"
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1e6
            self._lbl_model.setText(ok(f"best_model.pth ({size_mb:.1f} MB)"))
        else:
            self._lbl_model.setText(warn("No trained model yet"))

    # ------------------------------------------------------------------
    def _start_download(self):
        username = self._txt_username.text().strip()
        api_key  = self._txt_apikey.text().strip()
        if not username or not api_key:
            self._console.appendPlainText("⚠ Please enter your Kaggle username and API key.")
            return

        self._btn_download.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._console.clear()

        self._worker = DownloadWorker(username, api_key)
        self._worker.log_signal.connect(self._console.appendPlainText)
        self._worker.done_signal.connect(self._download_done)
        self._worker.start()

    # ------------------------------------------------------------------
    def _cancel_download(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._console.appendPlainText("⛔ Download cancelled by user.")
        self._reset_buttons()

    # ------------------------------------------------------------------
    def _download_done(self, success: bool):
        self._reset_buttons()
        if success:
            self._console.appendPlainText("\n🎉 Dataset download complete! You can now train the model.")
        else:
            self._console.appendPlainText("\n❌ Download failed. Check the log above for details.")
        self._refresh_status()

    # ------------------------------------------------------------------
    def _reset_buttons(self):
        self._btn_download.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
