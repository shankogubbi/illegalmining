"""
Main window — dark-themed PyQt6 app shell with tab bar.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.ui.about_tab     import AboutTab
from app.ui.dashboard_tab import DashboardTab
from app.ui.predict_tab   import PredictTab
from app.ui.setup_tab     import SetupTab
from app.ui.train_tab     import TrainTab


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0D1117;
    color: #C9D1D9;
}
QTabWidget::pane {
    border: 1px solid #30363D;
    background: #0D1117;
    border-radius: 4px;
}
QTabBar::tab {
    background: #161B22;
    color: #8B949E;
    padding: 8px 20px;
    border: 1px solid #30363D;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 12px;
}
QTabBar::tab:selected {
    background: #0D1117;
    color: #4FC3F7;
    border-bottom: 2px solid #4FC3F7;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background: #1F2937;
    color: #C9D1D9;
}
QGroupBox {
    border: 1px solid #30363D;
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
    color: #8B949E;
    font-size: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #4FC3F7;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 4px;
    color: #C9D1D9;
    padding: 4px 8px;
    selection-background-color: #1F6FEB;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #4FC3F7;
}
QPushButton {
    background: #21262D;
    color: #C9D1D9;
    border: 1px solid #30363D;
    border-radius: 5px;
    padding: 5px 14px;
    font-size: 12px;
}
QPushButton:hover {
    background: #30363D;
    border-color: #4FC3F7;
}
QPushButton:pressed {
    background: #0D1117;
}
QPushButton:disabled {
    color: #484F58;
    border-color: #21262D;
}
QProgressBar {
    border: 1px solid #30363D;
    border-radius: 4px;
    background: #161B22;
    text-align: center;
    color: #C9D1D9;
    font-size: 10px;
}
QProgressBar::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #1565C0, stop:1 #4FC3F7
    );
    border-radius: 3px;
}
QScrollBar:vertical {
    background: #0D1117;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #30363D;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QLabel { color: #C9D1D9; }
QFormLayout { spacing: 8px; }
QSplitter::handle { background: #30363D; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌎 Illegal Mining Detector — Amazon Rainforest AI")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 820)
        self.setStyleSheet(DARK_STYLESHEET)

        # ── Header ──────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(56)
        header.setStyleSheet(
            "background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0,"
            "stop:0 #0D2137, stop:0.5 #0D3B55, stop:1 #0D1117);"
        )
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(20, 0, 20, 0)
        title_lbl = QLabel(
            "🌎  <b>Illegal Mining Detector</b>  "
            '<span style="color:#78909C; font-size:13px; font-weight:normal;">'
            "| Amazon Rainforest AI — EfficientNet-B0 + Grad-CAM"
            "</span>"
        )
        title_lbl.setStyleSheet("font-size: 17px; color: #4FC3F7;")
        title_lbl.setTextFormat(Qt.TextFormat.RichText)
        h_layout.addWidget(title_lbl)

        # ── Tabs ────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._setup_tab     = SetupTab()
        self._train_tab     = TrainTab()
        self._predict_tab   = PredictTab()
        self._dashboard_tab = DashboardTab()
        self._about_tab     = AboutTab()

        self._tabs.addTab(self._setup_tab,     "⚙  Setup & Download")
        self._tabs.addTab(self._train_tab,     "🧠  Train Model")
        self._tabs.addTab(self._predict_tab,   "🔍  Predict")
        self._tabs.addTab(self._dashboard_tab, "📊  Dashboard")
        self._tabs.addTab(self._about_tab,     "📖  Guide")

        # ── Root layout ─────────────────────────────────────────────
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(header)
        root.addWidget(self._tabs)
        self.setCentralWidget(central)

        # ── Status bar ───────────────────────────────────────────────
        self._status = QStatusBar()
        self._status.setStyleSheet(
            "QStatusBar { background: #161B22; color: #8B949E; font-size: 11px; "
            "border-top: 1px solid #30363D; }"
        )
        self.setStatusBar(self._status)
        self._update_status()

    def _update_status(self):
        import torch
        device = "CUDA: " + torch.cuda.get_device_name(0) \
            if torch.cuda.is_available() else "CPU (no GPU)"
        self._status.showMessage(
            f"Device: {device}   |   "
            f"PyTorch {torch.__version__}   |   "
            "Ready — Start in the Setup tab if you haven't downloaded the dataset yet."
        )
