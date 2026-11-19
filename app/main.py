"""
Application entry point.

Run with:
    python app/main.py
or from the project root:
    python -m app.main
"""
import sys
import os

# Ensure the project root is on sys.path so `app.*` imports work
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from app.ui.main_window import MainWindow


def main():
    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Illegal Mining Detector")
    app.setOrganizationName("AmazonAI")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
