"""Logger — writes to file + emits Qt signals for in-app console display."""
import logging
import sys
from pathlib import Path
from app.utils.config import LOGS_DIR

LOG_FILE = LOGS_DIR / "app.log"


def get_logger(name: str = "mining_detector") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
