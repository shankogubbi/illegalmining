"""
Central configuration for Illegal Mining Detector.
All paths, constants, and default hyperparameters live here.
"""
import os
from pathlib import Path

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT_DIR = Path(__file__).resolve().parents[2]          # repo root
DATA_DIR = ROOT_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
TRAIN_IMG_DIR = RAW_DIR / "train-jpg"
TEST_IMG_DIR  = RAW_DIR / "test-jpg"
TRAIN_CSV     = RAW_DIR / "train_v2.csv"

MODELS_DIR    = ROOT_DIR / "models"
LOGS_DIR      = ROOT_DIR / "logs"
RESULTS_DIR   = ROOT_DIR / "results"

for d in [DATA_DIR, RAW_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# â”€â”€ Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KAGGLE_COMPETITION = "planet-understanding-the-amazon-from-space"
IMAGE_SIZE = 224          # resize target
NUM_WORKERS = 0           # set >0 if on Linux/Mac

# 17 Amazon tags as present in the official Kaggle dataset train_v2.csv
# Note: Kaggle dataset uses 'artisinal_mine' (with an 'i')
ALL_TAGS = [
    "agriculture", "artisinal_mine", "bare_ground", "blooming",
    "blow_down", "clear", "cloudy", "conventional_mine", "cultivation",
    "habitation", "haze", "partly_cloudy", "primary", "road",
    "selective_logging", "slash_burn", "water",
]

# Indices of mining-related tags (used to compute final "mining score")
MINING_TAG_INDICES = [ALL_TAGS.index("artisinal_mine"),
                      ALL_TAGS.index("conventional_mine")]

NUM_CLASSES = len(ALL_TAGS)   # 17

# â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BACKBONE = "efficientnet_b0"
CHECKPOINT_NAME = "best_model.pth"

# â”€â”€ Training defaults (overridden by GUI) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEFAULT_EPOCHS      = 15
DEFAULT_BATCH_SIZE  = 32
DEFAULT_LR          = 1e-3
DEFAULT_WEIGHT_DECAY= 1e-4
DEFAULT_VAL_SPLIT   = 0.15
DEFAULT_PATIENCE    = 4       # early stopping patience

# â”€â”€ Inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEFAULT_THRESHOLD   = 0.5     # per-class sigmoid threshold
MINING_ALERT_THRESHOLD = 0.4  # lower â†’ more sensitive for mining detection
