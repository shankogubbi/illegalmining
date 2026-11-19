"""
Downloader — handles downloading & extracting the Amazon dataset.

Primary: Downloads the verified Kaggle mirror dataset 'nikitarom/planets-dataset'
         (which includes all 40,479 JPG images and train_v2.csv in ~1.5 GB).
Fallback: Can also attempt downloading directly from the Kaggle competition.
Organizes files into data/raw/train-jpg/ and data/raw/train_v2.csv.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable, Optional

from app.utils.config import KAGGLE_COMPETITION, RAW_DIR, TRAIN_CSV, TRAIN_IMG_DIR
from app.utils.logger import get_logger

log = get_logger("downloader")

# Kaggle mirror dataset hosting the 40k+ Planet Amazon JPG images
KAGGLE_MIRROR_DATASET = "nikitarom/planets-dataset"


class DatasetDownloader:
    """
    Downloads and extracts the Planet Amazon dataset.
    """

    def __init__(
        self,
        dest: Path = RAW_DIR,
        username: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.dest     = Path(dest)
        self.username = username
        self.api_key  = api_key

    # ------------------------------------------------------------------
    def _set_env(self):
        if self.username:
            os.environ["KAGGLE_USERNAME"] = self.username
        if self.api_key:
            os.environ["KAGGLE_KEY"] = self.api_key

    # ------------------------------------------------------------------
    def check_credentials(self) -> bool:
        """Return True if credentials are configured or cached."""
        try:
            self._set_env()
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    def is_dataset_ready(self) -> bool:
        """Check every labeled filename; this does not replace a decode audit."""
        if not TRAIN_CSV.is_file() or not TRAIN_IMG_DIR.is_dir():
            return False
        import pandas as pd
        try:
            df = pd.read_csv(TRAIN_CSV)
            if not {'image_name', 'tags'}.issubset(df.columns) or df.empty:
                return False
            return all((TRAIN_IMG_DIR / f'{name}.jpg').is_file() or
                       (TRAIN_IMG_DIR / f'{name}.tif').is_file()
                       for name in df.image_name)
        except (OSError, ValueError):
            return False

    # ------------------------------------------------------------------
    def _organize_extracted_files(self, log_fn: Callable[[str], None]):
        """
        Organize extracted directories/files into:
          - data/raw/train_v2.csv
          - data/raw/train-jpg/*.jpg
        """
        self.dest.mkdir(parents=True, exist_ok=True)

        # 1. Normalize train_v2.csv
        # Check if train_v2.csv is a directory containing train_v2.csv
        nested_csv_dir = self.dest / "train_v2.csv"
        if nested_csv_dir.is_dir():
            candidates = list(nested_csv_dir.rglob("*.csv"))
            if candidates:
                tmp_csv = self.dest / "train_v2_normalized.csv"
                shutil.copy2(candidates[0], tmp_csv)
                shutil.rmtree(nested_csv_dir)
                tmp_csv.rename(self.dest / "train_v2.csv")
                log_fn("   ✓ Extracted train_v2.csv file")

        # Check if CSV was extracted under a subfolder (e.g. planet/planet/train_classes.csv or train_v2.csv)
        if not (self.dest / "train_v2.csv").is_file():
            found_csvs = list(self.dest.rglob("train*.csv"))
            for fc in found_csvs:
                if fc.name in ("train_v2.csv", "train_classes.csv"):
                    shutil.copy2(fc, self.dest / "train_v2.csv")
                    log_fn(f"   ✓ Found and placed {fc.name} -> train_v2.csv")
                    break

        # 2. Extract nested archives (tar, 7z, zip)
        for zf in list(self.dest.rglob("*.zip")):
            if zf.parent != self.dest and "venv" not in str(zf):
                try:
                    log_fn(f"   📦 Extracting nested {zf.name} …")
                    with zipfile.ZipFile(zf, "r") as z:
                        z.extractall(self.dest)
                    zf.unlink()
                except Exception as e:
                    log_fn(f"   ⚠ Could not extract {zf.name}: {e}")

        # Check for tar.gz or tar archives
        for tf in list(self.dest.rglob("*.tar*")):
            try:
                import tarfile
                log_fn(f"   📦 Extracting tarball {tf.name} …")
                with tarfile.open(tf, "r:*") as t:
                    t.extractall(self.dest)
                tf.unlink()
            except Exception as e:
                log_fn(f"   ⚠ Could not extract {tf.name}: {e}")

        # 3. Locate train-jpg folder and ensure it's at data/raw/train-jpg
        TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
        current_count = len(list(TRAIN_IMG_DIR.glob("*.jpg")))
        if current_count < 100:
            # Search for any directory containing train_*.jpg or inside planet/
            for d in list(self.dest.rglob("*")):
                if d.is_dir() and d != TRAIN_IMG_DIR and "train" in d.name.lower():
                    jpgs = list(d.glob("*.jpg"))
                    if jpgs:
                        log_fn(f"   🚚 Moving {len(jpgs)} images from {d.name} to train-jpg/ …")
                        for img in jpgs:
                            dest_file = TRAIN_IMG_DIR / img.name
                            if not dest_file.exists():
                                shutil.move(str(img), str(dest_file))
                        break

        # Also check for images in a generic 'images' or 'train' folder
        current_count = len(list(TRAIN_IMG_DIR.glob("*.jpg")))
        if current_count < 100:
            for img in self.dest.rglob("train_*.jpg"):
                dest_file = TRAIN_IMG_DIR / img.name
                if not dest_file.exists():
                    shutil.move(str(img), str(dest_file))

    # ------------------------------------------------------------------
    def download(self, log_fn: Callable[[str], None] = print) -> bool:
        """
        Download the Planet Amazon images and labels via Kaggle CLI.
        """
        if self.is_dataset_ready():
            n = len(list(TRAIN_IMG_DIR.glob("*.jpg")))
            log_fn(f"✅ Dataset already present ({n:,} images found) — skipping download.")
            return True

        self._set_env()
        self.dest.mkdir(parents=True, exist_ok=True)

        # ── Method 1: Download the complete Kaggle mirror dataset ────────
        log_fn("⏬ Downloading Planet Amazon dataset (~1.5 GB) …")
        log_fn(f"   Source: {KAGGLE_MIRROR_DATASET}")
        log_fn("   Please keep this running until extraction completes.")

        cmd = [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", KAGGLE_MIRROR_DATASET,
            "-p", str(self.dest),
            "--unzip",
        ]

        success = False
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                line_clean = line.rstrip()
                if line_clean:
                    log_fn(line_clean)
            proc.wait()
            if proc.returncode == 0:
                success = True
            else:
                log_fn(f"⚠ Mirror download exited with code {proc.returncode}. Trying competition files…")
        except Exception as e:
            log_fn(f"⚠ Mirror download error: {e}")

        # ── Method 2: Fallback to competition download ───────────────────
        if not success:
            log_fn(f"⏬ Trying competition download: {KAGGLE_COMPETITION} …")
            cmd_comp = [
                sys.executable, "-m", "kaggle", "competitions", "download",
                "-c", KAGGLE_COMPETITION,
                "-p", str(self.dest),
            ]
            try:
                proc = subprocess.Popen(
                    cmd_comp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    line_clean = line.rstrip()
                    if line_clean:
                        log_fn(line_clean)
                proc.wait()
            except Exception as e:
                log_fn(f"❌ Competition download error: {e}")

        # ── Extract any remaining zip files ─────────────────────────────
        zips = list(self.dest.glob("*.zip"))
        for zf_path in zips:
            try:
                log_fn(f"📦 Extracting {zf_path.name} …")
                with zipfile.ZipFile(zf_path, "r") as zf:
                    zf.extractall(self.dest)
                zf_path.unlink()
                log_fn(f"   ✓ Extracted {zf_path.name}")
            except Exception as e:
                log_fn(f"   ⚠ Error extracting {zf_path.name}: {e}")

        # ── Organize directory hierarchy ────────────────────────────────
        log_fn("🔧 Organizing image directories and label files …")
        self._organize_extracted_files(log_fn)

        if self.is_dataset_ready():
            n = len(list(TRAIN_IMG_DIR.glob("*.jpg")))
            log_fn(f"🎉 Dataset ready! {n:,} training images verified.")
            log_fn(f"   Labels: {TRAIN_CSV}")
            log_fn(f"   Images: {TRAIN_IMG_DIR}")
            return True
        else:
            n = len(list(TRAIN_IMG_DIR.glob("*.jpg"))) if TRAIN_IMG_DIR.exists() else 0
            has_csv = TRAIN_CSV.is_file()
            log_fn(f"⚠ Download finished but verified count is low ({n} images, CSV present: {has_csv}).")
            log_fn("   Please check the data/raw directory or retry.")
            return False
