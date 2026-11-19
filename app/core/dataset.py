"""
Dataset — PyTorch Dataset + DataLoader factories for the Amazon dataset.

The Planet Amazon dataset has multi-label tags per chip. We load
train_v2.csv (image_name, tags) and produce:
  - images : float32 tensors, shape (3, H, W), normalised to ImageNet stats
  - labels : float32 binary vectors, shape (17,)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.utils.config import (
    ALL_TAGS,
    DEFAULT_VAL_SPLIT,
    IMAGE_SIZE,
    NUM_WORKERS,
    TRAIN_CSV,
    TRAIN_IMG_DIR,
)


# ── Transforms ─────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transforms(size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        # Keep the whole chip: cropping/erasing can remove a small labeled mine.
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms(size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_predict_transforms(size: int = IMAGE_SIZE) -> transforms.Compose:
    return get_val_transforms(size)


# ── Dataset class ───────────────────────────────────────────────────────
class AmazonDataset(Dataset):
    """
    Multi-label dataset from Amazon / Planet Kaggle competition.

    Args:
        df         : DataFrame with columns ['image_name', 'tags']
        img_dir    : directory containing JPEG images
        transform  : torchvision transform pipeline
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path = TRAIN_IMG_DIR,
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.tag2idx = {tag: i for i, tag in enumerate(ALL_TAGS)}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['image_name']}.jpg"

        # Fall-back: try .tif extension
        if not img_path.exists():
            img_path = img_path.with_suffix(".tif")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Build multi-hot label vector
        label = torch.zeros(len(ALL_TAGS), dtype=torch.float32)
        tags = row["tags"].split() if isinstance(row["tags"], str) else []
        for t in tags:
            if t in self.tag2idx:
                label[self.tag2idx[t]] = 1.0

        return img, label


# ── DataLoader factories ────────────────────────────────────────────────
def build_dataloaders(
    csv_path: Path = TRAIN_CSV,
    img_dir: Path = TRAIN_IMG_DIR,
    val_split: float = DEFAULT_VAL_SPLIT,
    batch_size: int = 32,
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame]:
    """
    Returns (train_loader, val_loader, full_df).
    Raises FileNotFoundError if the CSV doesn't exist yet.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training CSV not found at {csv_path}.\n"
            "Please download the dataset first via the Setup tab."
        )

    df = pd.read_csv(csv_path)
    if not {"image_name", "tags"}.issubset(df.columns):
        raise ValueError("Training CSV must have image_name and tags columns.")
    if df['image_name'].isna().any() or df['image_name'].duplicated().any():
        raise ValueError("Training image names must be nonempty and unique.")
    if df['tags'].isna().any():
        raise ValueError("Missing labels must be resolved before training.")
    unknown = set(t for tags in df['tags'] for t in tags.split()) - set(ALL_TAGS)
    if unknown:
        raise ValueError(f"Unknown labels in training CSV: {sorted(unknown)}")

    # Check the actual decode rather than just the file extension or size: some
    # downloaded JPGs contain zeros or are truncated and crash mid-epoch.
    valid_rows = []
    bad_images = []
    for idx, name in enumerate(df["image_name"]):
        image_path = Path(img_dir) / f"{name}.jpg"
        if not image_path.is_file():
            image_path = image_path.with_suffix(".tif")
        try:
            with Image.open(image_path) as image:
                image.convert("RGB").load()
            valid_rows.append(idx)
        except (OSError, ValueError) as exc:
            bad_images.append(f"{image_path.name}: {exc}")

    if bad_images and log_fn:
        log_fn(f"⚠ Excluding {len(bad_images):,} missing/unreadable training images:")
        for message in bad_images[:10]:
            log_fn(f"   {message}")
        if len(bad_images) > 10:
            log_fn(f"   … and {len(bad_images) - 10:,} more.")
    df = df.iloc[valid_rows].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Fewer than two readable labeled images; check the dataset download.")

    mining_strata = df.tags.map(lambda tags: tuple(t in tags.split() for t in
                               ('artisinal_mine', 'conventional_mine'))).map(str)
    # Preserve rare mining combinations when there is enough support to split.
    strata_counts = mining_strata.value_counts()
    n_val = int(len(df) * val_split + 0.999999)
    stratify = mining_strata if (strata_counts.min() >= 2 and
        min(n_val, len(df) - n_val) >= len(strata_counts)) else None
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed, shuffle=True, stratify=stratify
    )

    train_ds = AmazonDataset(train_df, img_dir, get_train_transforms())
    val_ds   = AmazonDataset(val_df,   img_dir, get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, df


def get_class_distribution(csv_path: Path = TRAIN_CSV) -> dict:
    """Returns tag name → count mapping for the full training set."""
    df = pd.read_csv(csv_path)
    counts: dict = {t: 0 for t in ALL_TAGS}
    for tags_str in df["tags"]:
        for t in set(tags_str.split()):
            if t in counts:
                counts[t] += 1
    return counts
