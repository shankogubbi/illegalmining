"""Read-only full image integrity audit and optional legacy validation replay.

Run: python -m app.workflows.audit [--evaluate]
Replay is diagnostic, NOT an independent test: old split membership is unknown.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from app.core.dataset import AmazonDataset, get_val_transforms
from app.core.metrics import mining_metrics, decision_scores
from app.core.model import build_model, load_checkpoint
from app.utils.config import ALL_TAGS, TRAIN_CSV, TRAIN_IMG_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    df = pd.read_csv(TRAIN_CSV)
    counts = Counter(t for tags in df.tags for t in set(tags.split()))
    bad, valid = [], []
    hashes = defaultdict(list)
    sizes = Counter()
    for i, row in df.iterrows():
        path = TRAIN_IMG_DIR / (row.image_name + '.jpg')
        if not path.exists():
            path = path.with_suffix('.tif')
        try:
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                rgb.load()
                sizes[str(rgb.size)] += 1
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            hashes[digest].append(row.image_name)
            valid.append(i)
        except (OSError, ValueError) as exc:
            bad.append(dict(image=row.image_name, error=str(exc)))
        if (i + 1) % 5000 == 0:
            print(f'Checked {i + 1}/{len(df)}', flush=True)
    report = dict(rows=len(df), readable=len(valid), unreadable=bad,
                  rows_with_repeated_tags=int(df.tags.map(lambda s: len(s.split()) != len(set(s.split()))).sum()),
                  duplicate_names=int(df.image_name.duplicated().sum()),
                  unknown_tags=sorted(set(counts) - set(ALL_TAGS)),
                  class_counts=dict(counts), dimensions=dict(sizes),
                  exact_pixel_duplicate_groups=[v for v in hashes.values() if len(v) > 1],
                  legal_status_labels_available=False,
                  independent_test_available=False)
    path = RESULTS_DIR / 'data_model_audit.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.evaluate:
        clean = df.iloc[valid].reset_index(drop=True)
        _, val = train_test_split(clean, test_size=0.15, random_state=42)
        model = build_model('cpu', pretrained=False)
        ckpt = load_checkpoint(model)
        model.eval()
        loader = DataLoader(AmazonDataset(val, TRAIN_IMG_DIR, get_val_transforms(ckpt.get('image_size', 224))), batch_size=32)
        probs, labels = [], []
        with torch.inference_mode():
            for i, (x, y) in enumerate(loader):
                probs.append(model(x).sigmoid().numpy())
                labels.append(y.numpy())
                if i % 20 == 0:
                    print(f'Inference batch {i + 1}/{len(loader)}', flush=True)
        p, y = np.concatenate(probs), np.concatenate(labels)
        decisions = decision_scores(p, ckpt['thresholds']) if 'thresholds' in ckpt else p
        report['checkpoint'] = dict(epoch=ckpt.get('epoch'), metrics=ckpt.get('metrics'))
        report['diagnostic_replay'] = dict(
            warning='Split reconstructed from current readable rows; training overlap unknown. Not independent test performance.',
            mining=mining_metrics(y, decisions),
            per_mining_tag={tag: mining_metrics(
                np.repeat(y[:, j:j+1], len(ALL_TAGS), axis=1),
                np.repeat(decisions[:, j:j+1], len(ALL_TAGS), axis=1))
                for j, tag in enumerate(ALL_TAGS) if 'mine' in tag})
        np.savez_compressed(RESULTS_DIR / 'audit_predictions.npz', probabilities=p, labels=y,
                            image_names=val.image_name.to_numpy(dtype=str))
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['unreadable', 'exact_pixel_duplicate_groups']}, indent=2), flush=True)
    print(f'Full report: {path}', flush=True)


if __name__ == '__main__':
    main()
