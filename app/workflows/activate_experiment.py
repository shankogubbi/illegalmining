"""Verify the single active checkpoint against the completed experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from app.workflows.experiment import RUN_DIR, report_metrics, write_json
from app.core.dataset import AmazonDataset, get_val_transforms
from app.core.model import build_model, load_checkpoint
from app.core.trainer import fbeta_score
from app.utils.config import ALL_TAGS, TRAIN_CSV, TRAIN_IMG_DIR, MODELS_DIR, CHECKPOINT_NAME
import pandas as pd


def main():
    torch.set_num_threads(4)
    report_path = RUN_DIR / 'final_metrics.json'
    report = json.loads(report_path.read_text())
    candidate = MODELS_DIR / CHECKPOINT_NAME
    report['checkpoint'] = str(candidate)
    arrays = np.load(RUN_DIR / 'evaluation_predictions.npz')
    manifest = json.loads((RUN_DIR / 'split_manifest.json').read_text())
    df = pd.read_csv(TRAIN_CSV).set_index('image_name')
    names = manifest['partitions']['validation'][:8]
    sample = df.loc[names].reset_index()
    checkpoint = torch.load(candidate, map_location='cpu', weights_only=True)
    if checkpoint['tags'] != ALL_TAGS:
        raise ValueError('Checkpoint tag order mismatch.')
    model = build_model('cpu', pretrained=False)
    load_checkpoint(model, candidate)
    model.eval()
    dataset = AmazonDataset(sample, TRAIN_IMG_DIR, get_val_transforms(checkpoint['image_size']))
    x = torch.stack([dataset[i][0] for i in range(len(dataset))])
    with torch.inference_mode():
        actual = model(x).sigmoid().numpy()
    expected = arrays['validation_probs'][:len(sample)]
    error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=2e-5, rtol=2e-4):
        raise ValueError(f'Export parity check failed: max probability error {error}')
    # Recompute summaries from saved raw predictions using shared inference scoring.
    thresholds = arrays['thresholds'].tolist()
    for split in ['validation', 'test']:
        report[split] = report_metrics(arrays[f'{split}_labels'], arrays[f'{split}_probs'], thresholds)
    active = candidate
    new_hash = hashlib.sha256(active.read_bytes()).hexdigest()
    report['active_checkpoint_sha256'] = new_hash
    report.pop('previous_checkpoint_backup', None)
    write_json(report_path, report)
    write_json(RUN_DIR / 'deployment_check.json', dict(samples=len(sample),
               maximum_probability_error=error, active_checkpoint=str(active), sha256=new_hash))
    print(json.dumps(dict(active=str(active), parity_error=error, test=report['test']), indent=2))


if __name__ == '__main__':
    main()
