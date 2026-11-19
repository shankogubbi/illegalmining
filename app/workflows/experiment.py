"""Reproducible CPU transfer-learning experiments with a sealed image holdout.

Run from repository root: python -m app.workflows.experiment
Features are cached; subsequent head-training iterations reuse the cache.
"""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from app.core.dataset import AmazonDataset, get_val_transforms
from app.core.metrics import mining_metrics, decision_scores
from app.core.model import build_model
from app.utils.config import ALL_TAGS, MINING_TAG_INDICES, TRAIN_CSV, TRAIN_IMG_DIR, RESULTS_DIR, MODELS_DIR, CHECKPOINT_NAME

RUN_DIR = RESULTS_DIR / 'transfer_v1'
SEED = 20261108
SIZE = 160


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)


def make_split(df, seed=SEED):
    """Disjoint image split; all mining subtypes represented when supported."""
    strata = df.tags.map(lambda s: 2 if 'conventional_mine' in s.split() else
                         1 if 'artisinal_mine' in s.split() else 0).to_numpy()
    train, rest = train_test_split(np.arange(len(df)), test_size=.30,
                                  random_state=seed, stratify=strata)
    val, test = train_test_split(rest, test_size=.5, random_state=seed,
                                stratify=strata[rest])
    return {k: v.tolist() for k, v in [('train', train), ('validation', val), ('test', test)]}


def select_thresholds(labels, probabilities):
    """Validation-only class F2 thresholds with deterministic tie breaking."""
    thresholds = []
    for j in range(labels.shape[1]):
        best = (-1, .5)
        for threshold in np.linspace(.01, .99, 99):
            truth, pred = labels[:, j].astype(bool), probabilities[:, j] >= threshold
            tp = (truth & pred).sum()
            denom = 5 * tp + 4 * (truth & ~pred).sum() + (~truth & pred).sum()
            score = float(5 * tp / denom) if denom else 0
            best = max(best, (score, float(threshold)))
        thresholds.append(best[1])
    return thresholds


def report_metrics(y, p, thresholds):
    # Normalize scores so any mining-class threshold crossing maps to score .4.
    # Raw sigmoid outputs are retained separately.
    scaled = decision_scores(p, thresholds)
    return dict(mining=mining_metrics(y, scaled),
                mining_ranking_average_precision=mining_metrics(y, p)['average_precision'],
                subtype={ALL_TAGS[j]: {
                    'average_precision': float(average_precision_score(y[:, j], p[:, j])),
                    'positives': int(y[:, j].sum()),
                    'tp': int(((p[:, j] >= thresholds[j]) & (y[:, j] == 1)).sum()),
                    'fn': int(((p[:, j] < thresholds[j]) & (y[:, j] == 1)).sum()),
                    'fp': int(((p[:, j] >= thresholds[j]) & (y[:, j] == 0)).sum()),
                    'threshold': thresholds[j]}
                    for j in MINING_TAG_INDICES})


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if (RUN_DIR / 'final_metrics.json').exists():
        print('Completed run exists. Keep the sealed test result; use a new run directory for new experiments.', flush=True)
        return
    audit = json.loads((RESULTS_DIR / 'data_model_audit.json').read_text())
    excluded = {r['image'] for r in audit['unreadable']}
    df = pd.read_csv(TRAIN_CSV)
    df = df[~df.image_name.isin(excluded)].reset_index(drop=True)
    split = make_split(df)
    digest = hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest()
    manifest = dict(seed=SEED, csv_sha256=digest, image_size=SIZE, tags=ALL_TAGS,
                    split_type='stratified_random_image', excluded=sorted(excluded),
                    partitions={k: df.iloc[v].image_name.tolist() for k, v in split.items()})
    manifest_path = RUN_DIR / 'split_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Dataset or configuration changed; existing split must not be overwritten.')
    write_json(manifest_path, manifest)
    print('Fixed split:', {k: len(v) for k, v in split.items()}, flush=True)

    model = build_model('cpu', pretrained=True).eval()
    cache_key = hashlib.sha256((digest + str(SIZE) + str(SEED) + '|'.join(df.image_name)).encode()).hexdigest()
    cache_meta = RUN_DIR / 'feature_cache.json'
    cache_path = RUN_DIR / 'features.npy'
    if not (cache_meta.exists() and cache_path.exists() and
            json.loads(cache_meta.read_text()).get('key') == cache_key):
        features = np.lib.format.open_memmap(cache_path, mode='w+', dtype='float32', shape=(len(df), 1280))
        loader = DataLoader(AmazonDataset(df, TRAIN_IMG_DIR, get_val_transforms(SIZE)),
                            batch_size=32, num_workers=2)
        offset, started = 0, time.perf_counter()
        with torch.inference_mode():
            for x, _ in loader:
                z = model.net.avgpool(model.net.features(x)).flatten(1).numpy()
                features[offset:offset + len(z)] = z
                offset += len(z)
                if offset % 640 == 0 or offset == len(df):
                    features.flush()
                    status = dict(stage='features', complete=offset, total=len(df),
                                  elapsed_s=round(time.perf_counter() - started, 1))
                    write_json(RUN_DIR / 'progress.json', status)
                    print(status, flush=True)
        features.flush()
        write_json(cache_meta, dict(key=cache_key, backbone='ImageNet EfficientNet-B0', complete=True))
    features = np.load(cache_path, mmap_mode='r')
    labels = df.tags.str.get_dummies(sep=' ').reindex(columns=ALL_TAGS, fill_value=0).to_numpy(dtype=np.float32)
    tr, va, te = [np.array(split[k]) for k in ['train', 'validation', 'test']]
    # Fit normalization on training examples only.
    mean = features[tr].mean(axis=0)
    std = features[tr].std(axis=0).clip(.1)
    xtr = torch.from_numpy((features[tr] - mean) / std)
    xva = torch.from_numpy((features[va] - mean) / std)
    ytr = torch.from_numpy(labels[tr])
    train_counts = ytr.sum(0)
    ratio = (len(tr) - train_counts) / train_counts.clamp_min(1)
    best_score, best_state, best_config = -1., None, None
    trials = []
    for exponent, decay in [(0., .01), (.5, .01), (.5, .1)]:
        torch.manual_seed(SEED)
        head = torch.nn.Linear(1280, len(ALL_TAGS))
        torch.nn.init.zeros_(head.weight)
        weights = ratio.pow(exponent).clamp(1, 30)
        with torch.no_grad():
            head.bias.copy_(torch.log((train_counts * weights + 1) / (len(tr) - train_counts + 1)))
        optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=decay)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        trial_best, stale = -1., 0
        for epoch in range(1, 81):
            head.train()
            order = torch.randperm(len(tr))
            for batch in order.split(1024):
                optimizer.zero_grad()
                loss = criterion(head(xtr[batch]), ytr[batch])
                loss.backward()
                optimizer.step()
            if epoch % 2:
                continue
            head.eval()
            with torch.no_grad():
                pv = head(xva).sigmoid().numpy()
            score = float(np.mean([average_precision_score(labels[va, j], pv[:, j]) for j in MINING_TAG_INDICES]))
            row = dict(weight_exponent=exponent, decay=decay, epoch=epoch, validation_mining_macro_ap=score)
            trials.append(row)
            write_json(RUN_DIR / 'trials.json', trials)
            if score > best_score:
                best_score, best_state, best_config = score, {k: v.detach().clone() for k, v in head.state_dict().items()}, row
            if score > trial_best + 1e-4:
                trial_best, stale = score, 0
            else:
                stale += 2
            if epoch % 10 == 0:
                print(row, flush=True)
            if stale >= 14:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pv = head(xva).sigmoid().numpy()
    thresholds = select_thresholds(labels[va], pv)
    # Fold training normalization into the deployment classifier.
    with torch.no_grad():
        linear = model.net.classifier[1]
        linear.weight.copy_(head.weight / torch.from_numpy(std))
        linear.bias.copy_(head.bias - (linear.weight * torch.from_numpy(mean)).sum(1))
    candidate_path = MODELS_DIR / CHECKPOINT_NAME
    torch.save(dict(tags=ALL_TAGS, epoch=best_config['epoch'], state_dict=model.state_dict(),
                    image_size=SIZE, thresholds=thresholds,
                    metrics=dict(selection=best_config, validation=report_metrics(labels[va], pv, thresholds)),
                    provenance=dict(backbone='ImageNet EfficientNet-B0 frozen', csv_sha256=digest,
                                    split_manifest=str(manifest_path), seed=SEED)), candidate_path)
    # Test is inspected once, after all model and threshold decisions are fixed.
    with torch.no_grad():
        pt = head(torch.from_numpy((features[te] - mean) / std)).sigmoid().numpy()
    final = dict(selection=best_config, validation=report_metrics(labels[va], pv, thresholds),
                 test=report_metrics(labels[te], pt, thresholds), checkpoint=str(candidate_path),
                 evaluation='Previously unused stratified image holdout for this ImageNet-initialized candidate; not a geographic/time holdout.')
    np.savez_compressed(RUN_DIR / 'evaluation_predictions.npz', validation_probs=pv, validation_labels=labels[va],
                        test_probs=pt, test_labels=labels[te], thresholds=np.array(thresholds))
    write_json(RUN_DIR / 'final_metrics.json', final)
    write_json(RUN_DIR / 'progress.json', dict(stage='complete', checkpoint=str(candidate_path)))
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    main()
