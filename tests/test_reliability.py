import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from app.core.trainer import Trainer
from app.core.metrics import mining_metrics
from app.utils.config import NUM_CLASSES, MINING_TAG_INDICES


def test_accuracy_can_hide_all_missed_mines():
    y = np.zeros((100, NUM_CLASSES))
    y[0, MINING_TAG_INDICES[0]] = 1
    metrics = mining_metrics(y, np.zeros_like(y))
    assert metrics['accuracy'] == 0.99
    assert metrics['recall'] == 0
    assert metrics['fn'] == 1
    assert metrics['f2'] == 0


def test_threshold_boundary_matches_alerts():
    y = np.zeros((2, NUM_CLASSES))
    y[0, MINING_TAG_INDICES[0]] = 1
    p = y * 0.4
    metrics = mining_metrics(y, p)
    assert metrics['tp'] == 1
    assert metrics['tn'] == 1
    assert metrics['recall'] == 1


def test_no_positives_does_not_claim_valid_recall():
    y = np.zeros((2, NUM_CLASSES))
    assert mining_metrics(y, y)['recall'] is None


def test_empty_evaluation_is_rejected():
    model = torch.nn.Linear(3, NUM_CLASSES)
    trainer = Trainer(model, [], [], epochs=1, log_fn=lambda _: None)
    with pytest.raises(ValueError, match='No batches'):
        trainer._run_epoch([], training=False)


def test_cancelled_validation_cannot_save_checkpoint_or_history(tmp_path, monkeypatch):
    import app.core.trainer as module
    monkeypatch.setattr(module, 'LOGS_DIR', tmp_path)
    saved = []
    monkeypatch.setattr(module, 'save_checkpoint', lambda *args: saved.append(args))
    trainer = Trainer(torch.nn.Linear(3, NUM_CLASSES), [1], [1], epochs=1,
                      log_fn=lambda _: None)
    def epoch(loader, training):
        if not training:
            trainer.stop()
        return {'loss': 0, 'f2': 1, 'mining_acc': 1}
    monkeypatch.setattr(trainer, '_run_epoch', epoch)
    assert trainer.fit() == []
    assert not saved
    assert not (tmp_path / 'training_history.json').exists()
