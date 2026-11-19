import numpy as np
import pandas as pd
import pytest
from app.workflows.experiment import make_split, select_thresholds, report_metrics
from app.core.metrics import decision_scores
import torch
from PIL import Image
from app.utils.config import ALL_TAGS, MINING_TAG_INDICES


def test_split_is_disjoint_reproducible_and_preserves_subtypes():
    df = pd.DataFrame({'tags': ['primary'] * 900 + ['artisinal_mine'] * 70 + ['conventional_mine'] * 30})
    split = make_split(df)
    assert split == make_split(df)
    groups = [set(v) for v in split.values()]
    assert len(set.union(*groups)) == len(df)
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    for rows in split.values():
        assert set(df.iloc[rows].tags) == set(df.tags)


def test_decision_score_preserves_threshold_and_endpoints():
    t = np.array([.1, .7])
    assert np.allclose(decision_scores(t, t), .4)
    assert np.allclose(decision_scores(np.zeros(2), t), 0)
    assert np.allclose(decision_scores(np.ones(2), t), 1)
    assert np.all(decision_scores(t - .01, t) < .4)
    assert np.all(decision_scores(t + .01, t) > .4)
    with pytest.raises(ValueError):
        decision_scores([.2], [0])


def test_threshold_selection_separates_positive_and_negative():
    y = np.array([[0, 1], [1, 0]])
    p = np.array([[.01, .3], [.2, .02]])
    t = select_thresholds(y, p)
    assert np.array_equal(p >= t, y.astype(bool))


def test_predictor_applies_checkpoint_resize_and_thresholds(tmp_path, monkeypatch):
    import app.core.predictor as module
    class ConstantModel(torch.nn.Module):
        def forward(self, x):
            assert x.shape[-2:] == (160, 160)
            p = torch.full((len(x), len(ALL_TAGS)), .01)
            p[:, MINING_TAG_INDICES[1]] = .3
            return torch.logit(p)
    thresholds = [.5] * len(ALL_TAGS)
    thresholds[MINING_TAG_INDICES[1]] = .2
    calls = []
    def build(device, pretrained):
        calls.append(pretrained)
        return ConstantModel()
    monkeypatch.setattr(module, 'build_model', build)
    monkeypatch.setattr(module, 'load_checkpoint', lambda *args: {'image_size': 160, 'thresholds': thresholds})
    image = tmp_path / 'image.jpg'
    Image.new('RGB', (256, 256)).save(image)
    predictor = module.Predictor(device='cpu')
    result = predictor.predict(image, generate_cam=False)
    assert calls == [False]
    assert ALL_TAGS[MINING_TAG_INDICES[1]] in result.tags
    assert result.alert_level == 'WARNING'
    assert .4 < result.mining_score < .65


def test_folded_standardization_preserves_classifier_logits():
    torch.manual_seed(1)
    x = torch.randn(32, 1280)
    mean, std = torch.randn(1280), torch.rand(1280).clamp_min(.1)
    original = torch.nn.Linear(1280, len(ALL_TAGS))
    folded = torch.nn.Linear(1280, len(ALL_TAGS))
    with torch.no_grad():
        folded.weight.copy_(original.weight / std)
        folded.bias.copy_(original.bias - (folded.weight * mean).sum(1))
    torch.testing.assert_close(folded(x), original((x - mean) / std), atol=1e-5, rtol=1e-5)
