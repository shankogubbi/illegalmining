"""Mining metrics: accuracy alone conceals rare-class failures."""
import numpy as np
from sklearn.metrics import average_precision_score
from app.utils.config import MINING_TAG_INDICES


def decision_scores(probabilities, thresholds, operating_point=0.4):
    """Map each class threshold to a common alert point, preserving endpoints.

    These are decision scores, not calibrated probabilities.
    """
    p, t = np.asarray(probabilities), np.asarray(thresholds)
    if not np.isfinite(t).all() or np.any((t <= 0) | (t >= 1)):
        raise ValueError('Decision thresholds must be finite and strictly between zero and one.')
    return np.where(p < t, p * operating_point / t,
                    operating_point + (p - t) * (1 - operating_point) / (1 - t))


def mining_metrics(y_true, y_prob, threshold=0.4):
    truth = np.asarray(y_true)[:, MINING_TAG_INDICES].max(axis=1).astype(bool)
    score = np.asarray(y_prob)[:, MINING_TAG_INDICES].max(axis=1)
    pred = score >= threshold
    tp = int((truth & pred).sum())
    fn = int((truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, positives=int(truth.sum()),
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else None,
                f2=5 * tp / (5 * tp + 4 * fn + fp) if 5 * tp + 4 * fn + fp else 0.0,
                average_precision=float(average_precision_score(truth, score)) if truth.any() else None,
                accuracy=float((pred == truth).mean()),
                always_negative_accuracy=float((~truth).mean()), threshold=threshold)
