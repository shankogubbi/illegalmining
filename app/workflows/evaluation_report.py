"""Render the completed transfer experiment without retuning on its test set."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from app.workflows.experiment import RUN_DIR
from app.utils.config import MINING_TAG_INDICES
from app.core.metrics import decision_scores


def wilson(successes, total):
    if not total:
        return (0., 1.)
    z = 1.96
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return center - half, center + half


def main():
    results = json.loads((RUN_DIR / 'final_metrics.json').read_text())
    arrays = np.load(RUN_DIR / 'evaluation_predictions.npz')
    metrics = results['test']['mining']
    labels, probs = arrays['test_labels'], arrays['test_probs']
    scores = decision_scores(probs, arrays['thresholds'])[:, MINING_TAG_INDICES].max(1)
    truth = labels[:, MINING_TAG_INDICES].max(1)
    precision, recall, _ = precision_recall_curve(truth, scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(recall, precision, linewidth=2)
    axes[0].scatter([metrics['recall']], [metrics['precision']], color='darkorange', s=65, label='Validation-selected operating point')
    axes[0].set(xlabel='Mining recall', ylabel='Mining precision', xlim=(0, 1), ylim=(0, 1.02), title='Held-out image precision–recall')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    matrix = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
    axes[1].imshow(np.log1p(matrix), cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        axes[1].text(j, i, str(value), ha='center', va='center', fontsize=18, color='white' if value > 1000 else 'black')
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['No alert', 'Mining alert'],
                yticklabels=['Non-mining', 'Mining'], xlabel='Prediction', ylabel='Label', title='Held-out image confusion matrix')
    fig.savefig(RUN_DIR / 'evaluation.png', dpi=160)
    plt.close(fig)
    lo, hi = wilson(metrics['tp'], metrics['tp'] + metrics['fn'])
    lines = ['# Mining detector — transfer experiment', '',
             f"Selected classifier: {json.dumps(results['selection'])}", '',
             '| Held-out metric | Result |', '|---|---:|',
             f"| Precision | {metrics['precision']:.2%} |",
             f"| Recall | {metrics['recall']:.2%} |",
             f"| Recall 95% Wilson interval (image level) | {lo:.2%}–{hi:.2%} |",
             f"| F2 | {metrics['f2']:.4f} |",
             f"| Average precision (threshold-adjusted ranking) | {metrics['average_precision']:.4f} |",
             f"| Accuracy | {metrics['accuracy']:.2%} |",
             f"| TP / FN / FP / TN | {metrics['tp']} / {metrics['fn']} / {metrics['fp']} / {metrics['tn']} |", '',
             '| Mining subtype | Positives | Detected | Missed | False alerts | Recall |', '|---|---:|---:|---:|---:|---:|']
    for tag, m in results['test']['subtype'].items():
        lines.append(f"| {tag} | {m['positives']} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tp']/m['positives']:.2%} |")
    lines += ['', '![Evaluation](evaluation.png)', '',
              'Method: ImageNet EfficientNet-B0 features at 160 pixels, training-only feature standardization, ',
              'weighted logistic heads, validation selection by mean mining-subtype average precision, ',
              'and validation-only per-tag F2 threshold selection. Test images were excluded from training, ',
              'normalization, checkpoint selection and threshold selection. The split is by image.', '',
              'Reproduction: `python -m app.experiment`. Fixed membership is in `split_manifest.json`; ',
              'candidate histories are in `trials.json`. Raw prediction arrays are retained for traceability.', '']
    (RUN_DIR / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(RUN_DIR / 'REPORT.md')


if __name__ == '__main__':
    main()
