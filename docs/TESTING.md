# Testing guide

Run commands from the project root in PowerShell.

## 1. Automated software checks

```powershell
.\scripts\setup.bat
.\scripts\test.bat -q
```

Setup is needed once per environment. A successful test run reports passed checks and exits with code 0. The suite currently contains 20 tests. It uses generated images and small tensors; it does not require Kaggle credentials, the real dataset or pretrained downloads.

```powershell
# Detailed output
.\scripts\test.bat -v

# One file
.\venv\Scripts\python.exe -m pytest tests/test_reliability.py -v

# Select tests by name
.\scripts\test.bat -k threshold -v
```

| Test file | What it checks |
|---|---|
| `test_smoke.py` | Model shape/range, class indices, F2, transforms and unreadable-image filtering |
| `test_reliability.py` | Rare-class metrics, threshold boundaries, missing-positive recall, empty evaluation and cancelled validation |
| `test_experiment.py` | Reproducible disjoint splits, decision-score mapping, threshold selection, checkpoint preprocessing and classifier export equivalence |
| `conftest.py` | Sets a small CPU thread count for test execution |

## 2. Predict an image using the real model

```powershell
.\venv\Scripts\python.exe -m app.cli predict data\raw\train-jpg\train_7582.jpg --device cpu --output results\manual\prediction.json --heatmap results\manual\heatmap.jpg
```

Check that:

1. The command exits successfully.
2. `results/manual/prediction.json` contains the image, score, alert, tags and per-tag scores.
3. `results/manual/heatmap.jpg` opens as an image overlay.

Use your own image path to test new inputs. The bundled demonstration is a functional check; the experiment's held-out report measures model performance.

Omitting `--heatmap` skips gradient computation. `models/best_model.pth` is the only saved model and is selected by default. `--device cpu` works without a GPU. Use `--help` to list all options.

## 3. Desktop walkthrough

```powershell
.\run.bat
```

- Predict: browse/drop an image; confirm the original, tags, score and heatmap appear.
- Batch: select a small folder of images; confirm the progress finishes and CSV export contains one row per successful image.
- Dashboard: confirm the active model summary shows held-out precision/recall and subtype recall when its hash matches the saved report.
- Guide: confirm the help content opens.

The GUI Train tab starts a real training run. It is not needed to test an existing model.

## 4. Read evaluation results

Open `results/transfer_v1/REPORT.md`, `evaluation.png` and `final_metrics.json`. The report contains the selected operating-point metrics and the separate test split. Regenerate the report from saved arrays without retraining:

```powershell
.\venv\Scripts\python.exe -m app.workflows.evaluation_report
```

## 5. Dataset audit

```powershell
.\venv\Scripts\python.exe -m app.workflows.audit
```

This decodes every labeled image and takes longer than pytest. It writes `results/data_model_audit.json` with unreadable images, exact pixel duplicates, class counts and dimensions. Preserve prior reports if you need their history. The experiment reads its exclusion list from this file.

The optional `--evaluate` mode reconstructs a diagnostic random validation split and is not the saved experiment's held-out test. Use the experiment report for the recorded model evaluation.

## Troubleshooting

| Symptom | Action |
|---|---|
| `No module named pytest` or another dependency | Use the project venv and run `scripts/setup.bat` |
| `py` is not recognized | Follow the manual `python -m venv venv` setup in the README |
| Missing active checkpoint | Include the final `models/best_model.pth` file |
| Missing image or CSV | Check `data/raw/train-jpg/` and `data/raw/train_v2.csv`; use Setup to download |
| Cached features absent | The experiment regenerates them from the dataset and pretrained backbone |
| Dashboard only shows saved-result location | Check that active weights match the experiment; another GUI training run may have replaced them |
| Invalid copied virtual environment | Recreate a venv from requirements instead of copying an environment between computers |
| More error detail needed | Inspect `logs/app.log` and the command's error output |

To check imports and syntax separately: `.\venv\Scripts\python.exe -m compileall -q app tests`.
