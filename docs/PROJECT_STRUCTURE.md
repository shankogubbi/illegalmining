# Project map

## Source code

| Location | Responsibility |
|---|---|
| `app/main.py` | Creates the Qt application and opens the main window |
| `app/cli.py` | Single-image command-line prediction and JSON/heatmap export |
| `app/core/dataset.py` | CSV labels, RGB image loading, transforms and train/validation loaders |
| `app/core/model.py` | EfficientNet-B0 classifier and checkpoint loading/saving |
| `app/core/trainer.py` | GUI training loop, metrics, early stopping and checkpoint selection |
| `app/core/predictor.py` | Checkpoint-specific inference, alert decisions and Grad-CAM |
| `app/core/metrics.py` | Mining metrics and threshold-adjusted decision scores |
| `app/core/downloader.py` | Kaggle download, extraction and filename readiness checks |
| `app/ui/main_window.py` | Desktop shell and tab wiring |
| `app/ui/setup_tab.py` | Download and environment controls |
| `app/ui/train_tab.py` | Training settings, worker and live curves |
| `app/ui/predict_tab.py` | Single-image analysis, batch scan and CSV export |
| `app/ui/dashboard_tab.py` | Training history and matching active-model evaluation |
| `app/ui/about_tab.py` | Embedded usage guide |
| `app/utils/config.py` | Root-relative paths, class ordering and defaults |
| `app/utils/logger.py` | File and console logging |
| `app/workflows/audit.py` | Image decode audit and optional diagnostic evaluation replay |
| `app/workflows/experiment.py` | Fixed 70/15/15 split, cached features and classifier comparisons |
| `app/workflows/activate_experiment.py` | Verification of the final model against saved predictions |
| `app/workflows/evaluation_report.py` | Evaluation Markdown, plots and recall interval |

The four older modules directly under `app/` remain as small forwarding entry points. Existing commands such as `python -m app.experiment` still work; new commands should use `app.workflows`.

## Data and models

`data/raw/train_v2.csv` maps `image_name` to space-separated tags. Active training images are in `data/raw/train-jpg/`. The original nested `planet/` download layout and test-image directories are retained as source material. They are not automatically substituted for the active labeled image directory. The source tag spelling is `artisinal_mine`.

| Artifact | Meaning |
|---|---|
| `models/best_model.pth` | Checkpoint loaded by the GUI and CLI by default |
| `archive/runs/previous_run_20260929/` | Historical training history; old checkpoint removed |

Model class order, image size and learned thresholds travel in the current checkpoint. Historical absolute provenance paths describe the original run; runtime paths are derived from configuration. Do not reorder `ALL_TAGS` independently of the checkpoint.

## Current experiment: `results/transfer_v1/`

| File | Contents |
|---|---|
| `REPORT.md` | Human-readable model results |
| `evaluation.png` | Precision-recall curve and confusion matrix |
| `final_metrics.json` | Validation/test metrics, final checkpoint path and active model hash |
| `split_manifest.json` | Exact image membership, seed, label order and CSV hash |
| `trials.json` | Validation scores for classifier configurations and epochs |
| `evaluation_predictions.npz` | Raw validation/test predictions, labels and chosen thresholds |
| `features.npy` | Cached ImageNet feature vectors; the largest experiment artifact |
| `feature_cache.json` | Cache identity and completion metadata |
| `progress.json` | Last extraction/training status |
| `environment.json` | Recorded Python and library versions |
| `deployment_check.json` | Export parity error and activated checkpoint hash |
| `smoke_prediction.json` | Saved prediction from an integration check |
| `prediction_heatmap.jpg` | Grad-CAM example from that check |

Other results include `data_model_audit.json`, `audit_predictions.npz`, demonstration images in `illegal_mining_examples/`, and user-created exports. The old audit replay describes its recorded checkpoint, not the new active model; use the current experiment report for the active model's held-out results.

## Configuration and generated files

`ROOT_DIR` comes from the source location, not the terminal's working directory. Paths for raw data, the final model, results and logs are centralized in `app/utils/config.py`.

`venv/`, `.idea/`, `.pytest_cache/` and `__pycache__/` are local environment/tool files. They are retained locally and ignored by Git. `assets/` is reserved for static application images or styles; evaluation charts and example heatmaps belong under results.
