# Training and model management

## Active model

The application loads `models/best_model.pth`. Only the latest selected checkpoint is retained. The current model uses ImageNet EfficientNet-B0 features at 160 pixels with a trained 17-tag logistic head. Training feature normalization is folded into the exported linear layer. Input size, tag order and validation-selected thresholds are stored in the checkpoint and used during prediction.

## Reproduce the transfer workflow

Run from the project root:

```powershell
.\venv\Scripts\python.exe -m app.workflows.audit
.\venv\Scripts\python.exe -m app.workflows.experiment
.\venv\Scripts\python.exe -m app.workflows.activate_experiment
.\venv\Scripts\python.exe -m app.workflows.evaluation_report
```

1. **Audit** decodes images and records unreadable-image exclusions.
2. **Experiment** fixes a stratified 70/15/15 train/validation/test split. It uses ImageNet weights rather than the old application checkpoint, extracts features once, and compares three classifier settings. Training-only statistics normalize features. Validation mining-subtype average precision selects the classifier; validation F2 selects per-tag thresholds. The test set is evaluated after these decisions.
3. **Verification** checks the final model against eight saved validation outputs and updates the report hash for the dashboard. The command retains its compatibility name `activate_experiment`, but does not copy, back up or rewrite weights.
4. **Reporting** renders saved metrics and predictions to Markdown and a figure. It does not train or activate a model.

ImageNet weights must already be cached or available for initial download. Application prediction needs only the saved checkpoint. Feature extraction is the main CPU cost; the existing full feature cache is retained. The latest recorded extraction on this machine took about 18 minutes; timing depends on hardware and load.

## Completed-run behavior

The current run lives in `results/transfer_v1/`; its sole checkpoint is `models/best_model.pth`. When `final_metrics.json` exists, the experiment exits without retraining. When creating an unfinished run, an existing split manifest must match the dataset and configuration. Incomplete feature caches are regenerated rather than trusted.

To create a genuinely new experiment, explicitly configure a new `RUN_DIR` in `app/workflows/experiment.py`, and update the verification/report workflow to target that run. This is currently source configuration, not an advertised command-line flag. Preserve the existing report, split and predictions. Choose models and thresholds using validation data; keep final test evaluation separate from iterative selection.

## GUI training path

The **Train** tab uses `app/core/trainer.py`, rather than the cached-feature experiment. It trains network parameters with BCEWithLogitsLoss, AdamW and a cosine learning-rate schedule. It selects checkpoints using validation mining average precision and saves GUI history to `logs/training_history.json`.

This path can overwrite the active checkpoint when it saves a better model within that run. Back up a checkpoint you want to retain before using this path. Cancelling an incomplete epoch does not save that partial epoch. The GUI history is separate from the transfer experiment's evaluation; it should not be used as the transfer model's training trace.

## Files to preserve

For prediction: the source, dependencies and `models/best_model.pth`.

For reproducible evaluation: the final checkpoint, `split_manifest.json`, `evaluation_predictions.npz`, `final_metrics.json`, `trials.json` and `environment.json`.

For faster follow-up feature-based experiments: keep `features.npy` and `feature_cache.json` as well. Their rows follow the readable CSV image order. Do not edit labels, excluded rows or image ordering and then reuse a cache as if it were unchanged.

The old `python -m app.experiment`, `app.audit`, `app.activate_experiment` and `app.evaluation_report` commands are compatibility wrappers for the organized workflow package.
