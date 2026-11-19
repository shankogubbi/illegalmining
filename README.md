# Illegal Mining Detector

Desktop application and command-line tools for detecting mining activity in satellite images using EfficientNet-B0 and Grad-CAM.

## Start here

Run these commands in **PowerShell from the project folder**:

```powershell
# Install application and test dependencies (first setup)
.\scripts\setup.bat

# Open the desktop application
.\run.bat

# Run the automated checks
.\scripts\test.bat -q
```

The existing environment and trained model are ready to use on this computer. Setup is needed when preparing a new environment or installing missing dependencies. The launch command does not reinstall packages every time.

If the Windows `py` launcher is unavailable:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m app.main
```

Always use the project's `venv` Python for commands below. No PowerShell activation-policy changes are required. Python 3.13 is used in the current environment; package versions are recorded in `results/transfer_v1/environment.json`. On Linux/macOS, create a virtual environment and use `venv/bin/python` in place of the Windows executable.

## Where everything lives

```text
illegal-mining-detector/
|-- app/                         Application source
|   |-- main.py                  Desktop entry point
|   |-- cli.py                   Command-line image prediction
|   |-- core/                    Dataset, model, training, prediction, metrics
|   |-- ui/                      Qt windows and tabs
|   |-- utils/                   Central paths/settings and logging
|   |-- workflows/               Audit, experiment, activation, report commands
|   `-- *.py                     Older workflow command compatibility wrappers
|-- tests/                       Automated software and regression checks
|-- scripts/                     Windows setup and test launchers
|-- docs/                        Detailed project, testing and training guides
|-- data/
|   `-- raw/                     Downloaded images and labels
|-- models/
|   `-- best_model.pth           Single final model used everywhere
|-- results/
|   |-- transfer_v1/             Current evaluation, split, cache and figures
|   |-- illegal_mining_examples/ Existing demonstration images
|   `-- ...                      Dataset audit and exported predictions
|-- logs/                        Runtime log and GUI training history
|-- archive/runs/                Historical training records
|-- assets/                      Application static assets
|-- requirements.txt             Application dependencies
|-- requirements-dev.txt         Application dependencies plus pytest
|-- .gitignore                   Local data, weights, caches and credentials
`-- run.bat                      Desktop launcher
```

Source-code responsibilities and artifact details: [project map](docs/PROJECT_STRUCTURE.md).

## Test an image

### Desktop

1. Start `run.bat` and open **Predict**.
2. Browse to an image or drag it into the image panel.
3. Review the mining score, active tags and Grad-CAM overlay.
4. Use **Batch Scan Folder** to process multiple images, then **Export CSV** to save the results.
5. Open **Dashboard** to inspect the active model's evaluation.

The active model is already trained; opening Predict does not require training again. The **Setup** tab handles dataset downloads when training data are needed.

### Command line

This example uses an image already present in the dataset:

```powershell
.\venv\Scripts\python.exe -m app.cli predict data\raw\train-jpg\train_7582.jpg --device cpu --output results\manual\prediction.json --heatmap results\manual\heatmap.jpg
```

Replace the image path with your own file. Quote paths containing spaces. Omit `--heatmap` for a faster prediction without Grad-CAM.

```powershell
.\venv\Scripts\python.exe -m app.cli predict --help
```

Prediction JSON contains the image path, mining score, alert level, active tags and raw tag scores. The heatmap is a separate image. Supported desktop inputs include JPEG, PNG, TIFF and BMP; the model uses RGB input. Thresholds and input size are loaded from the checkpoint.

Mining scores are model decision scores rather than calibrated probabilities. The dataset labels identify mining appearance; they do not encode legal status.

## Run tests

```powershell
.\scripts\test.bat -q
.\scripts\test.bat -v

# Run a specific group of checks
.\venv\Scripts\python.exe -m pytest tests/test_experiment.py -v
```

The suite covers model outputs, image transforms, filtering unreadable images, mining metrics, split independence, threshold handling, exported classifier equivalence and cancellation behavior. Tests use small generated inputs and do not retrain the model or scan the full dataset.

See [testing guide](docs/TESTING.md) for manual checks, outputs and troubleshooting.

## Model evaluation

The active checkpoint was selected from three transfer-learning configurations using validation mining-subtype average precision. Its separate 6,072-image test split produced:

| Metric | Result |
|---|---:|
| Mining recall | 80.30% |
| Mining precision | 66.25% |
| Mining F2 | 0.7703 |
| Mining images detected | 53 / 66 |
| False mining alerts | 27 |
| Artisanal-tag recall | 82.69% |
| Conventional-tag recall | 53.33% |

This evaluation uses a fixed split by image. Software tests and model evaluation serve different purposes; passing pytest checks application behavior, while the report measures detection performance.

- [Full evaluation report](results/transfer_v1/REPORT.md)
- [Precision-recall curve and confusion matrix](results/transfer_v1/evaluation.png)
- [Original data/model review](docs/RELIABILITY_REVIEW.md)

## Training and maintenance commands

```powershell
# Full labeled-image integrity audit; this decodes the dataset
.\venv\Scripts\python.exe -m app.workflows.audit

# Train the transfer experiment, reusing its feature cache when available
.\venv\Scripts\python.exe -m app.workflows.experiment

# Verify the final model against saved evaluation outputs
.\venv\Scripts\python.exe -m app.workflows.activate_experiment

# Regenerate evaluation Markdown and charts from saved predictions
.\venv\Scripts\python.exe -m app.workflows.evaluation_report
```

The completed experiment is protected: rerunning its training command reports that the run already exists. It does not silently retrain or overwrite the finished test result. The verification command leaves model weights unchanged; report generation only regenerates report files. The GUI **Train** tab is a separate full-network training path and can replace the active checkpoint when a new best model is saved.

See [training guide](docs/TRAINING.md) before starting a new training run. CPU execution is supported; CUDA is selected automatically when available. Initial feature extraction is the expensive step; the existing cache is retained.

## Paths, logs and sharing

- All application paths are derived from `app/utils/config.py`; moving the entire project folder does not require editing source paths.
- Runtime messages: `logs/app.log`. GUI training history: `logs/training_history.json`.
- Current model evaluation: `results/transfer_v1/final_metrics.json`; historical GUI training charts are separate from this report.
- Keep the active checkpoint with the project when sharing a runnable copy. Large data, weights, results, local environments and credentials are excluded from Git by `.gitignore` and must be transferred separately if needed.
- Keep experiment split files, predictions and metrics together when sharing results. Keep the feature cache if you want fast follow-up head training.
