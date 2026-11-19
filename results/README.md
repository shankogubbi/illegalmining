# Results

- `transfer_v1/REPORT.md`: current model evaluation.
- `transfer_v1/`: fixed split, classifier trials, predictions, metrics, charts and feature cache.
- `data_model_audit.json`: full image integrity audit and any recorded diagnostic replay.
- `audit_predictions.npz`: raw predictions from the recorded diagnostic replay.
- `illegal_mining_examples/`: existing demonstration images and overlays.
- `manual/`: command-line prediction outputs created by README examples.

GUI batch exports are saved to the user-selected path, defaulting to this directory. Historical training records are in `archive/runs/`. Keep evaluation reports and their raw prediction/split artifacts together. Generated results are excluded from Git; transfer them separately when sharing a runnable/evaluated project.
