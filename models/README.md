# Model

`best_model.pth` is the single final selected checkpoint. Both the desktop application and command-line prediction load this file by default. Older and duplicate weights have been removed.

`python -m app.workflows.activate_experiment` verifies this model against the saved experiment predictions without modifying its weights. Evaluation results remain in `results/transfer_v1/`.
