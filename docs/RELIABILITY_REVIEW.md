# Reliability review — 8 November 2026

Follow-up: the selected transfer experiment is now active. Its held-out image results are 80.30% recall, 66.25% precision and F2 0.7703; see [the new experiment report](../results/transfer_v1/REPORT.md). Only the latest selected checkpoint is retained; the original checkpoints were removed. The following review records the original audit findings.

## Verdict

The existing model is a research prototype. It is not validated for reliable field deployment, and there is no defensible basis to call it the best model. No model can guarantee perfect predictions. The useful goal is a measured error rate, with uncertainty, on representative independent data and a human-review process.

The current data are useful for learning Amazon land-cover and mining appearance. They are insufficient for establishing illegal mining, generalization to a different region/sensor, or operational reliability.

## Evidence from this workspace

* Labels: 40,479 images, 17 tags, 339 artisanal-mine labels and 100 conventional-mine labels. There are 435 distinct mining-positive images (four have both tags).
* Always predicting no mining yields **98.9254% accuracy**. Thus a 99% headline accuracy would provide almost no evidence of useful detection.
* The CSV contains only `image_name` and `tags`. It has no legal-status labels, site IDs, coordinates, capture dates, permits, or adjudicated illegality records. Geographic/temporal independence cannot be established from this CSV.
* The active checkpoint is epoch 2 with stored macro class F2 = 0.5386342669735924. This is a score, not 53.86% accuracy. Its original split and training settings were not saved.
* The history file describes epoch 1, zero validation loss and zero validation metrics. It does not describe the active checkpoint. The old cancellation path could produce and save empty validation metrics; that is a plausible failure mode, not proven provenance for this historical file.
* Six images in `results/illegal_mining_examples` were selected from training data using model scores. They demonstrate output appearance; they cannot establish generalization. Their mining labels do not prove illegality.
* CUDA is unavailable in the installed PyTorch environment. A historical epoch took about 29,659 seconds (8.24 hours). Published README timing estimates do not describe this machine.

Machine-readable image integrity and diagnostic model results are in `results/data_model_audit.json`. Saved prediction arrays and replay image names are in `results/audit_predictions.npz`.

### Completed audit results

40,478 of 40,479 images decode successfully. The unreadable image is `train_38164.jpg`. All readable images are 256 by 256 RGB after conversion. No duplicate image names, exact pixel duplicates or unknown tags were found. There are 73 rows with repeated tags; multi-hot encoding already ignores repetitions, and audit/distribution counts now count each tag once per image.

On the 6,072-image diagnostic replay at threshold 0.4:

| Measure | Result |
|---|---:|
| Mining positives | 57 |
| Correct mining alerts | 27 |
| Missed mining images | 30 |
| False mining alerts | 9 |
| Mining precision | 75.00% |
| Mining recall | 47.37% |
| Mining F2 | 0.5114 |
| Mining average precision | 0.6229 |
| Binary accuracy | 99.36% |
| Always-negative accuracy | 99.06% |

Artisanal-tag recall is 60.98% (25/41). Conventional-tag recall is **0% (0/16)** at the current threshold; its average precision is 0.0842. The combined mining alert flags 27 positive images, while the subtype-specific artisanal decision correctly identifies 25 artisanal positives; an image can receive a mining alert from the wrong subtype. This demonstrates why combined accuracy and selected positive examples cannot establish reliability. These figures remain diagnostic, with unknown training overlap.

## What the diagnostic evaluation means

The audit decodes every labeled image and checks exact pixel duplicates, duplicate names, unknown tags, dimensions and class counts. It reconstructs the previous default 85/15 random split over currently readable images and evaluates the saved model. It reports mining TP, FP, FN, TN, precision, recall, F2, average precision and the always-negative baseline, including separate results for each mining tag.

**This replay is not an independent test.** We cannot prove that these images were withheld from the legacy model. It does not assess near duplicates, spatially adjacent tiles, temporal leakage, annotation correctness or deployment distribution shift. Exact duplicate checking does not establish site independence. Threshold 0.4 is a historical heuristic, not a validated operating point. The maximum of two sigmoid outputs is an uncalibrated ranking score, not a calibrated probability of either mining or illegality.

## Corrections implemented

* Added reproducible image audit and diagnostic replay command: `python -m app.audit --evaluate`.
* Added mining confusion counts, precision, recall, F2, average precision and trivial-baseline accuracy to future epoch histories/checkpoints.
* Changed future checkpoint selection and early stopping to mining average precision instead of overall macro class F2. This emphasizes mining ranking; it does not itself guarantee useful recall or calibration.
* Reject empty evaluation and validation with no mining-positive examples. Cancelled or partial epochs cannot save a model or overwrite history with an empty run.
* Recorded image membership and tag order in future checkpoints; mismatched tag orders are rejected when loading.
* Preserved rare mining label combinations in random splits when sufficient support exists. This is still an internal development split, not a geographic test.
* Removed random crop/erasing because a small labeled mine could disappear while its positive label remained. Retained flips, mild rotation and color changes.
* Kept incomplete batches, preventing zero training batches on small datasets. Loss now weights samples rather than batches equally.
* Reject missing/unknown labels and duplicate image names. Dataset readiness now checks all labeled filenames instead of accepting 51 images.
* Inference loads checkpoint weights without downloading ImageNet weights first. Requested pretrained training now fails visibly if weights cannot load, instead of quietly changing to random initialization.
* Added visible limitations to prediction UI and guide, and replaced the reassuring low-score banner. The legacy `CLEAR` API/CSV code remains for compatibility and means only below the heuristic score threshold.

The original model was retained during the initial review and later replaced by the selected transfer model. These corrections do not retrospectively validate or improve the saved model.

## Required work before selecting a final model

1. Define deployment region, sensor, resolution, acceptable missed-mine rate and review workload. Define the target as visible mining unless verified legal records support a separate legality decision.
2. Obtain representative, expert-reviewed positives and hard negatives: construction, quarries, bare riverbanks, logging, agriculture, roads, cloud/haze, seasonal changes and small/abandoned mines. Audit annotation disagreement and ambiguous cases. Include representative data from the actual intended deployment region.
3. Obtain site IDs, dates and coordinates. Deduplicate before splitting, and reserve whole sites/regions and later dates as an untouched final test. Keep calibration/threshold tuning separate from final testing. Keep final-test data out of augmentation, pretraining adaptation and model selection.
4. Compare the corrected EfficientNet-B0 baseline with stronger candidates under identical splits, tuning budgets and mining metrics. Tune training-set class weighting or sampling, resolution, augmentation and learning rates. Select by validation evidence, compute cost and robustness; do not assume a larger architecture wins.
5. Choose an operating threshold from a validation precision-recall curve based on the required recall and false-alert budget. Evaluate each mining subtype and difficult condition. Calibrate scores using separate representative data if probabilities are needed.
6. Run the untouched site/time-separated test once after selection. Report precision/recall confidence intervals with site-aware resampling, false alerts per area/site, missed mines, subtype support and comparisons with simple baselines. Small positive counts imply wide uncertainty even when all examples are detected.
7. Deploy first in shadow mode with expert review. Route uncertain/cloudy/unsupported imagery to review, monitor drift and collect verified errors for retraining. Establish release thresholds and rollback criteria from the actual operational requirements.

No full retraining was started: the current data/evaluation gaps prevent an honest final reliability claim, and repeated training against an unverified split would not resolve them.

## Verification

All 15 tests passed, including rare-class metric failures, threshold boundaries, empty evaluation and cancelled-validation checkpoint protection. The main Qt window also constructed successfully in offscreen mode. Compilation succeeded. These are software checks; they do not establish model reliability.

Dataset and metric context: [official competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/overview), [participant report describing sample-averaged F2](https://cs231n.stanford.edu/reports/2017/pdfs/914.pdf). The application's macro class F2 must not be compared directly to sample-averaged competition scores.
