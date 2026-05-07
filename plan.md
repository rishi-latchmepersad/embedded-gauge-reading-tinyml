# PLAN For GLM: Canonical Data + CNN Recovery

## Objective
Train a CNN that generalizes on gauge readings by using the full labeled dataset (CVAT + hard cases + board captures), with deterministic splits and slice-aware evaluation.

## Why We Are Doing This
- Recent runs used too little data (`~409` labeled rows, `~236` train rows after split).
- Overfitting was severe (train MAE dropped while val/test got worse).
- We must fix data assembly and evaluation before architecture changes.

## Success Criteria
1. Canonical manifest includes all valid labeled rows from agreed sources.
2. Deterministic splits are saved and reused.
3. Baseline retrain uses canonical splits and reports slice metrics.
4. `board_capture` MAE and `hard_case` MAE improve vs previous run.
5. No critical regressions on core slice.

## Ground Rules
1. Always restart WSL before running long jobs:
```bash
wsl --shutdown
```
2. Run ML commands in WSL, from `ml/`, with `.venv` active.
3. Keep edits small and testable.
4. Store temporary files only under `tmp/`.

## Phase 1: Canonical Manifest

### Inputs
- `ml/data/unified_manifest_with_crops_v2.csv`
- `ml/data/full_labelled_plus_board30_valid_with_new5.csv`
- `ml/data/hard_cases_plus_board30_valid_with_new6.csv`
- `ml/data/new_labelled_captures4.csv`
- `ml/data/all_captured_images_manifest.csv` (rows with non-empty labels only)

### Output Files
- `ml/data/canonical_manifest_v1.csv`
- `ml/data/canonical_manifest_conflicts_v1.csv`
- `ml/data/canonical_manifest_v1_summary.json`

### Required Canonical Columns
- `image_path`
- `value`
- `source_tag`
- `hardness_tag`
- `crop_x_min`
- `crop_y_min`
- `crop_x_max`
- `crop_y_max`
- `origin_manifest`

### Normalization Rules
1. Normalize path separators and casing policy consistently.
2. Resolve each path against repo root and verify file exists.
3. Parse `value` as float and enforce allowed range for this gauge.
4. Drop rows with missing file or invalid label.

### Deduplication Rules
1. Key: normalized absolute `image_path`.
2. If duplicate labels differ by `<= 1.0`, keep row based on priority:
   - `hard_cases_plus_board30_valid_with_new6.csv`
   - `new_labelled_captures4.csv`
   - `full_labelled_plus_board30_valid_with_new5.csv`
   - `unified_manifest_with_crops_v2.csv`
   - `all_captured_images_manifest.csv`
3. If duplicate labels differ by `> 1.0`, write all versions to `canonical_manifest_conflicts_v1.csv` and exclude from training.

## Phase 2: Tests

### Add/Update Pytest Coverage
1. Path normalization.
2. Invalid/missing label filtering.
3. Missing-file filtering.
4. Deduplication priority ordering.
5. Conflict detection threshold (`abs(delta) > 1.0`).
6. Output schema validation.

### Test Command
```bash
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
source .venv/bin/activate
pytest -q
```

## Phase 3: Deterministic Split Build

### Output Files
- `ml/data/splits/canonical_split_v1_train.csv`
- `ml/data/splits/canonical_split_v1_val.csv`
- `ml/data/splits/canonical_split_v1_test.csv`
- `ml/data/splits/canonical_split_v1_metadata.json`

### Split Requirements
1. Ratio: 70 / 15 / 15.
2. Stratify by 5C value bins.
3. Preserve `source_tag` proportions as much as possible.
4. Ensure `board_capture` and `hard_case` are present in val/test.
5. Deterministic seed; save seed and parameters in metadata.

## Phase 4: Baseline Retrain On Canonical Splits

### Training Configuration (first recovery run)
1. MobileNetV2 with reduced capacity (`alpha=0.35` first).
2. Backbone frozen for full run (`backbone_trainable=False`).
3. Dropout `0.4` to `0.5`.
4. Huber loss.
5. Weighted sampling or loss weighting by value-bin frequency.
6. Use checkpointing with best-weight restore by `val_mae`.

### Runtime Requirements
1. Ensure WSL GPU is available with `nvidia-smi`.
2. Ensure TensorFlow runtime paths include `.venv` NVIDIA libs and `/usr/lib/wsl/lib`.
3. Keep strict shell mode in scripts: `set -euo pipefail`.

### Command Pattern
Use your existing training script entrypoint and pass split manifests explicitly. Save logs under:
- `ml/artifacts/training/canonical_v1_baseline/`

## Phase 5: Evaluation And Reporting

### Required Metrics
1. Overall MAE / RMSE.
2. MAE by `source_tag`.
3. MAE by value bin.
4. MAE for `hardness_tag == hard_case`.
5. Top-30 worst predictions table:
   - `image_path`, `true_value`, `pred_value`, `abs_error`, `source_tag`.

### Report Output
- `ml/artifacts/reports/canonical_v1_baseline_report.md`

## Phase 6: Controlled Ablation (if baseline improves)
Run these four experiments only:
1. `A`: head-only, alpha=0.35, no extra aug.
2. `B`: A + stronger photometric aug.
3. `C`: A + hard-case/value-bin weighting.
4. `D`: A + aug + weighting.

Select winner by:
1. Lowest val MAE.
2. Best `board_capture` and `hard_case` MAE.
3. No major core regression.

## Failure Protocol
If command shows no output or appears stuck:
1. `wsl --shutdown`
2. Re-run exact command.

If TensorFlow cannot see GPU:
1. `nvidia-smi` in WSL.
2. Confirm TensorFlow GPU probe in `.venv`.
3. Confirm `LD_LIBRARY_PATH` includes:
   - `/usr/lib/wsl/lib`
   - `.venv/lib/python*/site-packages/nvidia/*/lib`

## Execution Checklist For GLM
1. Build canonical manifest + conflicts + summary.
2. Add/green tests.
3. Build deterministic splits + metadata.
4. Run baseline retrain on canonical splits.
5. Generate full slice report.
6. Run 4 ablations only if baseline is stable.
7. Propose next action using measured deltas, not intuition.

