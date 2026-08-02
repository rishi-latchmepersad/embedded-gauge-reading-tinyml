# WSL ML Workspace

This directory is the WSL-side workspace for training, offline evaluation,
TFLite export, and model handoff. Use the Linux checkout, not the Windows
firmware checkout, for these commands:

```bash
cd /home/rishi_latchmepersad/Projects/embedded-gauge-reading-tinyml/ml
poetry install --with dev
```

Read the canonical pipeline contract before choosing a model:
[`docs/ai-memory/current-state/ml-pipeline.md`](../docs/ai-memory/current-state/ml-pipeline.md).

## Layout

- `src/embedded_gauge_reading_tinyml/`: importable Python package.
- `scripts/`: runnable preparation, training, evaluation, and export jobs.
- `tests/`: default collection-safe pytest suite.
- `data/`: source archives, manifests, and captured inputs.
- `artifacts/`: local model checkpoints and exports; ignored by Git.
- `workspace_manifest.toml`: required inputs and optional artifact contract.

## Workspace Check

Run this before a training or evaluation job:

```bash
bash scripts/run_wsl_guarded.sh poetry run python scripts/verify_ml_workspace.py
```

If a model artifact is required for a replay, request strict checking:

```bash
bash scripts/run_wsl_guarded.sh poetry run python scripts/verify_ml_workspace.py \
  --require-artifacts
```

Most model artifacts are intentionally not committed. A fresh checkout should
report them as optional and name the exact paths that must be produced or
handed off; it should not fail during Python import.

## Memory-Safe Jobs

Every training, conversion, evaluation, and packaging job must run through the
guarded launcher:

```bash
bash scripts/run_wsl_guarded.sh poetry run python scripts/<job>.py [args...]
```

The launcher enforces a single active job, a host-memory floor, bounded CPU
thread fan-out, and a 15,000 MiB TensorFlow GPU budget environment. Training
scripts that allocate TensorFlow devices must also configure the 15,000 MiB
logical GPU limit themselves. Do not launch a long job with bare `setsid`,
`nohup`, or direct `poetry run`.

The guarded launcher must be invoked from `ml/`, because Poetry resolves the
project from `ml/pyproject.toml`.

## Data Preparation

The LittleGood board archives may contain annotations without image bytes in a
fresh checkout. Repair them from the tracked raw captures before board-pool
training:

```bash
bash scripts/run_wsl_guarded.sh poetry run python scripts/repair_board_archive_images.py
```

Never train on `data/labelled/initial_temp_gauge/board_captures_2.zip`; it is a
duplicate of the `test_3.zip` holdout. See
`../docs/ai-memory/current-state/labelled-data.md` for the complete archive
inventory and split rules.

## Current Offline Candidate

The current WSL research candidate is the ellipse -> keypoint -> temperature
pipeline. Its exact model contracts, validation results, calibration constants,
and promotion status are in the current-state and model-update notes.

```bash
bash scripts/run_wsl_guarded.sh poetry run python \
  scripts/pipeline_ellipse_keypoint_temperature.py \
  --ellipse artifacts/ellipse_iter8_universal_wide_deep/model_int8.tflite \
  --keypoint artifacts/keypoint_unet_224g_wide_aug/model_int8.tflite \
  --images data/labelled/test_3.zip
```

The board production contract remains the Windows-validated OBB localizer
followed by `tip_focus_v18_int8_n6_npu` until the ellipse/keypoint candidate is
explicitly promoted through a Windows firmware handoff.

## Tests

Run the default collection-safe smoke suite from `ml/`:

```bash
bash scripts/run_wsl_guarded.sh poetry run pytest -q
```

The default suite intentionally covers only environment and contract smoke
tests. The older tests under `ml/tests/` and the `ml/scripts/test_*.py` files
are retained historical or asset-dependent suites; run one explicitly only
after supplying its required model and dataset paths.

## Handoff Rules

WSL exports belong under `ml/artifacts/`. Do not write WSL-relative paths into
firmware instructions. After export, copy the final raw xSPI2 blob and its
`c_info.json` and `network.csv` to the Windows firmware package, then perform
board packaging and flashing from Windows. The Windows package is the final
source of truth for signatures and flash addresses.
