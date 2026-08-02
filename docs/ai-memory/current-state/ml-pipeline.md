# Current WSL/ML pipeline

Date: 2026-08-02
Status: current
Scope: WSL training, offline validation, and model handoff for STM32N657

## Source of truth

The checked-out WSL repository is the source of truth for Python code,
training data preparation, offline evaluation, and export metadata. The
Windows checkout is the source of truth for the flashed firmware package,
model signatures, xSPI2 slots, and board validation. Do not treat the
`firmware/` directory in this WSL checkout as the final board package.

## Two pipeline states

Keep these states separate when reporting results:

1. **Board production contract:** the flashed OBB localizer followed by
   `tip_focus_v18_int8_n6_npu`. The tip-focus model accepts `224x224` color
   input and emits `56x56` center/tip heatmaps plus `confidence` and
   `is_main_needle`. The Windows package beside the raw xSPI2 blob must retain
   `c_info.json` and `network.csv`.
2. **Current WSL research candidate:** the ellipse localizer followed by the
   center/tip keypoint model and deterministic temperature conversion. The
   latest validated candidate is described in
   `../model-updates/2026-08-01-firmware-handoff-ellipse-keypoint-temp.md` and
   `../model-updates/2026-08-01-live-board-validation.md`. It is not the board
   production contract until the Windows handoff and flash validation promote
   it.

The old scalar MobileNet, rectifier, and experimental geometry families remain
available for comparison, but none of their scores should be described as the
current production result without a dated promotion note.

The large set of older shell wrappers is documented as historical in
`ml/scripts/README.md`. Use the canonical commands below for new work instead
of copying an old wrapper's path or concurrency behavior.

## Canonical WSL commands

Run these commands from `ml/`. Long-running work must use the guard so the
host-memory floor, GPU budget, and single-job lock are active:

```bash
cd /home/rishi_latchmepersad/Projects/embedded-gauge-reading-tinyml/ml
poetry install --with dev
bash scripts/run_wsl_guarded.sh poetry run python scripts/verify_ml_workspace.py
bash scripts/run_wsl_guarded.sh poetry run pytest -q
```

Repair the annotation-only board archives before experiments that consume the
LittleGood board pool:

```bash
bash scripts/run_wsl_guarded.sh poetry run python scripts/repair_board_archive_images.py
```

The duplicate `initial_temp_gauge/board_captures_2.zip` source must remain out
of training because it duplicates `test_3.zip`. The established training list
and split rules are documented in `current-state/labelled-data.md`.

For the current offline ellipse/keypoint candidate, use the exact model paths
and calibration constants in the handoff note:

```bash
bash scripts/run_wsl_guarded.sh poetry run python \
  scripts/pipeline_ellipse_keypoint_temperature.py \
  --ellipse artifacts/ellipse_iter8_universal_wide_deep/model_int8.tflite \
  --keypoint artifacts/keypoint_unet_224g_wide_aug/model_int8.tflite \
  --images data/labelled/test_3.zip
```

Do not substitute `keypoint_unet_224g_stride2` for an N6 deployment package:
the 112x112 activation footprint exceeds the validated no-HyperRAM budget.

## Data and artifact contract

`ml/workspace_manifest.toml` lists source inputs and optional local model
artifacts. Run `scripts/verify_ml_workspace.py` before a job. Missing optional
artifacts are expected in a fresh clone, but the verifier reports the exact
paths and the producing workflow instead of failing later during test
collection.

Generated datasets and model artifacts are intentionally local and ignored.
Keep preparation scripts, manifests, reports, and checksums under version
control. Exported board blobs must be handed off to the Windows package with
their generated metadata; do not write a WSL-relative firmware path into a
Windows deployment instruction.

## Validation policy

Use `ml/tests/smoke/` for the collection-safe fresh-checkout suite. The other
tests under `ml/tests/`, plus files named `test_*.py` under `ml/scripts/`, are
legacy or asset-dependent probes and are run explicitly with their required
model/data paths; they are not part of the default pytest suite.
Every deployment candidate needs a Keras-vs-TFLite parity check before package
handoff, followed by board validation from the Windows workspace.
