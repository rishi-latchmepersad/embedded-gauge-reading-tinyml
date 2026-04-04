# ML Workflow

The ML side of this project is intended to run in WSL on Ubuntu 24.04 with Poetry and GPU support.

This is the default workflow I would use for:
- baseline classical CV runs
- CNN training
- future export and evaluation scripts

## Recommended Environment

Use the WSL shell rather than native Windows Python when working in `ml/`.

The helper script below keeps the common commands as one-liners:

```sh
bash scripts/wsl_ml.sh help
bash scripts/wsl_ml.sh setup
bash scripts/wsl_ml.sh gpu-check
bash scripts/wsl_ml.sh baseline --max-samples 24
bash scripts/wsl_ml.sh single-image --image-path ../captured_images/capture_0006.png
bash scripts/wsl_ml.sh train
bash scripts/wsl_ml.sh fit-search
bash scripts/wsl_ml.sh export
bash scripts/wsl_ml.sh pytest tests/test_baseline_runner.py
```

The script lives at [ml/scripts/wsl_ml.sh](scripts/wsl_ml.sh).

```sh
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
~/.local/bin/poetry --version
```

If `~/.local/bin` is not already on your PATH, either call Poetry with the full path
or add this line to your WSL shell profile:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

## Setup

```sh
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
~/.local/bin/poetry install --with dev
```

## GPU Sanity Check

Verify that TensorFlow can see the GPU inside the Poetry environment:

```sh
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
~/.local/bin/poetry run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Baseline Run

Run the classical Canny + Hough baseline from WSL:

```sh
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
~/.local/bin/poetry run python scripts/run_classical_baseline.py
```

Use `--max-samples` for a faster smoke test.

## CNN Training

Run the CNN training job from the same WSL Poetry environment:

```sh
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
~/.local/bin/poetry run python scripts/run_training.py
```

Training now defaults to the strongest known MobileNetV2 preset for this
dataset: `224x224`, `epochs=40`, `batch_size=8`, `seed=21`, `strict_labels=False`,
and GPU mode when TensorFlow sees a GPU.

If you want the same preset with explicit logging and a tee'd artifact log, use
`bash scripts/run_mobilenetv2_full.sh`.

To probe the largest MobileNetV2 width that still fits the STM32N6 relocatable
memory pools, use:

```sh
bash scripts/wsl_ml.sh fit-search
```

Future export/evaluation scripts should import their defaults from
[`embedded_gauge_reading_tinyml.presets`](src/embedded_gauge_reading_tinyml/presets.py)
so they stay aligned with the same `224x224` MobileNetV2 baseline.

## Board Export

When you want a deployable artifact for the STM32 side, export the calibrated
scalar model to int8 TFLite plus a metadata sidecar:

```sh
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
~/.local/bin/poetry run python scripts/export_board_artifacts.py
```

The default export reads the calibrated full-finetune model and writes the
board bundle under `artifacts/deployment/scalar_full_finetune_from_best_calibrated_int8/`.
The metadata file records the `224x224` board input size and the TFLite tensor
quantization parameters the MCU runtime will need.

The packaging step also refreshes the canonical raw blob at
`../st_ai_output/atonbuf.xSPI2.raw`. That is the single file you should copy to
the SD card root for board boot testing.

If you prefer the WSL helper:

```sh
bash scripts/wsl_ml.sh export
```

## Export and Evaluation

Keep any future export/evaluation scripts in `ml/scripts/` and run them from the
same WSL Poetry environment so they share the same TensorFlow and CUDA setup.

That keeps the baseline, training, export, and evaluation workflows aligned on
the same runtime.

## Output Layout

Generated model artifacts continue to live under `ml/artifacts/`.

The repository-root `st_ai_output/` directory is reserved for the ST Edge AI
generator workspaces and packaged outputs that accompany the STM32N6 flow:

- `st_ai_output/packages/`
- `st_ai_output/gauge_scalar_c_info.json`
- `st_ai_output/gauge_scalar_clean_c_info.json`

That keeps the training/deployment artifacts separate from the ST Edge AI
workspace files while still giving the package outputs a single home.
