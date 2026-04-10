#!/usr/bin/env bash
set -euo pipefail

# Run all ML commands from the ml/ directory so Poetry finds the project root.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer the Poetry install we verified in WSL, but fall back to PATH if needed.
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "Poetry was not found. Install it with pipx inside WSL first." >&2
  exit 1
fi

cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/wsl_ml.sh setup
  bash scripts/wsl_ml.sh gpu-check
  bash scripts/wsl_ml.sh baseline [baseline args...]
  bash scripts/wsl_ml.sh baseline-manifest [manifest args...]
  bash scripts/wsl_ml.sh train [training args...]
  bash scripts/wsl_ml.sh train-tiny [tiny training args...]
  bash scripts/wsl_ml.sh fit-search [fit search args...]
  bash scripts/wsl_ml.sh export [export args...]
  bash scripts/wsl_ml.sh pytest [pytest args...]
EOF
}

cmd="${1:-}"
if [[ -n "${cmd}" ]]; then
  shift
fi

case "${cmd}" in
  setup)
    # Install the project and dev dependencies in the active WSL Poetry env.
    exec "${POETRY_BIN}" install --with dev
    ;;
  gpu-check)
    # Confirm that the WSL Poetry env sees TensorFlow and the GPU device.
    exec "${POETRY_BIN}" run python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
    ;;
  baseline)
    # Run the classical Canny + Hough baseline on the labelled dataset.
    exec "${POETRY_BIN}" run python scripts/run_classical_baseline.py "$@"
    ;;
  baseline-manifest)
    # Run the classical baseline on an arbitrary image/value manifest.
    exec "${POETRY_BIN}" run python scripts/eval_classical_baseline_on_manifest.py "$@"
    ;;
  single-image)
    # Run the classical baseline on one specific camera frame.
    exec "${POETRY_BIN}" run python scripts/run_single_image_baseline.py "$@"
    ;;
  train)
    # Run the CNN training pipeline on the same WSL Poetry environment.
    exec "${POETRY_BIN}" run python scripts/run_training.py "$@"
    ;;
  train-tiny)
    # Run the compressed MobileNetV2 training preset for STM32N6 deployment.
    exec "${POETRY_BIN}" run python scripts/run_training.py \
      --model-family mobilenet_v2_tiny \
      --mobilenet-alpha 0.35 \
      --mobilenet-head-units 64 \
      --mobilenet-head-dropout 0.15 \
      "$@"
    ;;
  fit-search)
    # Probe candidate MobileNetV2 widths against the STM32N6 relocatable fit.
    exec bash scripts/run_mobilenetv2_fit_search.sh "$@"
    ;;
  export)
    # Export the calibrated CNN to TFLite + board metadata.
    exec "${POETRY_BIN}" run python scripts/export_board_artifacts.py "$@"
    ;;
  pytest)
    # Run the ML test suite from the same environment used for training.
    exec "${POETRY_BIN}" run pytest "$@"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
esac
