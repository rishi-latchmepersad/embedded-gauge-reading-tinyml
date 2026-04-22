#!/usr/bin/env bash
set -euo pipefail

# Retrain the rectifier with zoom-in augmentation so it handles close-up board
# framing as well as the far-away phone-photo distribution.
#
# Key differences from v3:
#   - augment_training enabled (now applies _augment_rectifier_image_and_box:
#     random zoom 0.40-1.0x + photometric jitter + label recomputation)
#   - warm-start from v3 (already converged on ellipse detection)
#   - full labelled dataset used (--use-all-labelled) for max dial-geometry coverage
#   - hard-case repeat reduced (board captures have dummy full-frame labels so
#     repeating them a lot adds noise, not signal)
#   - slightly more epochs to absorb the wider augmentation distribution

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectifier_zoom_aug_v4.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

cd "${ROOT_DIR}"
echo "[WRAPPER] Starting MobileNetV2 rectifier zoom-aug retrain v4." | tee "${LOG_FILE}"
echo "[WRAPPER] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_rectifier \
  --init-model artifacts/training/mobilenetv2_rectifier_hardcase_finetune_v3/model.keras \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 3 \
  --edge-focus-strength 1.0 \
  --crop-pad-ratio 0.50 \
  --device gpu \
  --no-gpu-memory-growth \
  --no-mobilenet-pretrained \
  --mobilenet-backbone-trainable \
  --mobilenet-warmup-epochs 0 \
  --learning-rate 2e-6 \
  --epochs 15 \
  --run-name mobilenetv2_rectifier_zoom_aug_v4 \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
