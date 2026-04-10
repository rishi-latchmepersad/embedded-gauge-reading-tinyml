#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the geometry-first MobileNetV2 keypoint model.
#
# This keeps the learning problem aligned with the gauge geometry:
# predict the needle/keypoint structure with auxiliary heatmaps, then let the
# scalar gauge head learn from the shared spatial features.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_keypoint_geometry_first.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_keypoint_geometry_first.model.keras"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${BASE_MODEL_LOCAL}")"
cp -f "${BASE_MODEL_SRC}" "${BASE_MODEL_LOCAL}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting geometry-first MobileNetV2 keypoint fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_keypoint \
  --device gpu \
  --epochs 8 \
  --learning-rate 1e-6 \
  --mobilenet-warmup-epochs 1 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 8 \
  --edge-focus-strength 1.0 \
  --keypoint-heatmap-size 28 \
  --keypoint-heatmap-loss-weight 0.5 \
  --run-name mobilenetv2_keypoint_geometry_first \
  2>&1 | tee "${LOG_FILE}"
