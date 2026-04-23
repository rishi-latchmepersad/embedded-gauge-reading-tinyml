#!/usr/bin/env bash
set -euo pipefail

# Train the MobileNetV2 OBB localizer on the labeled dataset only.
#
# This run intentionally stays on the CVAT-labeled pool:
# - the model learns ellipse-style oriented-box parameters for the dial
# - validation and test are held out by fraction, not by board manifests
# - the board-style rectifier/scalar path remains the deployment benchmark
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_obb_longterm.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/mobilenetv2_geometry_uncertainty_full_range/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_obb_longterm.model.keras"

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

echo "[WRAPPER] Starting long-term MobileNetV2 OBB localizer fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_obb \
  --device gpu \
  --no-gpu-memory-growth \
  --epochs 20 \
  --learning-rate 5e-7 \
  --mobilenet-warmup-epochs 4 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --val-fraction 0.15 \
  --test-fraction 0.15 \
  --run-name mobilenetv2_obb_longterm \
  2>&1 | tee "${LOG_FILE}"
