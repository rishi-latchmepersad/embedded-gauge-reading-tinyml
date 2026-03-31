#!/usr/bin/env bash
set -euo pipefail

# Launch the MobileNetV2 direction model training job in the WSL Poetry env.
# The job writes a dedicated log file and keeps running after this wrapper exits.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_direction_224_full.log"
PID_FILE="${LOG_DIR}/mobilenetv2_direction_224_full.pid"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "[WRAPPER] Starting MobileNetV2 direction training in the background."
echo "[WRAPPER] Log file: ${LOG_FILE}"

# Use nohup so the training process survives after the wrapper exits.
nohup "${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_direction \
  --device gpu \
  --run-name mobilenetv2_direction_224_full \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo $! > "${PID_FILE}"
echo "[WRAPPER] PID: $(cat "${PID_FILE}")"
echo "[WRAPPER] Training launched successfully."
