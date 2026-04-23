#!/usr/bin/env bash
set -euo pipefail

# Evaluate the OBB localizer + scalar-reader cascade on the board probe set.
#
# The localizer operates on the board-style crop domain, and the scalar reader
# converts the OBB-derived tighter crop into the final temperature estimate.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/obb_scalar_eval_board_probe.log"
OBB_MODEL="${OBB_MODEL:-artifacts/training/mobilenetv2_obb_longterm/model.keras}"
SCALAR_MODEL="${SCALAR_MODEL:-artifacts/deployment/mobilenetv2_rectified_scalar_finetune_v2_int8/model_int8.tflite}"
MANIFEST="${MANIFEST:-data/board_rectified_probe_20260422.csv}"
OBB_CROP_SCALE="${OBB_CROP_SCALE:-1.20}"
MIN_CROP_SIZE="${MIN_CROP_SIZE:-48.0}"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[OBB-EVAL] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

cd "${REPO_ROOT}"
echo "[OBB-EVAL] Starting OBB + scalar board-probe evaluation." | tee "${LOG_FILE}"
echo "[OBB-EVAL] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# The evaluation is CPU-bound and should stay away from WSL GPU driver issues.
CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/eval_obb_scalar_on_manifest.py \
  --obb-model "${OBB_MODEL}" \
  --scalar-model "${SCALAR_MODEL}" \
  --manifest "${MANIFEST}" \
  --obb-crop-scale "${OBB_CROP_SCALE}" \
  --min-crop-size "${MIN_CROP_SIZE}" \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
