#!/usr/bin/env bash
set -euo pipefail

# Sweep a few lightweight calibration families over a TFLite scalar model and
# print the raw plus calibrated errors for the hard-case manifests.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
MODEL_PATH="${1:-artifacts/deployment/scalar_full_finetune_from_best_board30_piecewise_calibrated_int8_board30_v2/model_int8.tflite}"
if [[ -n "${MANIFEST_OVERRIDE:-}" ]]; then
  MANIFESTS=("${MANIFEST_OVERRIDE}")
else
  MANIFESTS=(
    "data/hard_cases.csv"
    "data/hard_cases_plus_board30.csv"
  )
fi
KNOT_MODES=("all" "interior" "quantile")
KNOT_COUNTS=(4 8 12 16)
WEIGHT_PREFIX="${WEIGHT_PREFIX:-capture_2026-04-09_}"
WEIGHT_FACTORS=(${WEIGHT_FACTORS:-1.0 2.0 4.0 8.0})
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/tflite_calibration_sweep.log"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[SWEEP] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

{
  echo "[SWEEP] Model: ${MODEL_PATH}"
  for manifest in "${MANIFESTS[@]}"; do
    echo "[SWEEP] Manifest: ${manifest}"
    for knot_mode in "${KNOT_MODES[@]}"; do
      for weight_factor in "${WEIGHT_FACTORS[@]}"; do
        if [[ "${knot_mode}" == "quantile" ]]; then
          for knot_count in "${KNOT_COUNTS[@]}"; do
            echo "[SWEEP] knot_mode=${knot_mode} knot_count=${knot_count} weight_factor=${weight_factor}"
            "${POETRY_BIN}" run python -u scripts/eval_tflite_calibration.py \
              --model "${MODEL_PATH}" \
              --manifest "${manifest}" \
              --knot-mode "${knot_mode}" \
              --knot-count "${knot_count}" \
              --weight-prefix "${WEIGHT_PREFIX}" \
              --weight-factor "${weight_factor}"
          done
        else
          echo "[SWEEP] knot_mode=${knot_mode} weight_factor=${weight_factor}"
          "${POETRY_BIN}" run python -u scripts/eval_tflite_calibration.py \
            --model "${MODEL_PATH}" \
            --manifest "${manifest}" \
            --knot-mode "${knot_mode}" \
            --knot-count 8 \
            --weight-prefix "${WEIGHT_PREFIX}" \
            --weight-factor "${weight_factor}"
        fi
      done
    done
  done
} 2>&1 | tee "${LOG_FILE}"
