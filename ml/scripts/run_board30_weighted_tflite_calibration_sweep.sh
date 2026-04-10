#!/usr/bin/env bash
set -euo pipefail

# Run the weighted calibration sweep on the board30-expanded hard-case set.
# This keeps the invocation simple and avoids shell quoting issues when
# targeting WSL from Windows.

MODEL_PATH="${1:-artifacts/deployment/scalar_full_finetune_from_best_board30_piecewise_calibrated_int8_board30_v2/model_int8.tflite}"
export MANIFEST_OVERRIDE="data/hard_cases_plus_board30.csv"
export WEIGHT_FACTORS="1.0 2.0 4.0 8.0 16.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/sweep_tflite_calibrations.sh" "${MODEL_PATH}"

