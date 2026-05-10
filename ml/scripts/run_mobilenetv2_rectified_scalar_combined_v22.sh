#!/usr/bin/env bash
set -euo pipefail

# Run the combined real + synthetic rectified scalar experiment from the ml/ tree.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer the Poetry install verified in WSL, but fall back to PATH if needed.
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "Poetry was not found. Install it with pipx inside WSL first." >&2
  exit 1
fi

cd "${ROOT_DIR}"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

TMP_DIR="../tmp"
SYNTH_DIR="${TMP_DIR}/synth_gauge_all_v22"
SYNTH_MANIFEST="${SYNTH_DIR}/manifest.csv"
COMBINED_MANIFEST="${TMP_DIR}/combined_scalar_manifest_v22.csv"
INIT_WEIGHTS="${TMP_DIR}/mobilenetv2_rectified_scalar_strict_v5.weights.h5"

REAL_MANIFEST="data/full_scalar_manifest_v1.csv"
REAL_BOXES_CSV="data/rectified_crop_boxes_full_scalar_v1.csv"
OUTPUT_DIR="artifacts/training/mobilenetv2_rectified_scalar_combined_v22"
HARD_CASE_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"

LOG_DIR="artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_combined_v22.log"

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

echo "[WRAPPER] Starting combined real + synthetic rectified scalar v22." | tee "${LOG_FILE}"
echo "[WRAPPER] Synthetic dir:     ${SYNTH_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Combined manifest:  ${COMBINED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Real manifest:      ${REAL_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Real boxes CSV:     ${REAL_BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init weights:       ${INIT_WEIGHTS}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir:         ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Generating corrected synthetic gauge renders..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/generate_synthetic_gauge_dataset.py \
  --output-dir "${SYNTH_DIR}" \
  --manifest-path "${SYNTH_MANIFEST}" \
  --num-samples 1600 \
  --image-size 224 \
  --seed 22 \
  --profile standard \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building the weighted combined manifest..." | tee -a "${LOG_FILE}"
REAL_MANIFEST_PATH="${REAL_MANIFEST}" \
SYNTH_MANIFEST_PATH="${SYNTH_MANIFEST}" \
COMBINED_MANIFEST_PATH="${COMBINED_MANIFEST}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Combine the real and synthetic manifests with light synthetic weighting."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def _copy_rows(source_path: Path, writer: csv.DictWriter, *, sample_weight: float | None) -> int:
    """Copy rows into the combined manifest and optionally override the weight."""
    count = 0
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {source_path}")
        for row in reader:
            image_path = str(row.get("image_path", "")).strip()
            value = str(row.get("value", "")).strip()
            if not image_path or not value:
                continue
            output_row = {
                "image_path": image_path,
                "value": value,
                "sample_weight": (
                    str(sample_weight)
                    if sample_weight is not None
                    else str(float(row.get("sample_weight", 1.0)))
                ),
            }
            writer.writerow(output_row)
            count += 1
    return count


real_manifest = Path(os.environ["REAL_MANIFEST_PATH"])
synth_manifest = Path(os.environ["SYNTH_MANIFEST_PATH"])
combined_manifest = Path(os.environ["COMBINED_MANIFEST_PATH"])
combined_manifest.parent.mkdir(parents=True, exist_ok=True)

with combined_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "sample_weight"])
    writer.writeheader()
    real_count = _copy_rows(real_manifest, writer, sample_weight=1.0)
    synth_count = _copy_rows(synth_manifest, writer, sample_weight=0.2)

print(
    f"[WRAPPER] Combined manifest written to {combined_manifest} "
    f"(real={real_count}, synthetic={synth_count})."
)
PY

echo "[WRAPPER] Extracting the strict v5 weights checkpoint..." | tee -a "${LOG_FILE}"
STRICT_V5_MODEL="artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras"
STRICT_V5_MODEL_PATH="${STRICT_V5_MODEL}" \
INIT_WEIGHTS_PATH="${INIT_WEIGHTS}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Extract the raw weights payload from the strict v5 .keras archive."""

from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile


source_model = Path(os.environ["STRICT_V5_MODEL_PATH"])
target_weights = Path(os.environ["INIT_WEIGHTS_PATH"])
target_weights.parent.mkdir(parents=True, exist_ok=True)

with ZipFile(source_model, "r") as archive:
    with archive.open("model.weights.h5", "r") as src, target_weights.open("wb") as dst:
        dst.write(src.read())

print(f"[WRAPPER] Extracted weights checkpoint to {target_weights}.")
PY

echo "[WRAPPER] Fine-tuning strict v5 from the weighted combined manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${COMBINED_MANIFEST}" \
  --precomputed-crop-boxes "${REAL_BOXES_CSV}" \
  --init-model "${INIT_WEIGHTS}" \
  --linear-output \
  --augment-mode hard_preview \
  --batch-size 8 \
  --epochs 18 \
  --warmup-epochs 6 \
  --learning-rate 5e-5 \
  --fine-tune-lr 2e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --dropout 0.15 \
  --seed 22 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating the final model on the hard-case manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/eval_rectified_scalar_model_on_manifest.py \
  --model "${OUTPUT_DIR}/model.keras" \
  --manifest "${HARD_CASE_MANIFEST}" \
  --crop-boxes "${REAL_BOXES_CSV}" \
  2>&1 | tee -a "${LOG_FILE}"
