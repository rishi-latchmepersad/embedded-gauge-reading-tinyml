#!/usr/bin/env bash
set -euo pipefail

# Train a polar + sweep-distribution reader on the broad real-plus-synthetic pool.
#
# This keeps the MobileNetV2 trunk compact, but flattens the gauge into polar
# coordinates and predicts an ordered sweep distribution instead of a single
# direct scalar. The goal is to keep the geometry signal while still using the
# full label pool and the synthetic renders we already have.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_polar_sweep_distribution_v36.log"
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
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

TMP_DIR="../tmp"
BASE_MANIFEST="${TMP_DIR}/combined_sweep_distribution_manifest_v35.csv"
SYNTH_MANIFEST="${TMP_DIR}/synth_gauge_pretrain_v1/manifest.csv"
COMBINED_MANIFEST="${TMP_DIR}/combined_polar_sweep_distribution_manifest_v36.csv"
STRICT_V5_MODEL="artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras"
STRICT_V5_WEIGHTS="${TMP_DIR}/mobilenetv2_rectified_scalar_strict_v5.weights.h5"

BOXES_CSV="data/rectified_crop_boxes_v5_all.csv"
OUTPUT_DIR="artifacts/training/mobilenetv2_polar_sweep_distribution_v36"
HARD_CASE_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"

echo "[WRAPPER] Starting polar sweep-distribution fine-tune v36." | tee "${LOG_FILE}"
echo "[WRAPPER] Base manifest:     ${BASE_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Synthetic manifest:${SYNTH_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Combined manifest:  ${COMBINED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:          ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init model:         ${STRICT_V5_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir:         ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building the polar sweep-distribution combined manifest..." | tee -a "${LOG_FILE}"
BASE_MANIFEST_PATH="${BASE_MANIFEST}" \
SYNTH_MANIFEST_PATH="${SYNTH_MANIFEST}" \
COMBINED_MANIFEST_PATH="${COMBINED_MANIFEST}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Combine the broad real manifest with synthetic gauge renders."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def _copy_rows(
    source_path: Path,
    writer: csv.DictWriter,
    *,
    sample_weight: float,
) -> int:
    """Copy one source manifest into the combined manifest."""
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
            weight_raw = row.get("sample_weight", "")
            writer.writerow(
                {
                    "image_path": image_path.replace("\\", "/"),
                    "value": value,
                    "sample_weight": (
                        f"{sample_weight:.3f}"
                        if not weight_raw
                        else str(weight_raw).strip()
                    ),
                }
            )
            count += 1
    return count


base_manifest = Path(os.environ["BASE_MANIFEST_PATH"])
synth_manifest = Path(os.environ["SYNTH_MANIFEST_PATH"])
combined_manifest = Path(os.environ["COMBINED_MANIFEST_PATH"])
combined_manifest.parent.mkdir(parents=True, exist_ok=True)

with combined_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "sample_weight"])
    writer.writeheader()
    base_count = _copy_rows(base_manifest, writer, sample_weight=1.0)
    synth_count = _copy_rows(synth_manifest, writer, sample_weight=0.35)

print(
    f"[WRAPPER] Combined manifest written to {combined_manifest} "
    f"(base={base_count}, synthetic={synth_count})."
)
PY

echo "[WRAPPER] Extracting strict v5 warm-start weights..." | tee -a "${LOG_FILE}"
STRICT_V5_MODEL_PATH="${STRICT_V5_MODEL}" \
STRICT_V5_WEIGHTS_PATH="${STRICT_V5_WEIGHTS}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Extract the weights payload from the strict v5 checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile


source_model = Path(os.environ["STRICT_V5_MODEL_PATH"])
target_weights = Path(os.environ["STRICT_V5_WEIGHTS_PATH"])
target_weights.parent.mkdir(parents=True, exist_ok=True)

with ZipFile(source_model, "r") as archive:
    with archive.open("model.weights.h5", "r") as src, target_weights.open("wb") as dst:
        dst.write(src.read())

print(f"[WRAPPER] Extracted weights checkpoint to {target_weights}.")
PY

echo "[WRAPPER] Fine-tuning the polar sweep-distribution reader..." | tee -a "${LOG_FILE}"
VENV_DIR="$("${POETRY_BIN}" env info -p)"
# Source the Poetry venv so the WSL GPU library paths are restored before TF starts.
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

NVIDIA_LIB_DIRS=("${VENV_DIR}"/lib/python*/site-packages/nvidia/*/lib)
if (( ${#NVIDIA_LIB_DIRS[@]} > 0 )); then
  NVIDIA_LIB_PATH="$(IFS=:; echo "${NVIDIA_LIB_DIRS[*]}")"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

python -u scripts/train_all_data_baseline.py \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${COMBINED_MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${STRICT_V5_WEIGHTS}" \
  --aux-head-kind sweep_distribution \
  --polar-sweep-distribution-model \
  --no-gpu-memory-growth \
  --augment-mode hard_preview \
  --batch-size 8 \
  --epochs 16 \
  --warmup-epochs 6 \
  --learning-rate 5e-5 \
  --fine-tune-lr 2e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 96 \
  --dropout 0.15 \
  --sweep-distribution-bins 81 \
  --sweep-distribution-sigma-bins 1.5 \
  --seed 24 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating the final model on the hard-case manifest..." | tee -a "${LOG_FILE}"
python -u scripts/eval_rectified_scalar_model_on_manifest.py \
  --model "${OUTPUT_DIR}/model.keras" \
  --manifest "${HARD_CASE_MANIFEST}" \
  --crop-boxes "${BOXES_CSV}" \
  --polar-sweep-distribution-model \
  2>&1 | tee -a "${LOG_FILE}"
