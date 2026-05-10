#!/usr/bin/env bash
set -euo pipefail

# Train a sweep-distribution MobileNetV2 on the broadest usable data mix.
#
# This keeps the proven v5 MobileNetV2 trunk, but replaces the scalar head
# with a sweep-distribution reader so the model learns the gauge range as a
# smooth ordered distribution rather than a single direct number.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_sweep_distribution_v35.log"
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
SYNTH_DIR="${TMP_DIR}/synth_gauge_sweep_distribution_v35"
SYNTH_MANIFEST="${SYNTH_DIR}/manifest.csv"
COMBINED_MANIFEST="${TMP_DIR}/combined_sweep_distribution_manifest_v35.csv"
INIT_WEIGHTS="${TMP_DIR}/mobilenetv2_rectified_scalar_strict_v5.weights.h5"

LABELLED_MANIFEST="data/full_labelled_plus_board30_valid_with_new5.csv"
SCALAR_MANIFEST="data/full_scalar_manifest_v1.csv"
BOXES_CSV="data/rectified_crop_boxes_full_scalar_v1.csv"
OUTPUT_DIR="artifacts/training/mobilenetv2_sweep_distribution_v35"
HARD_CASE_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"

LOG_DIR_REL="artifacts/training_logs"

echo "[WRAPPER] Starting sweep-distribution all-sources fine-tune v35." | tee "${LOG_FILE}"
echo "[WRAPPER] Synthetic dir:     ${SYNTH_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Combined manifest:  ${COMBINED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Labelled manifest:  ${LABELLED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Scalar manifest:    ${SCALAR_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:          ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init weights:       ${INIT_WEIGHTS}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir:         ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Generating corrected synthetic gauge renders..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/generate_synthetic_gauge_dataset.py \
  --output-dir "${SYNTH_DIR}" \
  --manifest-path "${SYNTH_MANIFEST}" \
  --num-samples 1000 \
  --image-size 224 \
  --seed 24 \
  --profile standard \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building the deduped all-sources manifest..." | tee -a "${LOG_FILE}"
LABELLED_MANIFEST_PATH="${LABELLED_MANIFEST}" \
SCALAR_MANIFEST_PATH="${SCALAR_MANIFEST}" \
SYNTH_MANIFEST_PATH="${SYNTH_MANIFEST}" \
COMBINED_MANIFEST_PATH="${COMBINED_MANIFEST}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Build a broad real + synthetic manifest with source priority and light synthetic weight."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def _normalize(image_path: str) -> str:
    """Normalize manifest paths so duplicates collapse cleanly."""
    return image_path.replace("\\", "/").strip()


def _append_rows(
    source_path: Path,
    writer: csv.DictWriter,
    *,
    seen_paths: set[str],
    sample_weight: float,
    count_label: str,
) -> int:
    """Append unseen rows from one source manifest into the combined manifest."""
    count = 0
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = _normalize(str(row.get("image_path", "")))
            value = str(row.get("value", "")).strip()
            if not image_path or not value or image_path in seen_paths:
                continue
            seen_paths.add(image_path)
            writer.writerow(
                {
                    "image_path": image_path,
                    "value": value,
                    "sample_weight": f"{sample_weight:.3f}",
                }
            )
            count += 1
    print(f"[WRAPPER] Added {count} rows from {count_label}.")
    return count


labelled_manifest = Path(os.environ["LABELLED_MANIFEST_PATH"])
scalar_manifest = Path(os.environ["SCALAR_MANIFEST_PATH"])
synth_manifest = Path(os.environ["SYNTH_MANIFEST_PATH"])
combined_manifest = Path(os.environ["COMBINED_MANIFEST_PATH"])
combined_manifest.parent.mkdir(parents=True, exist_ok=True)

seen_paths: set[str] = set()
with combined_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "sample_weight"])
    writer.writeheader()
    labelled_count = _append_rows(
        labelled_manifest,
        writer,
        seen_paths=seen_paths,
        sample_weight=1.0,
        count_label="full_labelled_plus_board30_valid_with_new5",
    )
    scalar_count = _append_rows(
        scalar_manifest,
        writer,
        seen_paths=seen_paths,
        sample_weight=0.9,
        count_label="full_scalar_manifest_v1",
    )
    synth_count = _append_rows(
        synth_manifest,
        writer,
        seen_paths=seen_paths,
        sample_weight=0.12,
        count_label="synthetic_standard",
    )

print(
    f"[WRAPPER] Combined manifest written to {combined_manifest} "
    f"(labelled={labelled_count}, scalar={scalar_count}, synthetic={synth_count})."
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

echo "[WRAPPER] Fine-tuning sweep-distribution reader on all sources..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${COMBINED_MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${INIT_WEIGHTS}" \
  --augment-mode hard_preview \
  --batch-size 4 \
  --epochs 12 \
  --warmup-epochs 4 \
  --learning-rate 5e-5 \
  --fine-tune-lr 1e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --dropout 0.15 \
  --aux-head-kind sweep_distribution \
  --aux-loss-weight 0.35 \
  --sweep-distribution-bins 41 \
  --sweep-distribution-sigma-bins 1.5 \
  --seed 24 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating the final model on the hard-case manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/eval_rectified_scalar_model_on_manifest.py \
  --model "${OUTPUT_DIR}/model.keras" \
  --manifest "${HARD_CASE_MANIFEST}" \
  --crop-boxes "${BOXES_CSV}" \
  2>&1 | tee -a "${LOG_FILE}"
