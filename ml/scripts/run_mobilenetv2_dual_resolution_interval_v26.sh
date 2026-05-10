#!/usr/bin/env bash
set -euo pipefail

# Train the new dual-resolution MobileNetV2 interval model on all real sources
# plus corrected synthetic gauge renders.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_dual_resolution_interval_v26.log"
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

TMP_DIR="../tmp/dual_resolution_v26"
SYNTH_STANDARD_DIR="${TMP_DIR}/synthetic_standard"
SYNTH_HARD_DIR="${TMP_DIR}/synthetic_hard"
SYNTH_STANDARD_MANIFEST="${SYNTH_STANDARD_DIR}/manifest.csv"
SYNTH_HARD_MANIFEST="${SYNTH_HARD_DIR}/manifest.csv"
COMBINED_MANIFEST="${TMP_DIR}/combined_dual_resolution_manifest_v26.csv"
INIT_WEIGHTS="${TMP_DIR}/mobilenetv2_rectified_scalar_strict_v5.weights.h5"

LABELLED_MANIFEST="data/full_labelled_plus_board30_valid_with_new5.csv"
SCALAR_MANIFEST="data/full_scalar_manifest_v1.csv"
BOXES_CSV="data/rectified_crop_boxes_v5_all.csv"
STRICT_V5_MODEL="artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras"
OUTPUT_DIR="artifacts/training/mobilenetv2_dual_resolution_interval_v26"
HARD_CASE_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"

echo "[WRAPPER] Starting dual-resolution interval v26." | tee "${LOG_FILE}"
echo "[WRAPPER] Standard synth dir: ${SYNTH_STANDARD_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Hard synth dir:     ${SYNTH_HARD_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Combined manifest:  ${COMBINED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Labelled manifest:  ${LABELLED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Scalar manifest:    ${SCALAR_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:          ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init weights:       ${INIT_WEIGHTS}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir:         ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Generating standard synthetic gauge renders..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/generate_synthetic_gauge_dataset.py \
  --output-dir "${SYNTH_STANDARD_DIR}" \
  --manifest-path "${SYNTH_STANDARD_MANIFEST}" \
  --num-samples 1000 \
  --image-size 224 \
  --seed 26 \
  --profile standard \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Generating hard synthetic gauge renders..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/generate_synthetic_gauge_dataset.py \
  --output-dir "${SYNTH_HARD_DIR}" \
  --manifest-path "${SYNTH_HARD_MANIFEST}" \
  --num-samples 600 \
  --image-size 224 \
  --seed 126 \
  --profile hard \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building the combined dual-resolution manifest..." | tee -a "${LOG_FILE}"
LABELLED_MANIFEST_PATH="${LABELLED_MANIFEST}" \
SCALAR_MANIFEST_PATH="${SCALAR_MANIFEST}" \
SYNTH_STANDARD_MANIFEST_PATH="${SYNTH_STANDARD_MANIFEST}" \
SYNTH_HARD_MANIFEST_PATH="${SYNTH_HARD_MANIFEST}" \
COMBINED_MANIFEST_PATH="${COMBINED_MANIFEST}" \
"${POETRY_BIN}" run python -u - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Build a weighted real + synthetic manifest for the dual-resolution run."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def _normalize(image_path: str) -> str:
    """Normalize a manifest path so duplicates collapse cleanly."""
    return image_path.replace("\\", "/").strip()


def _append_rows(
    source_path: Path,
    writer: csv.DictWriter,
    *,
    seen_paths: set[str],
    sample_weight: float,
    count_label: str,
    add_tmp_prefix: bool = False,
) -> int:
    """Append unseen rows from one manifest with a fixed sample weight."""
    count = 0
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = _normalize(str(row.get("image_path", "")))
            value = str(row.get("value", "")).strip()
            if not image_path or not value or image_path in seen_paths:
                continue
            if add_tmp_prefix and not image_path.startswith("tmp/"):
                image_path = f"tmp/{image_path.lstrip('./')}"
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
synthetic_standard_manifest = Path(os.environ["SYNTH_STANDARD_MANIFEST_PATH"])
synthetic_hard_manifest = Path(os.environ["SYNTH_HARD_MANIFEST_PATH"])
combined_manifest = Path(os.environ["COMBINED_MANIFEST_PATH"])
combined_manifest.parent.mkdir(parents=True, exist_ok=True)

seen_paths: set[str] = set()
with combined_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=["image_path", "value", "sample_weight"]
    )
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
    standard_synth_count = _append_rows(
        synthetic_standard_manifest,
        writer,
        seen_paths=seen_paths,
        sample_weight=0.15,
        count_label="synthetic_standard",
        add_tmp_prefix=True,
    )
    hard_synth_count = _append_rows(
        synthetic_hard_manifest,
        writer,
        seen_paths=seen_paths,
        sample_weight=0.20,
        count_label="synthetic_hard",
        add_tmp_prefix=True,
    )

print(
    f"[WRAPPER] Combined manifest written to {combined_manifest} "
    f"(labelled={labelled_count}, scalar={scalar_count}, "
    f"synthetic_standard={standard_synth_count}, synthetic_hard={hard_synth_count})."
)
PY

echo "[WRAPPER] Extracting the strict v5 weights checkpoint..." | tee -a "${LOG_FILE}"
STRICT_V5_MODEL_PATH="${STRICT_V5_MODEL}" \
INIT_WEIGHTS_PATH="${INIT_WEIGHTS}" \
"${POETRY_BIN}" run python -u - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Extract weights from the strict v5 model so warm-start loading stays light."""

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

echo "[WRAPPER] Training the new dual-resolution interval model..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${COMBINED_MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --no-gpu-memory-growth \
  --init-model "${INIT_WEIGHTS}" \
  --dual-resolution-model \
  --dual-resolution-crop-ratio 0.78 \
  --aux-head-kind interval \
  --interval-bin-width 5.0 \
  --aux-loss-weight 0.25 \
  --augment-mode hard_preview \
  --batch-size 4 \
  --epochs 24 \
  --warmup-epochs 6 \
  --learning-rate 8e-5 \
  --fine-tune-lr 2e-5 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 96 \
  --dropout 0.20 \
  --seed 26 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating the final model on the hard-case manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/eval_rectified_scalar_model_on_manifest.py \
  --model "${OUTPUT_DIR}/model.keras" \
  --manifest "${HARD_CASE_MANIFEST}" \
  --crop-boxes "${BOXES_CSV}" \
  2>&1 | tee -a "${LOG_FILE}"
