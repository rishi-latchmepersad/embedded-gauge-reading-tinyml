#!/usr/bin/env bash
set -euo pipefail

# Train the strict v5 scalar model with a two-stage cold-tail curriculum.
#
# Stage 1 mixes all available real labels with light standard synthetic renders.
# Stage 2 reuses the stage-1 checkpoint and focuses the loss on the cold tail
# and preview-like frames so the regressor stops collapsing toward the middle.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_curriculum_v23.log"
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
HOLDOUT_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"
REAL_MANIFEST="data/full_scalar_manifest_v1.csv"
REAL_BOXES_CSV="data/rectified_crop_boxes_full_scalar_v1.csv"
STRICT_V5_MODEL="artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras"

STAGE1_SYNTH_DIR="${TMP_DIR}/synth_gauge_curriculum_v23_stage1"
STAGE1_SYNTH_MANIFEST="${STAGE1_SYNTH_DIR}/manifest.csv"
STAGE1_MANIFEST="${TMP_DIR}/curriculum_v23_stage1_manifest.csv"
STAGE1_OUTPUT_DIR="artifacts/training/mobilenetv2_rectified_scalar_curriculum_v23_stage1"
STAGE1_WEIGHTS="${TMP_DIR}/mobilenetv2_rectified_scalar_curriculum_v23_stage1.weights.h5"

STAGE2_SYNTH_DIR="${TMP_DIR}/synth_gauge_curriculum_v23_stage2"
STAGE2_SYNTH_MANIFEST="${STAGE2_SYNTH_DIR}/manifest.csv"
STAGE2_MANIFEST="${TMP_DIR}/curriculum_v23_stage2_manifest.csv"
STAGE2_OUTPUT_DIR="artifacts/training/mobilenetv2_rectified_scalar_curriculum_v23"

echo "[WRAPPER] Starting rectified scalar curriculum v23." | tee "${LOG_FILE}"
echo "[WRAPPER] Stage 1 synthetic dir: ${STAGE1_SYNTH_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Stage 2 synthetic dir: ${STAGE2_SYNTH_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Real manifest:        ${REAL_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Holdout manifest:     ${HOLDOUT_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Real boxes CSV:       ${REAL_BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Strict v5 model:      ${STRICT_V5_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir:           ${STAGE2_OUTPUT_DIR}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Generating stage-1 synthetic gauge renders..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/generate_synthetic_gauge_dataset.py \
  --output-dir "${STAGE1_SYNTH_DIR}" \
  --manifest-path "${STAGE1_SYNTH_MANIFEST}" \
  --num-samples 1200 \
  --image-size 224 \
  --seed 23 \
  --profile standard \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building stage-1 broad curriculum manifest..." | tee -a "${LOG_FILE}"
REAL_MANIFEST_PATH="${REAL_MANIFEST}" \
HOLDOUT_MANIFEST_PATH="${HOLDOUT_MANIFEST}" \
SYNTH_MANIFEST_PATH="${STAGE1_SYNTH_MANIFEST}" \
OUT_MANIFEST_PATH="${STAGE1_MANIFEST}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Build the stage-1 manifest for the cold-tail curriculum."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def _load_holdout_paths(path: Path) -> set[str]:
    """Load the held-out hard-case image paths so training never sees them."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row["image_path"]).replace("\\", "/").strip()
            for row in reader
            if row.get("image_path")
        }


def _copy_rows(
    source_path: Path,
    writer: csv.DictWriter,
    *,
    holdout_paths: set[str],
    default_weight: float,
) -> int:
    """Copy rows into the manifest while excluding held-out hard-case samples."""
    count = 0
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = str(row.get("image_path", "")).replace("\\", "/").strip()
            value = str(row.get("value", "")).strip()
            if not image_path or not value or image_path in holdout_paths:
                continue
            writer.writerow(
                {
                    "image_path": image_path,
                    "value": value,
                    "sample_weight": f"{default_weight:.3f}",
                }
            )
            count += 1
    return count


real_manifest = Path(os.environ["REAL_MANIFEST_PATH"])
holdout_manifest = Path(os.environ["HOLDOUT_MANIFEST_PATH"])
synth_manifest = Path(os.environ["SYNTH_MANIFEST_PATH"])
out_manifest = Path(os.environ["OUT_MANIFEST_PATH"])

holdout_paths = _load_holdout_paths(holdout_manifest)
out_manifest.parent.mkdir(parents=True, exist_ok=True)

with out_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "sample_weight"])
    writer.writeheader()
    real_count = _copy_rows(
        real_manifest,
        writer,
        holdout_paths=holdout_paths,
        default_weight=1.0,
    )
    synth_count = _copy_rows(
        synth_manifest,
        writer,
        holdout_paths=set(),
        default_weight=0.15,
    )

print(
    f"[WRAPPER] Stage-1 manifest written to {out_manifest} "
    f"(real={real_count}, synthetic={synth_count})."
)
PY

echo "[WRAPPER] Training stage 1..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${STAGE1_OUTPUT_DIR}" \
  --manifest-path "${STAGE1_MANIFEST}" \
  --precomputed-crop-boxes "${REAL_BOXES_CSV}" \
  --init-model "${STRICT_V5_MODEL}" \
  --linear-output \
  --augment-mode standard \
  --batch-size 8 \
  --epochs 12 \
  --warmup-epochs 4 \
  --learning-rate 5e-5 \
  --fine-tune-lr 2e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --dropout 0.15 \
  --seed 23 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Extracting stage-1 weights checkpoint..." | tee -a "${LOG_FILE}"
STAGE1_MODEL="${STAGE1_OUTPUT_DIR}/model.keras"
STAGE1_MODEL_PATH="${STAGE1_MODEL}" \
STAGE1_WEIGHTS_PATH="${STAGE1_WEIGHTS}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Extract the raw weights payload from the stage-1 .keras archive."""

from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile


source_model = Path(os.environ["STAGE1_MODEL_PATH"])
target_weights = Path(os.environ["STAGE1_WEIGHTS_PATH"])
target_weights.parent.mkdir(parents=True, exist_ok=True)

with ZipFile(source_model, "r") as archive:
    with archive.open("model.weights.h5", "r") as src, target_weights.open("wb") as dst:
        dst.write(src.read())

print(f"[WRAPPER] Extracted weights checkpoint to {target_weights}.")
PY

echo "[WRAPPER] Generating stage-2 hard synthetic gauge renders..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/generate_synthetic_gauge_dataset.py \
  --output-dir "${STAGE2_SYNTH_DIR}" \
  --manifest-path "${STAGE2_SYNTH_MANIFEST}" \
  --num-samples 1000 \
  --image-size 224 \
  --seed 24 \
  --profile hard \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building stage-2 cold-tail curriculum manifest..." | tee -a "${LOG_FILE}"
REAL_MANIFEST_PATH="${REAL_MANIFEST}" \
HOLDOUT_MANIFEST_PATH="${HOLDOUT_MANIFEST}" \
SYNTH_MANIFEST_PATH="${STAGE2_SYNTH_MANIFEST}" \
OUT_MANIFEST_PATH="${STAGE2_MANIFEST}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Build the stage-2 manifest with explicit cold-tail emphasis."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def _load_holdout_paths(path: Path) -> set[str]:
    """Load the held-out hard-case image paths so training never sees them."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row["image_path"]).replace("\\", "/").strip()
            for row in reader
            if row.get("image_path")
        }


def _tail_weight(value: float, image_path: str) -> float:
    """Bias the cold end while still keeping the mid-band anchored."""
    preview_bonus = 1.25 if "preview" in image_path.lower() else 1.0
    if value <= -15.0:
        return 5.0 * preview_bonus
    if value <= -5.0:
        return 3.8 * preview_bonus
    if value < 5.0:
        return 2.6 * preview_bonus
    if value < 20.0:
        return 1.2 * preview_bonus
    if value < 35.0:
        return 1.0 * preview_bonus
    return 2.2 * preview_bonus


def _copy_rows(
    source_path: Path,
    writer: csv.DictWriter,
    *,
    holdout_paths: set[str],
    synthetic: bool = False,
) -> int:
    """Copy rows into the curriculum manifest with strong cold-tail weights."""
    count = 0
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = str(row.get("image_path", "")).replace("\\", "/").strip()
            value_str = str(row.get("value", "")).strip()
            if not image_path or not value_str or image_path in holdout_paths:
                continue
            value = float(value_str)
            sample_weight = _tail_weight(value, image_path)
            if synthetic:
                sample_weight *= 0.08
            writer.writerow(
                {
                    "image_path": image_path,
                    "value": value_str,
                    "sample_weight": f"{sample_weight:.3f}",
                }
            )
            count += 1
    return count


real_manifest = Path(os.environ["REAL_MANIFEST_PATH"])
holdout_manifest = Path(os.environ["HOLDOUT_MANIFEST_PATH"])
synth_manifest = Path(os.environ["SYNTH_MANIFEST_PATH"])
out_manifest = Path(os.environ["OUT_MANIFEST_PATH"])

holdout_paths = _load_holdout_paths(holdout_manifest)
out_manifest.parent.mkdir(parents=True, exist_ok=True)

with out_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "sample_weight"])
    writer.writeheader()
    real_count = _copy_rows(
        real_manifest,
        writer,
        holdout_paths=holdout_paths,
        synthetic=False,
    )
    synth_count = _copy_rows(
        synth_manifest,
        writer,
        holdout_paths=set(),
        synthetic=True,
    )

print(
    f"[WRAPPER] Stage-2 manifest written to {out_manifest} "
    f"(real={real_count}, synthetic={synth_count})."
)
PY

echo "[WRAPPER] Fine-tuning stage 2 from the stage-1 weights..." | tee -a "${LOG_FILE}"
STAGE2_INIT_WEIGHTS="${STAGE1_WEIGHTS}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${STAGE2_OUTPUT_DIR}" \
  --manifest-path "${STAGE2_MANIFEST}" \
  --precomputed-crop-boxes "${REAL_BOXES_CSV}" \
  --init-model "${STAGE2_INIT_WEIGHTS}" \
  --linear-output \
  --augment-mode hard_preview \
  --batch-size 8 \
  --epochs 14 \
  --warmup-epochs 4 \
  --learning-rate 2e-5 \
  --fine-tune-lr 1e-6 \
  --mobilenet-unfreeze-last-n 8 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --dropout 0.15 \
  --seed 24 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating the curriculum model on the hard-case manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/eval_rectified_scalar_model_on_manifest.py \
  --model "${STAGE2_OUTPUT_DIR}/model.keras" \
  --manifest "${HOLDOUT_MANIFEST}" \
  --crop-boxes "${REAL_BOXES_CSV}" \
  2>&1 | tee -a "${LOG_FILE}"
