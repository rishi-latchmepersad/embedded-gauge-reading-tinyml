#!/usr/bin/env bash
set -euo pipefail

# Train a hard-tail specialist rectified scalar model from the strict v5 backbone.
#
# This run keeps the proven MobileNetV2 v5 setup, but builds a training mix that
# heavily emphasizes the cold and hot ends where the current scalar regressor
# still collapses toward the middle.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_hardtail_specialist_v25.log"
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
SYNTH_DIR="${TMP_DIR}/synth_gauge_hardtail_specialist_v25"
SYNTH_MANIFEST="${SYNTH_DIR}/manifest.csv"
TRAIN_MANIFEST="${TMP_DIR}/hardtail_specialist_manifest_v25.csv"
INIT_WEIGHTS="${TMP_DIR}/mobilenetv2_rectified_scalar_strict_v5.weights.h5"

STRICT_MANIFEST="data/rectified_scalar_strict_train_v5.csv"
LABELLED_MANIFEST="data/full_labelled_plus_board30_valid_with_new5.csv"
SCALAR_MANIFEST="data/full_scalar_manifest_v1.csv"
HOLDOUT_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"
BOXES_CSV="data/rectified_crop_boxes_full_scalar_v1.csv"
OUTPUT_DIR="artifacts/training/mobilenetv2_rectified_scalar_hardtail_specialist_v25"

echo "[WRAPPER] Starting hard-tail specialist rectified scalar v25." | tee "${LOG_FILE}"
echo "[WRAPPER] Synthetic dir:   ${SYNTH_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Train manifest:  ${TRAIN_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Strict manifest: ${STRICT_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Labelled source: ${LABELLED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Scalar source:   ${SCALAR_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Holdout eval:    ${HOLDOUT_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:       ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init weights:    ${INIT_WEIGHTS}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir:      ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Generating hard synthetic renders..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/generate_synthetic_gauge_dataset.py \
  --output-dir "${SYNTH_DIR}" \
  --manifest-path "${SYNTH_MANIFEST}" \
  --num-samples 1200 \
  --image-size 224 \
  --seed 25 \
  --profile hard \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building the hard-tail specialist manifest..." | tee -a "${LOG_FILE}"
STRICT_MANIFEST_PATH="${STRICT_MANIFEST}" \
LABELLED_MANIFEST_PATH="${LABELLED_MANIFEST}" \
SCALAR_MANIFEST_PATH="${SCALAR_MANIFEST}" \
SYNTH_MANIFEST_PATH="${SYNTH_MANIFEST}" \
HOLDOUT_MANIFEST_PATH="${HOLDOUT_MANIFEST}" \
TRAIN_MANIFEST_PATH="${TRAIN_MANIFEST}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Build a hard-tail training manifest for the specialist model."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def _normalize(path_str: str) -> str:
    """Normalize path separators and surrounding whitespace."""
    return path_str.replace("\\", "/").strip()


def _load_paths(path: Path) -> set[str]:
    """Load a manifest into a set of image paths."""
    rows: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = _normalize(str(row.get("image_path", "")))
            if image_path:
                rows.add(image_path)
    return rows


def _value_weight(value: float, image_path: str) -> float:
    """Emphasize cold and hot labels more than the middle band."""
    preview_bonus = 1.25 if "preview" in image_path.lower() else 1.0
    if value <= -15.0:
        return 5.0 * preview_bonus
    if value <= 0.0:
        return 4.0 * preview_bonus
    if value <= 10.0:
        return 2.8 * preview_bonus
    if value < 20.0:
        return 1.0 * preview_bonus
    if value < 35.0:
        return 1.2 * preview_bonus
    if value < 45.0:
        return 3.8 * preview_bonus
    return 5.2 * preview_bonus


strict_paths = _load_paths(Path(os.environ["STRICT_MANIFEST_PATH"]))
holdout_paths = _load_paths(Path(os.environ["HOLDOUT_MANIFEST_PATH"]))
train_manifest = Path(os.environ["TRAIN_MANIFEST_PATH"])
synth_manifest = Path(os.environ["SYNTH_MANIFEST_PATH"])
labelled_manifest = Path(os.environ["LABELLED_MANIFEST_PATH"])
scalar_manifest = Path(os.environ["SCALAR_MANIFEST_PATH"])

rows: list[dict[str, str]] = []
seen: set[str] = set()

def _append_strict_anchor() -> int:
    """Add the strict v5 anchor rows unchanged."""
    count = 0
    with Path(os.environ["STRICT_MANIFEST_PATH"]).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = _normalize(str(row.get("image_path", "")))
            value = str(row.get("value", "")).strip()
            if not image_path or not value or image_path in holdout_paths or image_path in seen:
                continue
            seen.add(image_path)
            rows.append(
                {
                    "image_path": image_path,
                    "value": value,
                    "sample_weight": "1.0",
                    "source_tag": "strict_anchor",
                }
            )
            count += 1
    return count


def _append_tail_rows(source_path: Path, source_tag: str) -> int:
    """Add only the cold/hot tail rows from a manifest."""
    count = 0
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = _normalize(str(row.get("image_path", "")))
            value_str = str(row.get("value", "")).strip()
            if not image_path or not value_str or image_path in holdout_paths or image_path in seen:
                continue
            try:
                value = float(value_str)
            except Exception:
                continue
            if -15.0 < value < 10.0 and value < 35.0:
                continue
            seen.add(image_path)
            rows.append(
                {
                    "image_path": image_path,
                    "value": value_str,
                    "sample_weight": f"{_value_weight(value, image_path):.3f}",
                    "source_tag": source_tag,
                }
            )
            count += 1
    return count


def _append_synthetic_rows(source_path: Path) -> int:
    """Append a small amount of hard synthetic regularization data."""
    count = 0
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = _normalize(str(row.get("image_path", "")))
            value_str = str(row.get("value", "")).strip()
            if not image_path or not value_str or image_path in seen:
                continue
            try:
                value = float(value_str)
            except Exception:
                continue
            if -15.0 < value < 5.0 and value < 35.0:
                # Keep the synthetic regularizer focused on the hard tails.
                continue
            seen.add(image_path)
            rows.append(
                {
                    "image_path": image_path,
                    "value": value_str,
                    "sample_weight": f"{0.08 * _value_weight(value, image_path):.3f}",
                    "source_tag": "synthetic_hard",
                }
            )
            count += 1
    return count


strict_count = _append_strict_anchor()
labelled_count = _append_tail_rows(labelled_manifest, "labelled_tail")
scalar_count = _append_tail_rows(scalar_manifest, "scalar_tail")
synth_count = _append_synthetic_rows(synth_manifest)

if len(rows) < 180:
    raise SystemExit(f"[WRAPPER] Refusing to train on only {len(rows)} rows")

train_manifest.parent.mkdir(parents=True, exist_ok=True)
with train_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=["image_path", "value", "sample_weight", "source_tag"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(
    f"[WRAPPER] Wrote {len(rows)} hard-tail rows to {train_manifest} "
    f"(strict={strict_count}, labelled_tail={labelled_count}, scalar_tail={scalar_count}, synthetic={synth_count})"
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

echo "[WRAPPER] Training the hard-tail specialist..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${TRAIN_MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${INIT_WEIGHTS}" \
  --linear-output \
  --augment-mode hard_preview \
  --batch-size 6 \
  --epochs 16 \
  --warmup-epochs 5 \
  --learning-rate 3e-5 \
  --fine-tune-lr 1e-6 \
  --mobilenet-unfreeze-last-n 8 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --dropout 0.15 \
  --seed 25 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating the hard-tail specialist on the hard-case manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/eval_rectified_scalar_model_on_manifest.py \
  --model "${OUTPUT_DIR}/model.keras" \
  --manifest "${HOLDOUT_MANIFEST}" \
  --crop-boxes "${BOXES_CSV}" \
  2>&1 | tee -a "${LOG_FILE}"
