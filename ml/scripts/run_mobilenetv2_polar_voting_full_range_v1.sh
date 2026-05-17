#!/usr/bin/env bash
set -euo pipefail

# Train a polar voting-style sweep-distribution reader on the full real pool.
#
# The goal here is to mirror the classical polar baseline's geometry bias as
# closely as we can in CNN form: build a deduplicated manifest from the full
# labelled pool, the live captured board images, and the hard-case rows, then
# fine-tune the existing polar sweep-distribution model on that real-data mix.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_polar_voting_full_range_v1.log"
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
MANIFEST_PATH="${TMP_DIR}/polar_voting_full_range_manifest_v1.csv"
HARD_CASES_MANIFEST="data/hard_cases_plus_board30_valid_with_new6.csv"
CAPTURED_MANIFEST="data/all_captured_images_manifest.csv"
FULL_LABELLED_MANIFEST="data/full_labelled_plus_board30_valid_with_new5.csv"
UNIFIED_MANIFEST="data/unified_training_manifest_v1.csv"
INIT_MODEL="artifacts/training/mobilenetv2_polar_sweep_distribution_v36/model.keras"
BOXES_CSV="data/rectified_crop_boxes_v5_all.csv"
OUTPUT_DIR="artifacts/training/mobilenetv2_polar_voting_full_range_v1"
HARD_CASE_EVAL_MANIFEST="data/hard_cases_plus_board30_valid_with_new6.csv"
BOARD_EVAL_MANIFEST="data/board_rectified_probe_20260422.csv"

echo "[WRAPPER] Starting polar voting full-range fine-tune v1." | tee "${LOG_FILE}"
echo "[WRAPPER] Manifest:       ${MANIFEST_PATH}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Hard cases:     ${HARD_CASES_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Captures:       ${CAPTURED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Full labelled:  ${FULL_LABELLED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Unified pool:   ${UNIFIED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init model:     ${INIT_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir:     ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building the deduplicated real-data manifest..." | tee -a "${LOG_FILE}"
HARD_CASES_MANIFEST_PATH="${HARD_CASES_MANIFEST}" \
CAPTURED_MANIFEST_PATH="${CAPTURED_MANIFEST}" \
FULL_LABELLED_MANIFEST_PATH="${FULL_LABELLED_MANIFEST}" \
UNIFIED_MANIFEST_PATH="${UNIFIED_MANIFEST}" \
MANIFEST_PATH="${MANIFEST_PATH}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Merge the real manifests into one weighted training pool."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceManifest:
    """One manifest source and its default sample weight."""

    path: Path
    default_weight: float


def _iter_rows(source: SourceManifest) -> tuple[int, list[dict[str, str]]]:
    """Load rows from one source manifest using UTF-8 BOM tolerant parsing."""
    rows: list[dict[str, str]] = []
    with source.path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {source.path}")
        for row in reader:
            image_path = str(row.get("image_path", "")).strip()
            value = str(row.get("value", "")).strip()
            if not image_path or not value:
                continue
            weight_raw = str(row.get("sample_weight", "")).strip()
            rows.append(
                {
                    "image_path": image_path.replace("\\", "/"),
                    "value": value,
                    "sample_weight": weight_raw or f"{source.default_weight:.3f}",
                }
            )
    return len(rows), rows


sources = [
    # Put the hard cases first so deduplication preserves their stronger weight.
    SourceManifest(
        Path(os.environ["HARD_CASES_MANIFEST_PATH"]),
        default_weight=1.75,
    ),
    # Keep the live board captures in the mix, but do not let them dominate.
    SourceManifest(
        Path(os.environ["CAPTURED_MANIFEST_PATH"]),
        default_weight=0.90,
    ),
    # The full labelled pool fills in the general dial range and middle values.
    SourceManifest(
        Path(os.environ["FULL_LABELLED_MANIFEST_PATH"]),
        default_weight=1.00,
    ),
    # The unified manifest contributes any extra real rows not covered above.
    SourceManifest(
        Path(os.environ["UNIFIED_MANIFEST_PATH"]),
        default_weight=0.85,
    ),
]

manifest_path = Path(os.environ["MANIFEST_PATH"])
manifest_path.parent.mkdir(parents=True, exist_ok=True)

seen_paths: set[str] = set()
total_written = 0
with manifest_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["image_path", "value", "sample_weight"],
    )
    writer.writeheader()

    for source in sources:
        count, rows = _iter_rows(source)
        kept = 0
        for row in rows:
            image_path = row["image_path"]
            if image_path in seen_paths:
                continue
            seen_paths.add(image_path)
            writer.writerow(row)
            kept += 1
        total_written += kept
        print(
            f"[WRAPPER] Source {source.path.name}: loaded={count} kept={kept} "
            f"default_weight={source.default_weight:.2f}"
        )

print(f"[WRAPPER] Combined manifest written to {manifest_path} ({total_written} rows).")
PY

echo "[WRAPPER] Fine-tuning the polar voting reader..." | tee -a "${LOG_FILE}"
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
  --manifest-path "${MANIFEST_PATH}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${INIT_MODEL}" \
  --aux-head-kind sweep_distribution \
  --polar-sweep-distribution-model \
  --no-gpu-memory-growth \
  --batch-size 8 \
  --epochs 24 \
  --warmup-epochs 8 \
  --learning-rate 3e-5 \
  --fine-tune-lr 1e-6 \
  --mobilenet-unfreeze-last-n 16 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 96 \
  --dropout 0.15 \
  --sweep-distribution-bins 81 \
  --sweep-distribution-sigma-bins 1.25 \
  --aux-loss-weight 0.40 \
  --seed 24 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating the final model on the hard-case manifest..." | tee -a "${LOG_FILE}"
python -u scripts/eval_rectified_scalar_model_on_manifest.py \
  --model "${OUTPUT_DIR}/model.keras" \
  --manifest "${HARD_CASE_EVAL_MANIFEST}" \
  --crop-boxes "${BOXES_CSV}" \
  --polar-sweep-distribution-model \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating the final model on the board probe manifest..." | tee -a "${LOG_FILE}"
python -u scripts/eval_rectified_scalar_model_on_manifest.py \
  --model "${OUTPUT_DIR}/model.keras" \
  --manifest "${BOARD_EVAL_MANIFEST}" \
  --crop-boxes "${BOXES_CSV}" \
  --polar-sweep-distribution-model \
  2>&1 | tee -a "${LOG_FILE}"
