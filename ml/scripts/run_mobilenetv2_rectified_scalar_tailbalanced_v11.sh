#!/usr/bin/env bash
set -euo pipefail

# Train a tail-balanced linear rectified scalar model.
#
# This keeps the strict rectified crop contract but gives the cold/hot ends more
# influence through explicit sample weights in the manifest.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_tailbalanced_v11.log"
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

TRAIN_MANIFEST="data/rectified_scalar_tailbalanced_train_v11.csv"
HELD_OUT_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"
BOXES_CSV="data/rectified_crop_boxes_v5_all.csv"
INIT_MODEL="artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras"
OUTPUT_DIR="artifacts/training/mobilenetv2_rectified_scalar_tailbalanced_v11"

echo "[WRAPPER] Starting tail-balanced rectified scalar fine-tune v11." | tee "${LOG_FILE}"
echo "[WRAPPER] Train manifest: ${TRAIN_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Held-out eval:  ${HELD_OUT_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:      ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init model:     ${INIT_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:       ${LOG_FILE}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building tail-balanced rectified training manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u - <<'PY'
from __future__ import annotations

import csv
from pathlib import Path

root = Path.cwd()
base_manifest = root / "data" / "full_scalar_manifest_v1.csv"
held_out_manifest = root / "data" / "hard_cases_plus_board30_valid_with_new5.csv"
out_manifest = root / "data" / "rectified_scalar_tailbalanced_train_v11.csv"
captured_prefix = "ml/data/captured_images/"

with held_out_manifest.open("r", encoding="utf-8", newline="") as handle:
    held_out_rows = {
        (row["image_path"].replace("\\", "/")).strip()
        for row in csv.DictReader(handle)
    }

def tail_weight(value: float) -> float:
    """Bias the cold and hot extremes more heavily than the middle band."""
    if value <= -20.0:
        return 3.5
    if value <= -10.0:
        return 2.5
    if value < 0.0:
        return 1.7
    if value < 20.0:
        return 1.0
    if value < 30.0:
        return 1.1
    if value < 40.0:
        return 1.8
    return 3.0

kept_rows: list[dict[str, str]] = []
with base_manifest.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        image_path = (row["image_path"].replace("\\", "/")).strip()
        if not image_path.startswith(captured_prefix):
            continue
        if image_path in held_out_rows:
            continue
        value = float(row["value"])
        kept_rows.append(
            {
                "image_path": image_path,
                "value": row["value"],
                "sample_weight": f"{tail_weight(value):.3f}",
            }
        )

if len(kept_rows) < 100:
    raise SystemExit(f"[WRAPPER] Refusing to train on only {len(kept_rows)} rows")

out_manifest.parent.mkdir(parents=True, exist_ok=True)
with out_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "sample_weight"])
    writer.writeheader()
    writer.writerows(kept_rows)

print(f"[WRAPPER] Wrote {len(kept_rows)} tail-balanced rows to {out_manifest}")
PY

echo "[WRAPPER] Launching training..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${TRAIN_MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${INIT_MODEL}" \
  --linear-output \
  --batch-size 4 \
  --epochs 16 \
  --warmup-epochs 6 \
  --learning-rate 1e-4 \
  --fine-tune-lr 2e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --dropout 0.15 \
  --seed 21 \
  2>&1 | tee -a "${LOG_FILE}"
