#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the strict rectified scalar model after dropping preview frames.
#
# The cleaned pool keeps the same rectified/captured-image contract, but it
# removes preview-style samples that repeatedly showed large hard-case errors.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_cleaned_v13.log"
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

TRAIN_MANIFEST="data/rectified_scalar_strict_cleaned_v13.csv"
HELD_OUT_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"
BOXES_CSV="data/rectified_crop_boxes_v5_all.csv"
INIT_MODEL="artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras"
OUTPUT_DIR="artifacts/training/mobilenetv2_rectified_scalar_cleaned_v13"

echo "[WRAPPER] Starting cleaned rectified fine-tune v13." | tee "${LOG_FILE}"
echo "[WRAPPER] Train manifest: ${TRAIN_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Held-out eval:  ${HELD_OUT_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:      ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init model:     ${INIT_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:       ${LOG_FILE}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Rebuilding cleaned rectified training manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u - <<'PY'
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

root = Path.cwd()
base_manifest = root / "data" / "rectified_scalar_strict_train_v5.csv"
out_manifest = root / "data" / "rectified_scalar_strict_cleaned_v13.csv"

kept_rows: list[dict[str, str]] = []
preview_dropped = 0
with base_manifest.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        image_path = (row["image_path"].replace("\\", "/")).strip()
        if "preview" in image_path.lower():
            preview_dropped += 1
            continue
        kept_rows.append(
            {
                "image_path": image_path,
                "value": row["value"],
            }
        )

if len(kept_rows) < 100:
    raise SystemExit(f"[WRAPPER] Refusing to train on only {len(kept_rows)} rows")

value_counts = Counter(row["value"] for row in kept_rows)
if len(value_counts) < 6:
    raise SystemExit(
        f"[WRAPPER] Refusing collapsed rectified pool with only {len(value_counts)} unique labels"
    )

out_manifest.parent.mkdir(parents=True, exist_ok=True)
with out_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value"])
    writer.writeheader()
    writer.writerows(kept_rows)

print(
    f"[WRAPPER] Wrote {len(kept_rows)} cleaned rectified rows to {out_manifest} "
    f"(dropped {preview_dropped} preview rows)"
)
PY

echo "[WRAPPER] Launching fine-tune..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${TRAIN_MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${INIT_MODEL}" \
  --linear-output \
  --batch-size 4 \
  --epochs 16 \
  --warmup-epochs 6 \
  --learning-rate 5e-5 \
  --fine-tune-lr 2e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --dropout 0.15 \
  --seed 21 \
  2>&1 | tee -a "${LOG_FILE}"
