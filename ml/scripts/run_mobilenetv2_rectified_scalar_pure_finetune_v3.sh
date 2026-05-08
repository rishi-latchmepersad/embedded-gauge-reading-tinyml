#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the rectified scalar model with a stricter manifest and a staged
# MobileNetV2 unfreeze schedule.
#
# This run keeps the hard-case eval manifest out of training, uses the
# rectifier-aligned crop boxes when available, and only unfreezes the top part
# of the pretrained backbone during the low-LR stage.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_strict_v4.log"
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

TRAIN_MANIFEST="data/rectified_scalar_strict_train_v5.csv"
BASE_MANIFEST="data/full_scalar_manifest_v1.csv"
HELD_OUT_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"
BOXES_CSV="data/rectified_crop_boxes_v5_all.csv"
INIT_MODEL="artifacts/training/mobilenetv2_tiny_224_full/model.keras"

echo "[WRAPPER] Starting strict rectified scalar fine-tune v4." | tee "${LOG_FILE}"
echo "[WRAPPER] Train manifest: ${TRAIN_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Held-out eval:  ${HELD_OUT_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:      ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init model:     ${INIT_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:       ${LOG_FILE}" | tee -a "${LOG_FILE}"

# Build a strict rectified-only training manifest on every run so we never
# accidentally reuse a stale file that still contains raw capture rows.
echo "[WRAPPER] Building strict rectified training manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u - <<'PY'
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

root = Path.cwd()
base_manifest = root / "data" / "full_scalar_manifest_v1.csv"
held_out_manifest = root / "data" / "hard_cases_plus_board30_valid_with_new5.csv"
out_manifest = root / "data" / "rectified_scalar_strict_train_v5.csv"

# Keep only rows that already live under captured_images. This is the cleanest
# rectified-adjacent pool we have without collapsing the dataset to a tiny
# single-temperature slice.
captured_prefix = "ml/data/captured_images/"

with held_out_manifest.open("r", encoding="utf-8", newline="") as handle:
    held_out_rows = {
        (row["image_path"].replace("\\", "/")).strip()
        for row in csv.DictReader(handle)
    }

kept_rows: list[dict[str, str]] = []
with base_manifest.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        image_path = (row["image_path"].replace("\\", "/")).strip()
        if not image_path.startswith(captured_prefix):
            continue
        if image_path in held_out_rows:
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

print(f"[WRAPPER] Wrote {len(kept_rows)} strict rectified rows to {out_manifest}")
PY

echo "[WRAPPER] Launching training..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir artifacts/training/mobilenetv2_rectified_scalar_strict_v5 \
  --manifest-path "${TRAIN_MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${INIT_MODEL}" \
  --batch-size 4 \
  --epochs 16 \
  --warmup-epochs 6 \
  --learning-rate 1e-4 \
  --fine-tune-lr 2e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --seed 21 \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
