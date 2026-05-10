#!/usr/bin/env bash
set -euo pipefail

# Fine-tune a rectified scalar MobileNetV2 on the full CVAT-labelled raw pool.
#
# This experiment keeps the 352 raw CVAT images from the broad labelled export,
# excludes the held-out hard-case manifest, and uses the stronger preview-heavy
# augmentation profile with a linear output head.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_full_labelled_v18.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
BASE_MODEL_SRC="${ROOT_DIR}/artifacts/training/mobilenetv2_synthetic_hard_pretrain_v12/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_synthetic_hard_pretrain_v12.model.keras"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${BASE_MODEL_LOCAL}")"
cp -f "${BASE_MODEL_SRC}" "${BASE_MODEL_LOCAL}"
: > "${LOG_FILE}"

cd "${ROOT_DIR}"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

TRAIN_MANIFEST="data/rectified_scalar_full_labelled_v18.csv"
FULL_MANIFEST="data/full_labelled_plus_board30_valid_with_new5.csv"
HELD_OUT_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"
BOXES_CSV="data/rectified_crop_boxes_v5_all.csv"
OUTPUT_DIR="artifacts/training/mobilenetv2_rectified_scalar_full_labelled_v18"

echo "[WRAPPER] Starting full-labelled rectified fine-tune v18." | tee "${LOG_FILE}"
echo "[WRAPPER] Train manifest: ${TRAIN_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Source manifest: ${FULL_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Held-out eval:   ${HELD_OUT_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:       ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Base model:      ${BASE_MODEL_LOCAL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:        ${LOG_FILE}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Rebuilding full-labelled training manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u - <<'PY'
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

root = Path.cwd()
source_manifest = root / "data" / "full_labelled_plus_board30_valid_with_new5.csv"
held_out_manifest = root / "data" / "hard_cases_plus_board30_valid_with_new5.csv"
out_manifest = root / "data" / "rectified_scalar_full_labelled_v18.csv"
captured_prefix = "ml/data/raw/"

with held_out_manifest.open("r", encoding="utf-8", newline="") as handle:
    held_out_rows = {
        (row["image_path"].replace("\\", "/")).strip()
        for row in csv.DictReader(handle)
    }

kept_rows: list[dict[str, str]] = []
with source_manifest.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        image_path = (row["image_path"].replace("\\", "/")).strip()
        if image_path in held_out_rows:
            continue
        if not image_path.startswith(captured_prefix):
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

print(f"[WRAPPER] Wrote {len(kept_rows)} rows to {out_manifest}")
print(f"[WRAPPER] Unique labels: {len(value_counts)}")
PY

echo "[WRAPPER] Launching fine-tune..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${TRAIN_MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${BASE_MODEL_LOCAL}" \
  --linear-output \
  --augment-mode hard_preview \
  --batch-size 8 \
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
