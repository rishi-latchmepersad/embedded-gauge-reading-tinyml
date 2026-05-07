#!/usr/bin/env bash
set -euo pipefail

# Train the strongest full-manifest scalar model on rectifier-aligned crops.
# This keeps the model deployment-friendly while giving it the best board-style
# framing we currently know how to produce from the GPU rectifier.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/full_rectified_scalar_full_manifest.log"

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

RECTIFIER_SRC="artifacts/training/mobilenetv2_rectifier_hardcase_finetune_v3/model.keras"
RECTIFIER_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_rectifier_hardcase_finetune_v3.model.keras"
MANIFEST="data/full_scalar_manifest_v1.csv"
BOXES_CSV="data/rectified_crop_boxes_full_scalar_v1.csv"
OUTPUT_DIR="artifacts/training/full_rectified_scalar_full_manifest_v1"
INIT_MODEL="artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras"

cp -f "${RECTIFIER_SRC}" "${RECTIFIER_LOCAL}"

echo "[WRAPPER] Starting full-manifest rectified scalar training." | tee "${LOG_FILE}"
echo "[WRAPPER] Manifest:  ${MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Rectifier:  ${RECTIFIER_LOCAL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:  ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init model: ${INIT_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

# Build the rectifier crop boxes on CPU so the GPU stays free for training.
CUDA_VISIBLE_DEVICES="" "${POETRY_BIN}" run python -u - <<'PYEOF'
from __future__ import annotations

import csv
from pathlib import Path
import os
import sys

import numpy as np
import tensorflow as tf

ROOT_DIR = Path("..").resolve()
MANIFEST = ROOT_DIR / "ml" / "data" / "full_scalar_manifest_v1.csv"
RECTIFIER_LOCAL = Path(os.environ["HOME"]) / "ml_eval_cache" / "mobilenetv2_rectifier_hardcase_finetune_v3.model.keras"
OUT_CSV = ROOT_DIR / "ml" / "data" / "rectified_crop_boxes_full_scalar_v1.csv"
IMAGE_SIZE = 224
CROP_SCALE = 1.80

sys.path.insert(0, str(ROOT_DIR / "ml" / "src"))

from embedded_gauge_reading_tinyml.board_crop_compare import load_rgb_image, resize_with_pad_rgb

print(f"[PRECOMPUTE] Loading rectifier from {RECTIFIER_LOCAL}", flush=True)
rectifier = tf.keras.models.load_model(str(RECTIFIER_LOCAL), compile=False)
print("[PRECOMPUTE] Rectifier loaded.", flush=True)

rows: list[dict[str, str]] = []
with MANIFEST.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        rows.append(row)

print(f"[PRECOMPUTE] Processing {len(rows)} manifest rows...", flush=True)
out_rows: list[dict[str, str]] = []

for index, row in enumerate(rows, start=1):
    image_path = Path(row["image_path"])
    if not image_path.is_absolute():
        image_path = ROOT_DIR / image_path
    source_image = load_rgb_image(image_path)
    source_h, source_w = source_image.shape[:2]

    full_frame = resize_with_pad_rgb(
        source_image,
        (0.0, 0.0, float(source_w), float(source_h)),
        image_size=IMAGE_SIZE,
    )
    rectifier_batch = np.expand_dims(full_frame.astype(np.float32) / 255.0, axis=0)
    rectifier_pred = rectifier.predict(rectifier_batch, verbose=0)
    if isinstance(rectifier_pred, dict):
        rectifier_box = np.asarray(rectifier_pred["rectifier_box"]).reshape(-1)
    else:
        rectifier_box = np.asarray(rectifier_pred).reshape(-1)

    center_x = float(np.clip(rectifier_box[0], 0.0, 1.0))
    center_y = float(np.clip(rectifier_box[1], 0.0, 1.0))
    box_w = min(1.0, float(np.clip(rectifier_box[2], 0.05, 1.0)) * CROP_SCALE)
    box_h = min(1.0, float(np.clip(rectifier_box[3], 0.05, 1.0)) * CROP_SCALE)

    if box_w < 0.25 or box_h < 0.25 or box_w > 0.95 or box_h > 0.95:
        x0, y0, x1, y1 = 0.1027 * IMAGE_SIZE, 0.2573 * IMAGE_SIZE, 0.7987 * IMAGE_SIZE, 0.8071 * IMAGE_SIZE
    else:
        canvas_w = float(IMAGE_SIZE)
        canvas_h = float(IMAGE_SIZE)
        x0c = max(0.0, (center_x - 0.5 * box_w) * canvas_w)
        y0c = max(0.0, (center_y - 0.5 * box_h) * canvas_h)
        x1c = min(canvas_w, (center_x + 0.5 * box_w) * canvas_w)
        y1c = min(canvas_h, (center_y + 0.5 * box_h) * canvas_h)
        scale = min(canvas_w / source_w, canvas_h / source_h)
        pad_x = (canvas_w - source_w * scale) * 0.5
        pad_y = (canvas_h - source_h * scale) * 0.5
        x0 = max(0.0, (x0c - pad_x) / scale)
        y0 = max(0.0, (y0c - pad_y) / scale)
        x1 = min(float(source_w), (x1c - pad_x) / scale)
        y1 = min(float(source_h), (y1c - pad_y) / scale)
        if x1 <= x0 + 1.0:
            x1 = min(float(source_w), x0 + 1.0)
        if y1 <= y0 + 1.0:
            y1 = min(float(source_h), y0 + 1.0)

    out_rows.append(
        {
            "image_path": row["image_path"],
            "value": row["value"],
            "x0": f"{x0:.2f}",
            "y0": f"{y0:.2f}",
            "x1": f"{x1:.2f}",
            "y1": f"{y1:.2f}",
        }
    )

    if index % 25 == 0 or index == len(rows):
        print(f"[PRECOMPUTE] {index}/{len(rows)}", flush=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "x0", "y0", "x1", "y1"])
    writer.writeheader()
    writer.writerows(out_rows)

print(f"[PRECOMPUTE] Written {len(out_rows)} rows to {OUT_CSV}", flush=True)
PYEOF

echo "[WRAPPER] Launching training..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${INIT_MODEL}" \
  --epochs 30 \
  --warmup-epochs 8 \
  --batch-size 4 \
  --learning-rate 5e-6 \
  --fine-tune-lr 1e-6 \
  --alpha 1.0 \
  --dropout 0.2 \
  --seed 21 \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
