#!/usr/bin/env bash
set -euo pipefail

# Sweep rectifier crop expansion factors and report manifest MAE for each one.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/rectified_scalar_sweep.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[SWEEP] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

cd "${ROOT_DIR}"
echo "[SWEEP] Starting rectified scalar crop-scale sweep." | tee "${LOG_FILE}"
echo "[SWEEP] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Sweep rectifier crop expansion factors against the board-style manifest."""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import tensorflow as tf

from embedded_gauge_reading_tinyml.board_crop_compare import (
    load_rgb_image,
    resize_with_pad_rgb,
)


def _load_items(manifest_path: Path, repo_root: Path) -> list[tuple[Path, float]]:
    """Load the labeled image paths used by the board-style evaluation."""
    items: list[tuple[Path, float]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = repo_root / image_path
            items.append((image_path, float(row["value"])))
    return items


def _quantize(batch: np.ndarray, input_details: dict[str, object]) -> np.ndarray:
    """Quantize a float batch to the TFLite input tensor format."""
    scale = float(input_details["quantization"][0])  # type: ignore[index]
    zero_point = int(input_details["quantization"][1])  # type: ignore[index]
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(batch / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(np.int8)


def _dequantize(output_tensor: np.ndarray, output_details: dict[str, object]) -> float:
    """Convert the TFLite output tensor back to a scalar prediction."""
    scale = float(output_details["quantization"][0])  # type: ignore[index]
    zero_point = int(output_details["quantization"][1])  # type: ignore[index]
    return float(scale * (int(output_tensor) - zero_point))


repo_root = Path.cwd().parent
rectifier_path = repo_root / "ml/artifacts/training/mobilenetv2_rectifier_hardcase_finetune_v3/model.keras"
scalar_path = repo_root / "ml/artifacts/deployment/mobilenetv2_rectified_scalar_finetune_v2_int8/model_int8.tflite"
manifest_path = repo_root / "ml/data/hard_cases_plus_board30_valid_with_new5.csv"
image_size = 224
scales = [0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50]

print(f"[SWEEP] Loading rectifier model from {rectifier_path}", flush=True)
rectifier = tf.keras.models.load_model(rectifier_path, compile=False, safe_mode=False)
print(f"[SWEEP] Rectifier loaded: {rectifier.name}", flush=True)

print(f"[SWEEP] Loading scalar reader from {scalar_path}", flush=True)
scalar_interpreter = tf.lite.Interpreter(model_path=str(scalar_path), num_threads=1)
scalar_interpreter.allocate_tensors()
input_details = scalar_interpreter.get_input_details()[0]
output_details = scalar_interpreter.get_output_details()[0]

items = _load_items(manifest_path, repo_root)
print(f"[SWEEP] Samples: {len(items)}", flush=True)

for scale in scales:
    errors: list[float] = []
    print(f"[SWEEP] scale={scale:.2f} start", flush=True)
    for image_path, true_value in items:
        source_image = load_rgb_image(image_path)
        full_frame = resize_with_pad_rgb(
            source_image,
            (
                0.0,
                0.0,
                float(source_image.shape[1]),
                float(source_image.shape[0]),
            ),
            image_size=image_size,
        )
        rectifier_batch = np.expand_dims(full_frame.astype(np.float32) / 255.0, axis=0)
        rectifier_pred = rectifier.predict(rectifier_batch, verbose=0)
        if isinstance(rectifier_pred, dict):
            rectifier_box = np.asarray(rectifier_pred["rectifier_box"]).reshape(-1)
        else:
            rectifier_box = np.asarray(rectifier_pred).reshape(-1)

        center_x = float(np.clip(rectifier_box[0], 0.0, 1.0))
        center_y = float(np.clip(rectifier_box[1], 0.0, 1.0))
        box_w = min(1.0, float(np.clip(rectifier_box[2], 0.05, 1.0)) * scale)
        box_h = min(1.0, float(np.clip(rectifier_box[3], 0.05, 1.0)) * scale)

        canvas_w = float(image_size)
        canvas_h = float(image_size)
        x_min = max(0.0, (center_x - 0.5 * box_w) * canvas_w)
        y_min = max(0.0, (center_y - 0.5 * box_h) * canvas_h)
        x_max = min(canvas_w, (center_x + 0.5 * box_w) * canvas_w)
        y_max = min(canvas_h, (center_y + 0.5 * box_h) * canvas_h)
        if x_max <= x_min + 1.0:
            x_max = min(canvas_w, x_min + 1.0)
        if y_max <= y_min + 1.0:
            y_max = min(canvas_h, y_min + 1.0)

        crop = resize_with_pad_rgb(
            full_frame,
            (x_min, y_min, x_max, y_max),
            image_size=image_size,
        )
        batch = np.expand_dims(crop.astype(np.float32) / 255.0, axis=0)
        quantized_batch = _quantize(batch, input_details)
        scalar_interpreter.set_tensor(int(input_details["index"]), quantized_batch)
        scalar_interpreter.invoke()
        raw_output = scalar_interpreter.get_tensor(int(output_details["index"]))[0][0]
        prediction = _dequantize(raw_output, output_details)
        errors.append(abs(prediction - true_value))

    print(
        f"[SWEEP] scale={scale:.2f} mean_abs_err={float(np.mean(errors)):.4f} "
        f"max_abs_err={float(np.max(errors)):.4f}",
        flush=True,
    )
PY
