#!/usr/bin/env bash
set -euo pipefail

# Precompute geometry-derived crop boxes for the raw CVAT-labelled pool.
#
# We use the geometry model's keypoint predictions to tighten the crop around
# the dial before scalar fine-tuning. This gives the scalar model a more
# stable, geometry-informed framing than the original fixed crop alone.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
GEOMETRY_MODEL_SRC="${REPO_ROOT}/artifacts/training/mobilenetv2_geometry_raw_cvat_v18/model.keras"
GEOMETRY_MODEL_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_geometry_raw_cvat_v18.model.keras"
MANIFEST="${REPO_ROOT}/data/full_labelled_plus_board30_valid_with_new5.csv"
OUT_CSV="${REPO_ROOT}/data/geometry_crop_boxes_v18.csv"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/precompute_geometry_boxes_v18.log"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[PRECOMPUTE] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${GEOMETRY_MODEL_LOCAL}")"
cp -f "${GEOMETRY_MODEL_SRC}" "${GEOMETRY_MODEL_LOCAL}"
: > "${LOG_FILE}"

cd "${REPO_ROOT}"

echo "[PRECOMPUTE] Loading geometry model from ${GEOMETRY_MODEL_LOCAL}" | tee "${LOG_FILE}"
echo "[PRECOMPUTE] Reading manifest ${MANIFEST}" | tee -a "${LOG_FILE}"
echo "[PRECOMPUTE] Writing crop boxes to ${OUT_CSV}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
from __future__ import annotations

import csv
import sys
from pathlib import Path

import keras
import tensorflow as tf

sys.path.insert(0, "src")

from embedded_gauge_reading_tinyml.board_crop_compare import load_rgb_image
from embedded_gauge_reading_tinyml.models import GaugeValueFromKeypoints, SpatialSoftArgmax2D
from embedded_gauge_reading_tinyml.geometry_cascade import run_geometry_cascade

repo_root = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml/ml")
workspace_root = repo_root.parent
geometry_model_path = Path.home() / "ml_eval_cache" / "mobilenetv2_geometry_raw_cvat_v18.model.keras"
manifest_path = repo_root / "data" / "full_labelled_plus_board30_valid_with_new5.csv"
out_path = repo_root / "data" / "geometry_crop_boxes_v18.csv"

print(f"[PRECOMPUTE] Loading geometry model: {geometry_model_path}", flush=True)
model = keras.models.load_model(
    geometry_model_path,
    compile=False,
    safe_mode=False,
    custom_objects={
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
        "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
        "GaugeValueFromKeypoints": GaugeValueFromKeypoints,
    },
)
print("[PRECOMPUTE] Geometry model loaded.", flush=True)

rows: list[dict[str, str]] = []
with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    rows.extend(csv.DictReader(handle))

print(f"[PRECOMPUTE] Processing {len(rows)} labelled rows...", flush=True)

out_rows: list[dict[str, str]] = []
for index, row in enumerate(rows, start=1):
    image_path = row["image_path"].replace("\\", "/").strip()
    value = row["value"]
    if image_path.startswith("ml/"):
        full_path = workspace_root / image_path
    elif image_path.startswith("data/"):
        full_path = repo_root / image_path
    else:
        full_path = workspace_root / image_path

    src = load_rgb_image(full_path)
    height, width = src.shape[:2]
    base_crop = (0.0, 0.0, float(width), float(height))
    result = run_geometry_cascade(
        model=model,
        source_image=src,
        base_crop_box_xyxy=base_crop,
        image_height=224,
        image_width=224,
        input_size=224,
        confidence_threshold=2.0,
        recrop_scale=0.75,
        min_recrop_size=64.0,
    )

    crop_box = result.second_pass.crop_box_xyxy if result.second_pass else result.first_pass.crop_box_xyxy
    x0, y0, x1, y1 = crop_box
    out_rows.append(
        {
            "image_path": image_path,
            "value": value,
            "x0": f"{float(x0):.2f}",
            "y0": f"{float(y0):.2f}",
            "x1": f"{float(x1):.2f}",
            "y1": f"{float(y1):.2f}",
        }
    )

    if index % 25 == 0 or index == len(rows):
        print(f"[PRECOMPUTE] {index}/{len(rows)}", flush=True)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "x0", "y0", "x1", "y1"])
    writer.writeheader()
    writer.writerows(out_rows)

print(f"[PRECOMPUTE] Wrote {len(out_rows)} rows to {out_path}", flush=True)
PY
