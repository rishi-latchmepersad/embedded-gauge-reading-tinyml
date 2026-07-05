#!/usr/bin/env python3
"""Evaluate QARepVGG TFLite int8 model on image data.

Loads PXL photos + board captures (same pipeline as training) and computes
centre MAE (px), angle MAE (deg), box MAE (px).

Usage:
  cd ml && poetry run python scripts/eval_tflite_qarepvgg.py \
      --tflite artifacts/training/qat_qarepvgg_mini_20260613_203310/qarepvgg_mini_int8.tflite
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

os.environ["TF_GPU_MEMORY_LIMIT_MB"] = "1024"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import _compute_fullframe_obb_params

IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
HEATMAP_SIZE = 40


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    sh = max(int(h * scale), 1)
    sw = max(int(w * scale), 1)
    resized = tf.image.resize(image, [sh, sw], method="nearest").numpy()
    py = max(0, (height - sh) // 2)
    px = max(0, (width - sw) // 2)
    padded = np.pad(resized, [[py, height - sh - py], [px, width - sw - px], [0, 0]])
    return padded.astype(np.float32)


def _load_image(path: str) -> np.ndarray:
    if path.endswith(".yuv422"):
        raw = tf.io.read_file(path)
        yuyv = tf.io.decode_raw(raw, tf.uint8)
        yuyv = tf.reshape(yuyv, [320, 640])
        y = tf.cast(yuyv[:, 0::2], tf.float32)
        u = tf.cast(yuyv[:, 1::4], tf.float32) - 128.0
        v = tf.cast(yuyv[:, 3::4], tf.float32) - 128.0
        u = tf.repeat(u, 2, axis=1)
        v = tf.repeat(v, 2, axis=1)
        rgb = tf.stack([
            y + 1.402 * v,
            y - 0.344136 * u - 0.714136 * v,
            y + 1.772 * u,
        ], axis=-1)
        image = tf.clip_by_value(rgb, 0, 255).numpy().astype(np.uint8)
    else:
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False).numpy()
    return _resize_with_pad(image, IMAGE_HEIGHT, IMAGE_WIDTH)


def _load_examples() -> list[dict[str, Any]]:
    """Load examples from manifest (same pipeline as training)."""
    ML_ROOT = PROJECT_ROOT  # ml/
    examples = []
    seen: set[str] = set()

    manifest_path = PROJECT_ROOT / "data" / "merged_geometry_board_manifest.csv"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return examples

    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row["image_path"]
            # PXL photos
            if "PXL_" in image_path:
                pass  # include
            # Captured images
            elif "captured_images" in image_path:
                image_path = image_path.replace("/captured_images/", "/captured_images_320/")
                if image_path.endswith(".jpg"):
                    image_path = image_path[:-4] + ".png"
            else:
                continue

            qf = row.get("quality_flag", "")
            if qf not in ("clean", "manual", ""):
                continue

            fpath = str(PROJECT_ROOT.parent / image_path)
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)

            source_w = int(float(row["source_width"]))
            source_h = int(float(row["source_height"]))
            cx = float(row["center_x_source"])
            cy = float(row["center_y_source"])
            radius = float(row.get("dial_radius_source", 0))
            if radius <= 0:
                continue

            obb = _compute_fullframe_obb_params(
                source_w, source_h, cx, cy, radius, radius, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append({
                "path": fpath,
                "obb": obb,
            })
    return examples


def _run_tflite(interpreter, image: np.ndarray) -> dict[str, np.ndarray]:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantize input
    in_scale, in_zero = input_details[0]["quantization"]
    if in_scale != 0:
        image_q = np.clip(image / in_scale + in_zero, -128, 127).astype(np.int8)
    else:
        image_q = image.astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], image_q[np.newaxis, ...])
    interpreter.invoke()

    results = {}
    for det in output_details:
        out = interpreter.get_tensor(det["index"])
        q = det["quantization"]
        if q[0] != 0:
            out = (out.astype(np.float32) - q[1]) * q[0]
        shape = list(out.shape)
        if shape == [1, HEATMAP_SIZE, HEATMAP_SIZE, 1]:
            results["heatmap"] = out
        elif shape == [1, HEATMAP_SIZE, HEATMAP_SIZE, 2]:
            # Determine by quantization scale: tanh output has 1/128 scale
            if q[0] == 0.0078125:
                results["angle"] = out
            else:
                results["box_size"] = out
    return results


def _decode(preds: dict[str, np.ndarray]) -> dict[str, float]:
    hm = 1.0 / (1.0 + np.exp(-preds["heatmap"][0, :, :, 0]))
    box = preds["box_size"][0]
    ang = preds["angle"][0]

    # Softargmax
    h_sum = np.sum(hm)
    if h_sum <= 0:
        py, px = 0.0, 0.0
    else:
        hm_n = hm / h_sum
        ys, xs = np.meshgrid(np.arange(HEATMAP_SIZE), np.arange(HEATMAP_SIZE), indexing="ij")
        py = float(np.sum(hm_n * ys))
        px = float(np.sum(hm_n * xs))

    cx_hm = px / (HEATMAP_SIZE - 1)
    cy_hm = py / (HEATMAP_SIZE - 1)

    argmax_idx = np.argmax(hm)
    cell_y = argmax_idx // HEATMAP_SIZE
    cell_x = argmax_idx % HEATMAP_SIZE

    w_hm = float(box[cell_y, cell_x, 0])
    h_hm = float(box[cell_y, cell_x, 1])
    sin2t = float(ang[cell_y, cell_x, 0])
    cos2t = float(ang[cell_y, cell_x, 1])
    theta = 0.5 * np.arctan2(sin2t, cos2t)

    return {
        "cx_norm": cx_hm, "cy_norm": cy_hm,
        "w_norm": w_hm, "h_norm": h_hm,
        "angle_deg": float(np.degrees(theta)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()

    tflite_path = Path(args.tflite)
    if not tflite_path.exists():
        print(f"Not found: {tflite_path}"); sys.exit(1)

    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    print(f"  Input:  {in_det['shape']} quant={in_det['quantization']}")
    for d in interpreter.get_output_details():
        print(f"  Output: {d['name']:20s} {d['shape']} quant={d['quantization']}")
    print(f"  Model size: {tflite_path.stat().st_size / 1024:.1f} KB")

    print("Loading examples...")
    examples = _load_examples()
    print(f"  Found {len(examples)} examples")
    if args.max > 0:
        examples = examples[:args.max]
        print(f"  Using {args.max} examples")

    errors = []
    for i, ex in enumerate(examples):
        img = _load_image(ex["path"])
        preds = _run_tflite(interpreter, img)
        decoded = _decode(preds)

        cx_t, cy_t, w_t, h_t, c2t_t, s2t_t = ex["obb"]
        cx_p, cy_p = decoded["cx_norm"], decoded["cy_norm"]

        center_px = np.sqrt((cx_p - cx_t)**2 + (cy_p - cy_t)**2) * (IMAGE_WIDTH - 1)

        gt_angle = 0.5 * np.arctan2(s2t_t, c2t_t)
        pred_angle = decoded["angle_deg"]
        angle_diff = abs(gt_angle - np.radians(pred_angle))
        angle_deg = float(np.degrees(np.arctan2(np.sin(angle_diff), np.cos(angle_diff))))

        errors.append({"center_px": float(center_px), "angle_deg": angle_deg})

        if (i + 1) % 50 == 0:
            avg_c = np.mean([e["center_px"] for e in errors])
            avg_a = np.mean([e["angle_deg"] for e in errors])
            print(f"  [{i+1}/{len(examples)}] center={avg_c:.1f}px angle={avg_a:.2f}deg")

    arr_c = np.array([e["center_px"] for e in errors])
    arr_a = np.array([e["angle_deg"] for e in errors])
    metrics = {
        "n": len(errors),
        "center_mae_px": float(np.mean(arr_c)),
        "center_median_px": float(np.median(arr_c)),
        "center_p10_px": float(np.percentile(arr_c, 10)),
        "center_p90_px": float(np.percentile(arr_c, 90)),
        "angle_mae_deg": float(np.mean(arr_a)),
        "angle_median_deg": float(np.median(arr_a)),
    }

    print("\n" + "=" * 50)
    print(f"  QARepVGG TFLite int8 — {metrics['n']} examples")
    print(f"  Centre MAE:     {metrics['center_mae_px']:.2f} px")
    print(f"  Centre median:  {metrics['center_median_px']:.2f} px")
    print(f"  Centre P10/P90: {metrics['center_p10_px']:.2f} / {metrics['center_p90_px']:.2f} px")
    print(f"  Angle MAE:      {metrics['angle_mae_deg']:.2f} deg")
    print(f"  Angle median:   {metrics['angle_median_deg']:.2f} deg")
    print("=" * 50)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
