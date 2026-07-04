#!/usr/bin/env python3
"""Generate OBB-cropped geometry training manifest from the deployment OBB model.

Runs the OBB int8 TFLite model on every image in the source geometry manifest,
decodes the crop box with the same firmware-aligned decoder, remaps center/tip
labels into the OBB crop coordinate space, and writes a new CSV manifest where
every row carries OBB crop coordinates (not the original loose crops).

This closes the domain gap between training and deployment: models trained on
OBB-cropped images see the same crops they will receive in production.

Usage:
    poetry run python scripts/generate_obb_geometry_manifest.py \
        --source-manifest ml/data/geometry_board_heatmap_manifest_v2_fixed.csv \
        --obb-model tmp/obb_box_board_bbox_deploy_candidate/model_int8.tflite \
        --output-manifest tmp/obb_geometry_manifest_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    SourceGeometryExample,
    load_geometry_manifest,
)
from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)


def _load_yuv422_as_rgb(image_path: Path, width: int, height: int) -> np.ndarray:
    raw = image_path.read_bytes()
    expected = height * (width // 2) * 4
    if len(raw) < expected:
        raise ValueError(f"{image_path} too small for {width}x{height}")
    yuyv = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width // 2, 4)
    luma = np.empty((height, width), dtype=np.uint8)
    luma[:, 0::2] = yuyv[:, :, 0]
    luma[:, 1::2] = yuyv[:, :, 2]
    return np.repeat(luma[:, :, None], 3, axis=2)


def _load_source_rgb(image_path: str, sw: int, sh: int) -> np.ndarray:
    path = Path(image_path)
    if path.suffix.lower() == ".yuv422":
        return _load_yuv422_as_rgb(path, sw, sh)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _preprocess_for_obb(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize to target, normalize to [0, 1] (OBB model contract)."""
    img = Image.fromarray(image).resize((target_size, target_size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_obb_model(model_path: Path) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    return interpreter


def _run_obb(
    interpreter: tf.lite.Interpreter,
    image: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Run OBB model on a full-frame image.

    Returns:
        Tuple of (confidence, box_params[4]) where box_params = [cx, cy, w, h].
        All values are float32, normalised.
    """
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    img_batch = np.expand_dims(image, axis=0).astype(np.float32)
    if input_details["dtype"] == np.int8:
        scale, zp = input_details["quantization"]
        img_batch = np.clip(
            np.round(img_batch / scale + zp), -128, 127,
        ).astype(np.int8)
    interpreter.set_tensor(int(input_details["index"]), img_batch)
    interpreter.invoke()

    # Output 0 = confidence (scalar), Output 1 = box [cx, cy, w, h].
    conf_raw = interpreter.get_tensor(int(output_details[0]["index"]))
    box_raw = interpreter.get_tensor(int(output_details[1]["index"]))

    box_f32 = np.asarray(box_raw, dtype=np.float32)
    conf_f32 = np.asarray(conf_raw, dtype=np.float32)

    # Dequantize if int8.
    for details, arr in [
        (output_details[1], box_f32),
        (output_details[0], conf_f32),
    ]:
        scale_out, zp_out = details["quantization"]
        if scale_out != 0.0:
            arr[:] = (arr - float(zp_out)) * float(scale_out)

    return float(conf_f32.flatten()[0]), box_f32.flatten()


def _clamp_norm(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _source_xy_from_canvas(
    canvas_x: float, canvas_y: float,
    source_width: int, source_height: int,
    image_size: int = 224,
) -> tuple[float, float]:
    """Map a canvas-space point to source-image coordinates.

    Mirrors the firmware's `source_xy_from_resized_xy` logic: the image is
    resized to fit within the model input while preserving aspect ratio,
    then zero-padded.  We invert this transform.
    """
    crop_w = float(source_width)
    crop_h = float(source_height)
    scale = min(float(image_size) / crop_w, float(image_size) / crop_h)
    resized_w = crop_w * scale
    resized_h = crop_h * scale
    pad_x = 0.5 * (float(image_size) - resized_w)
    pad_y = 0.5 * (float(image_size) - resized_h)
    sx = ((canvas_x - pad_x) / scale)
    sy = ((canvas_y - pad_y) / scale)
    return max(0.0, sx), max(0.0, sy)


def _obb_to_square_crop(
     obb_box: np.ndarray,
     source_width: int,
     source_height: int,
     input_size: int = 224,
     square_scale: float = 1.0,
) -> tuple[float, float, float, float]:
    """Decode OBB box to a SQUARE source-space crop centered on the gauge.

    Expands the OBB box to a square (max of width and height), scaled by
    square_scale for margin.  When this square crop is resized to 224×224,
    there is no aspect-ratio distortion — matching the firmware's
    resize-with-pad contract.

    Returns source-space (x1, y1, x2, y2) where x2-x1 == y2-y1 (square).
    """
    cx = float(np.clip(obb_box[0], 0.0, 1.0))
    cy = float(np.clip(obb_box[1], 0.0, 1.0))
    bw = max(0.05, min(1.0, float(obb_box[2])))
    bh = max(0.05, min(1.0, float(obb_box[3])))

    # Canvas-space center and half-size.
    canvas_cx = cx * float(input_size)
    canvas_cy = cy * float(input_size)
    # Use max half-size to create a square box, scaled for margin.
    half_size = max(bw, bh) * float(input_size) * 1.20 * square_scale * 0.5

    # Map canvas corners to source coordinates.
    corners = []
    for dx, dy in [(-half_size, -half_size), (half_size, -half_size),
                   (half_size, half_size), (-half_size, half_size)]:
        sx, sy = _source_xy_from_canvas(
            canvas_cx + dx, canvas_cy + dy,
            source_width, source_height, input_size,
        )
        corners.append((sx, sy))

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(source_width), max(xs))
    y2 = min(float(source_height), max(ys))

    # Enforce square: compute center, use max side length.
    cx_src = 0.5 * (x1 + x2)
    cy_src = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1, 48.0)
    x1 = max(0.0, cx_src - 0.5 * side)
    y1 = max(0.0, cy_src - 0.5 * side)
    x2 = min(float(source_width), x1 + side)
    y2 = min(float(source_height), y1 + side)
    # Re-center if clipped.
    if x2 - x1 < side:
        x1 = max(0.0, x2 - side)
    if y2 - y1 < side:
        y1 = max(0.0, y2 - side)

    return x1, y1, x2, y2


def build_obb_manifest(
    source_manifest: Path,
    obb_model_path: Path,
    output_path: Path,
    repo_root: Path,
    square_scale: float = 1.10,
) -> None:
    """Run OBB on every image and build a new manifest with OBB crop boxes."""
    examples = load_geometry_manifest(source_manifest)
    print(f"Loaded {len(examples)} source examples")

    interpreter = _load_obb_model(obb_model_path)

    output_rows = []
    success = 0
    failed = 0
    total_obb = 0

    for i, ex in enumerate(examples):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(examples)}... ({success} ok, {failed} failed)")

        # Resolve image path.
        img_path = ex.image_path
        if not img_path.startswith("/"):
            if img_path.startswith("ml/"):
                img_path = str(repo_root / img_path)
            else:
                img_path = str(repo_root / "ml" / img_path)

        if not Path(img_path).exists():
            failed += 1
            continue

        try:
            source_img = _load_source_rgb(
                img_path, ex.source_width, ex.source_height,
            )
        except Exception:
            failed += 1
            continue

        # Run OBB.
        preprocessed = _preprocess_for_obb(source_img)
        try:
            obb_conf, obb_box = _run_obb(interpreter, preprocessed)
            total_obb += 1
        except Exception:
            failed += 1
            continue

        # Decode OBB to square crop (no aspect-ratio distortion).
        crop_x1, crop_y1, crop_x2, crop_y2 = _obb_to_square_crop(
            obb_box, ex.source_width, ex.source_height,
            square_scale=1.0,
        )

        # Validate crop bounds.
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        if crop_w <= 0 or crop_h <= 0:
            failed += 1
            continue

        # Remap center/tip to OBB crop space (square crop → single scale).
        cx_crop = ex.center_x_source - crop_x1
        cy_crop = ex.center_y_source - crop_y1
        tx_crop = ex.tip_x_source - crop_x1
        ty_crop = ex.tip_y_source - crop_y1

        # Check that keypoints are inside or near the crop.
        margin = max(20.0, 0.15 * max(crop_w, crop_h))
        if (cx_crop < -margin or cx_crop > crop_w + margin or
                cy_crop < -margin or cy_crop > crop_h + margin):
            failed += 1
            continue

        success += 1

        # Compute angle/temperature from OBB-space 224 coords (single scale).
        scale_224 = 224.0 / max(crop_w, crop_h)
        cx_224 = cx_crop * scale_224
        cy_224 = cy_crop * scale_224
        tx_224 = tx_crop * scale_224
        ty_224 = ty_crop * scale_224
        angle = angle_degrees_from_center_to_tip(cx_224, cy_224, tx_224, ty_224)
        det_temp = celsius_from_inner_dial_angle_degrees(angle)

        output_rows.append({
            "image_path": ex.image_path,
            "temperature_c": str(ex.temperature_c) if not np.isnan(ex.temperature_c) else "nan",
            "split": ex.split,
            "source_width": ex.source_width,
            "source_height": ex.source_height,
            "obb_crop_x1": f"{crop_x1:.2f}",
            "obb_crop_y1": f"{crop_y1:.2f}",
            "obb_crop_x2": f"{crop_x2:.2f}",
            "obb_crop_y2": f"{crop_y2:.2f}",
            "center_x_source": ex.center_x_source,
            "center_y_source": ex.center_y_source,
            "tip_x_source": ex.tip_x_source,
            "tip_y_source": ex.tip_y_source,
            "center_x_crop": f"{cx_crop:.4f}",
            "center_y_crop": f"{cy_crop:.4f}",
            "tip_x_crop": f"{tx_crop:.4f}",
            "tip_y_crop": f"{ty_crop:.4f}",
            "dial_radius_source": ex.dial_radius_source if ex.dial_radius_source else 0.0,
            "label_quality": ex.label_quality,
            "source_manifest": ex.source_manifest,
            "notes": ex.notes,
            "angle_degrees_from_labels": angle,
            "deterministic_temperature_c": det_temp,
            "center_tip_distance_pixels": ex.center_tip_distance_pixels if ex.center_tip_distance_pixels else "nan",
            "quality_flag": ex.quality_flag,
            "obb_accepted": "true",
        })

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nDone: {success} rows written, {failed} failed, {total_obb} OBB runs")
    print(f"Output: {output_path}")

    # Collect split stats.
    splits: dict[str, int] = {}
    board_rows = 0
    for r in output_rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
        if "PXL" not in r["image_path"]:
            board_rows += 1
    print(f"Splits: {splits}")
    print(f"Board rows: {board_rows}, PXL rows: {len(output_rows) - board_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OBB-cropped geometry training manifest.",
    )
    parser.add_argument(
        "--source-manifest", type=str, required=True,
        help="Source CSV geometry manifest.",
    )
    parser.add_argument(
        "--obb-model", type=str, required=True,
        help="Path to OBB int8 TFLite model.",
    )
    parser.add_argument(
        "--output-manifest", type=str, required=True,
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--square-scale", type=float, default=1.10,
        help="Scale factor for square OBB crop expansion (default 1.10).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    source_manifest = Path(args.source_manifest)
    if not source_manifest.is_absolute():
        source_manifest = repo_root / source_manifest

    obb_model = Path(args.obb_model)
    if not obb_model.is_absolute():
        obb_model = repo_root / obb_model

    output_manifest = Path(args.output_manifest)
    if not output_manifest.is_absolute():
        output_manifest = repo_root / output_manifest

    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    build_obb_manifest(
        source_manifest, obb_model, output_manifest, repo_root,
        square_scale=args.square_scale,
    )


if __name__ == "__main__":
    main()
