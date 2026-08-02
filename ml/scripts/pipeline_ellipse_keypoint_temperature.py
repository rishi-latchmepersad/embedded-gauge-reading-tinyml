#!/usr/bin/env python3
"""End-to-end gauge temperature pipeline: ellipse -> crop -> center/tip -> temp.

Runs the two deployed int8 models in sequence on board-capture images:

1. ``ellipse_iter8_universal_wide_deep`` (384x384 input) finds the gauge
   face ellipse (cx, cy, rx, ry).
2. A 1.35x square crop around the ellipse is resized to 224x224.
3. ``keypoint_unet_224g_stride2`` (224x224 -> 112x112 heatmaps) predicts the
   needle center and tip.
4. The tip-center vector is converted to an angle, then to a temperature via
   the LittleGood gauge calibration (min_deg=135, sweep_deg=270,
   -30C..+50C, clockwise).

Ground truth is read from filenames like ``capture_p42c.jpg`` (+42C) and
``capture_m10c.png`` (-10C); images without a temperature in the name are
skipped for the error report but still decoded.

Usage:
    python scripts/pipeline_ellipse_keypoint_temperature.py \
        --ellipse artifacts/ellipse_iter8_universal_wide_deep/model_int8.tflite \
        --keypoint artifacts/keypoint_unet_224g_stride2/model_int8.tflite \
        --images data/labelled/test_3.zip
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from train_ellipse_multiscale_universal_384 import (  # noqa: E402
    IMAGE_SIZE as ELLIPSE_SIZE,
    CONFIDENCE_OFFSET,
    GEOMETRY_OFFSET,
    MAP_VALUES,
    SIZES,
    predict_int8 as ellipse_predict_int8,
)

ELLIPSE_CROP_SCALE = 1.35  # must match the keypoint data prep
KEYPOINT_INPUT = 224
HEATMAP_SIZE = 112  # stride-2 output

# LittleGood home temp gauge calibration (from git history of the
# gauge_calibration_parameters.toml; the current file was overwritten by a
# firmware-specific spec).
LITTLEGOOD_MIN_DEG = 135.0
LITTLEGOOD_SWEEP_DEG = 270.0
LITTLEGOOD_MIN_VALUE = -30.0
LITTLEGOOD_MAX_VALUE = 50.0

_TEMP_RE = re.compile(r"capture_([mp])(\d+)c")


def _temperature_from_name(name: str) -> float | None:
    """Parse a board-capture temperature from its filename, if present."""
    match = _TEMP_RE.search(name)
    if match is None:
        return None
    value = float(match.group(2))
    return -value if match.group(1) == "m" else value


def _load_test_zip(zip_path: Path) -> list[tuple[str, Image.Image, float | None]]:
    """Load every image in a CVAT test zip with its name and GT temperature."""
    items: list[tuple[str, Image.Image, float | None]] = []
    with zipfile.ZipFile(zip_path) as archive:
        root = ET.fromstring(archive.read("annotations.xml"))
        names = {Path(n).name: n for n in archive.namelist()}
        for image_node in root.findall("image"):
            name = image_node.get("name", "")
            member = names.get(Path(name).name)
            if member is None:
                continue
            image = Image.open(io.BytesIO(archive.read(member))).convert("L")
            items.append((Path(name).name, image, _temperature_from_name(name)))
    return items


def _load_image_dir(image_dir: Path) -> list[tuple[str, Image.Image, float | None]]:
    """Load every supported image in a directory (no CVAT annotations needed)."""
    items: list[tuple[str, Image.Image, float | None]] = []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image = Image.open(path).convert("L")
        items.append((path.name, image, _temperature_from_name(path.name)))
    return items


class EllipseModel:
    """Wrapper around the int8 ellipse detector (universal contract)."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    def detect(self, image: Image.Image) -> tuple[float, float, float, float, float]:
        """Return (cx, cy, rx, ry, confidence) in normalized [0,1] coords."""
        # letterbox to 384x384 like the ellipse training pipeline
        gray = np.asarray(image, dtype=np.float32)
        h, w = gray.shape
        scale = min(ELLIPSE_SIZE / w, ELLIPSE_SIZE / h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = np.asarray(
            Image.fromarray(gray.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        canvas = np.full((ELLIPSE_SIZE, ELLIPSE_SIZE), float(gray.mean()), dtype=np.float32)
        pad_x = (ELLIPSE_SIZE - new_w) // 2
        pad_y = (ELLIPSE_SIZE - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
        image_tensor = (canvas / 255.0)[None, ..., None].astype(np.float32)

        contract = ellipse_predict_int8(self.model_path, image_tensor)[0]
        confidence = contract[CONFIDENCE_OFFSET : CONFIDENCE_OFFSET + 3]
        selected = int(np.argmax(confidence))
        size = SIZES[selected]
        values = size * size
        center_start = sum(2 * v for v in MAP_VALUES[:selected])
        center_map = contract[center_start : center_start + values].reshape(size, size)
        coords = (np.arange(size, dtype=np.float32) + 0.5) / size
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        weights = np.maximum(center_map - 0.05, 0.0) ** 4.0
        total = max(float(weights.sum()), 1e-6)
        cx = float((weights * xx).sum() / total)
        cy = float((weights * yy).sum() / total)
        geo = contract[GEOMETRY_OFFSET + 4 * selected : GEOMETRY_OFFSET + 4 * selected + 4]
        rx, ry = float(geo[2]), float(geo[3])
        # why: the ellipse contract predicts normalized coords in the
        # letterboxed 384 canvas; undo the letterbox padding.
        x_norm = (ELLIPSE_SIZE - 2 * pad_x) / ELLIPSE_SIZE
        y_norm = (ELLIPSE_SIZE - 2 * pad_y) / ELLIPSE_SIZE
        cx = (cx - pad_x / ELLIPSE_SIZE) / x_norm
        cy = (cy - pad_y / ELLIPSE_SIZE) / y_norm
        rx = rx / x_norm
        ry = ry / y_norm
        return cx, cy, rx, ry, float(confidence[selected])


class KeypointModel:
    """Wrapper around the int8 stride-2 center/tip model (112x112 heatmaps)."""

    def __init__(self, model_path: Path) -> None:
        self.interp = tf.lite.Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.in_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        self.in_scale, self.in_zero = self.in_det["quantization"]
        self.out_scale, self.out_zero = self.out_det["quantization"]

    def predict(self, crop: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """Return (center, tip) in 224-crop pixels plus peak confidences."""
        x = (crop / 255.0).astype(np.float32)
        xq = np.clip(np.round(x[None, ..., None] / self.in_scale + self.in_zero), -128, 127).astype(np.int8)
        self.interp.set_tensor(self.in_det["index"], xq)
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out_det["index"]).astype(np.float32)
        heatmaps = ((raw[0] - self.out_zero) * self.out_scale)
        size = heatmaps.shape[0]
        center = self._decode(heatmaps[..., 0])
        tip = self._decode(heatmaps[..., 1])
        return center, tip, float(heatmaps[..., 0].max()), float(heatmaps[..., 1].max())

    @staticmethod
    def _decode(heatmap: np.ndarray) -> tuple[float, float]:
        """Soft-argmax decode to (x, y) in 224-crop pixels."""
        size = heatmap.shape[0]
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        weights = np.maximum(heatmap - 0.05, 0.0) ** 4.0
        total = max(float(weights.sum()), 1e-6)
        x = float((weights * xx).sum() / total)
        y = float((weights * yy).sum() / total)
        return x * (KEYPOINT_INPUT / size), y * (KEYPOINT_INPUT / size)


def crop_ellipse(
    image: Image.Image,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Crop a square around the ellipse (1.35x) and return 224x224 uint8 + mapping.

    Returns (crop, (crop_left, crop_top, actual_side)) in source pixels so
    crop-local coordinates can be mapped back.
    """
    width, height = image.size
    side = max(2.0 * rx * width, 2.0 * ry * height) * ELLIPSE_CROP_SCALE
    left = cx * width - side / 2.0
    top = cy * height - side / 2.0
    crop_left = max(0, int(left))
    crop_top = max(0, int(top))
    crop_right = min(width, int(left + side))
    crop_bottom = min(height, int(top + side))
    actual_side = max(crop_right - crop_left, crop_bottom - crop_top)
    cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    resized = cropped.resize((KEYPOINT_INPUT, KEYPOINT_INPUT), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8), (crop_left, crop_top, actual_side)


def angle_to_temperature(angle_deg: float) -> float:
    """Map a needle angle (degrees, image coords) to Celsius for LittleGood."""
    # Normalize into the gauge sweep starting at min_deg (clockwise).
    shifted = (angle_deg - LITTLEGOOD_MIN_DEG) % 360.0
    if shifted > LITTLEGOOD_SWEEP_DEG:
        shifted = LITTLEGOOD_SWEEP_DEG
    fraction = shifted / LITTLEGOOD_SWEEP_DEG
    return LITTLEGOOD_MIN_VALUE + fraction * (LITTLEGOOD_MAX_VALUE - LITTLEGOOD_MIN_VALUE)


def run_pipeline(
    ellipse_model: EllipseModel,
    keypoint_model: KeypointModel,
    image: Image.Image,
) -> dict[str, float]:
    """Run both models on one image and return geometry + temperature."""
    cx, cy, rx, ry, ellipse_conf = ellipse_model.detect(image)
    crop, (crop_left, crop_top, actual_side) = crop_ellipse(image, cx, cy, rx, ry)
    (pcx, pcy), (ptx, pty), center_conf, tip_conf = keypoint_model.predict(crop)

    # Map crop pixels back to source pixels.
    scale = actual_side / KEYPOINT_INPUT
    center_src = (crop_left + pcx * scale, crop_top + pcy * scale)
    tip_src = (crop_left + ptx * scale, crop_top + pty * scale)

    # Needle angle in image coords (atan2 with image y-down; clockwise positive
    # in image coords matches the gauge calibration).
    dx = tip_src[0] - center_src[0]
    dy = tip_src[1] - center_src[1]
    angle_deg = math.degrees(math.atan2(dy, dx))
    temperature = angle_to_temperature(angle_deg)

    return {
        "ellipse_cx": cx,
        "ellipse_cy": cy,
        "ellipse_rx": rx,
        "ellipse_ry": ry,
        "ellipse_conf": ellipse_conf,
        "center_x": center_src[0],
        "center_y": center_src[1],
        "tip_x": tip_src[0],
        "tip_y": tip_src[1],
        "center_conf": center_conf,
        "tip_conf": tip_conf,
        "angle_deg": angle_deg,
        "temperature_c": temperature,
    }


def main() -> None:
    """Run the pipeline over a test zip and report per-image temperatures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ellipse", type=Path, required=True)
    parser.add_argument("--keypoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True,
                        help="CVAT zip (e.g. data/labelled/test_3.zip) or a "
                             "directory of images")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ellipse_model = EllipseModel(args.ellipse)
    keypoint_model = KeypointModel(args.keypoint)
    items = (
        _load_image_dir(args.images)
        if args.images.is_dir()
        else _load_test_zip(args.images)
    )
    if args.limit:
        items = items[: args.limit]

    rows: list[dict[str, object]] = []
    errors: list[float] = []
    fieldnames: set[str] = {
        "image", "gt_temp_c", "ellipse_cx", "ellipse_cy", "ellipse_rx",
        "ellipse_ry", "ellipse_conf", "center_x", "center_y", "tip_x",
        "tip_y", "center_conf", "tip_conf", "angle_deg", "temperature_c",
        "abs_error_c",
    }
    for name, image, gt_temp in items:
        result = run_pipeline(ellipse_model, keypoint_model, image)
        row = {"image": name, "gt_temp_c": gt_temp, **result}
        rows.append(row)
        if gt_temp is not None:
            error = result["temperature_c"] - gt_temp
            errors.append(abs(error))
            row["abs_error_c"] = abs(error)
        print(
            f"{name:38s} ellipse_conf={result['ellipse_conf']:.2f} "
            f"angle={result['angle_deg']:6.1f}deg -> temp={result['temperature_c']:6.1f}C"
            + (f"  GT={gt_temp:+.0f}C  err={row.get('abs_error_c', float('nan')):5.1f}" if gt_temp is not None else "")
        )

    if errors:
        errors_arr = np.asarray(errors)
        print(f"\n=== SUMMARY ({len(errors)} images with GT temperature) ===")
        print(f"MAE: {errors_arr.mean():.2f}C   Median: {np.median(errors_arr):.2f}C   "
              f"Max: {errors_arr.max():.2f}C   ≤2C: {(errors_arr <= 2).mean() * 100:.0f}%   "
              f"≤5C: {(errors_arr <= 5).mean() * 100:.0f}%")

    output = args.images / f"{args.images.stem}_pipeline_results.csv" if args.images.is_dir() else args.images.parent / f"{args.images.stem}_pipeline_results.csv"
    import csv

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")

if __name__ == "__main__":
    main()
