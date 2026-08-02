#!/usr/bin/env python3
"""Evaluate one int8 ellipse model independently on every labelled test zip.

The three test archives represent different deployment domains. Keeping their
metrics separate prevents the large generic split from hiding failures on the
small high-resolution and board-capture sets.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data" / "labelled"
IMAGE_SIZE = 384
FRAME_SIZE = 640


def _extract_ellipse(image_node: ET.Element) -> tuple[float, float, float, float] | None:
    """Return normalized gauge-face ellipse coordinates from one CVAT image."""
    width = float(image_node.get("width", FRAME_SIZE))
    height = float(image_node.get("height", FRAME_SIZE))
    for shape in image_node.findall("ellipse"):
        # why: the refreshed board-capture archive uses ``temp_dial`` while
        # the generic and phone archives use ``GaugeFace``.
        if shape.get("label") in {"GaugeFace", "temp_dial"}:
            return (
                float(shape.get("cx")) / width,
                float(shape.get("cy")) / height,
                float(shape.get("rx")) / width,
                float(shape.get("ry")) / height,
            )
    return None


def _load_zip(zip_name: str, image_size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Load one CVAT zip and resize its images to the model contract."""
    images: list[np.ndarray] = []
    # Store source-frame geometry plus the letterbox transform so metrics can
    # be reported in original 640px-equivalent coordinates.
    targets: list[tuple[float, ...]] = []
    with zipfile.ZipFile(LABELLED / zip_name) as archive:
        members_by_basename = {
            Path(member).name: member for member in archive.namelist()
        }
        root = ET.fromstring(archive.read("annotations.xml"))
        for image_node in root.findall("image"):
            ellipse = _extract_ellipse(image_node)
            if ellipse is None:
                continue
            name = image_node.get("name", "")
            member = members_by_basename.get(Path(name).name)
            if member is None:
                continue
            image = Image.open(io.BytesIO(archive.read(member))).convert("L")
            source_width, source_height = image.size
            scale = min(image_size / source_width, image_size / source_height)
            resized_size = (
                max(1, int(round(source_width * scale))),
                max(1, int(round(source_height * scale))),
            )
            resized = image.resize(resized_size, Image.Resampling.BILINEAR)
            canvas = Image.new("L", (image_size, image_size), color=int(np.asarray(image).mean()))
            pad_x = (image_size - resized_size[0]) // 2
            pad_y = (image_size - resized_size[1]) // 2
            canvas.paste(resized, (pad_x, pad_y))
            images.append(np.asarray(canvas, dtype=np.float32)[..., None] / 255.0)
            cx, cy, rx, ry = ellipse
            targets.append((
                cx,
                cy,
                rx,
                ry,
                1.0,
                source_width * scale / image_size,
                source_height * scale / image_size,
                pad_x / image_size,
                pad_y / image_size,
            ))
    return np.asarray(images, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _predict(interpreter: tf.lite.Interpreter, images: np.ndarray) -> np.ndarray:
    """Run an int8 interpreter and return dequantized five-value predictions."""
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_detail["quantization"]
    predictions = np.zeros((len(images), 5), dtype=np.float32)
    for index, image in enumerate(images):
        # why: quantize with the model contract instead of assuming [-128, 127].
        quantized = np.clip(
            np.round(image[None] / input_scale + input_zero_point), -128, 127
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        if len(output_details) == 1:
            output_detail = output_details[0]
            raw = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
            output_scale, output_zero_point = output_detail["quantization"]
            decoded = (raw[0] - output_zero_point) * output_scale
            if decoded.ndim == 3 and decoded.shape[-1] == 5:
                # Tiny spatial YOLO: argmax the face heatmap, then read the
                # center/radius regression at that cell.
                iy, ix = np.unravel_index(np.argmax(decoded[..., 0]), decoded[..., 0].shape)
                predictions[index, :4] = decoded[iy, ix, 1:]
                predictions[index, 0] = (ix + 0.5) / decoded.shape[1]
                predictions[index, 1] = (iy + 0.5) / decoded.shape[0]
                predictions[index, 4] = decoded[iy, ix, 0]
            else:
                predictions[index] = decoded
            continue

        # v2's three-head export reorders outputs to radius, confidence, center.
        # why: dimension alone cannot distinguish the two 2-value heads.
        by_name = {}
        for output_detail in output_details:
            raw = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
            output_scale, output_zero_point = output_detail["quantization"]
            by_name[output_detail["name"]] = (raw[0] - output_zero_point) * output_scale
        center = by_name.get("StatefulPartitionedCall:0")
        confidence = by_name.get("StatefulPartitionedCall:1")
        radius = by_name.get("StatefulPartitionedCall:2")
        if center is None or confidence is None or radius is None:
            # why: generated names can change, but the known v2 shape order is stable.
            ordered = [by_name[detail["name"]] for detail in output_details]
            radius, confidence, center = ordered
        predictions[index] = np.concatenate([center, radius, confidence])
    return predictions


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Compute center, radius, and validity metrics in original 640px units."""
    if targets.shape[1] >= 9:
        x_scale = targets[:, 5:6]
        y_scale = targets[:, 6:7]
        padding = targets[:, 7:9]
        predictions = predictions.copy()
        predictions[:, 0:1] = (predictions[:, 0:1] - padding[:, 0:1]) / x_scale
        predictions[:, 1:2] = (predictions[:, 1:2] - padding[:, 1:2]) / y_scale
        predictions[:, 2:3] /= x_scale
        predictions[:, 3:4] /= y_scale
    center_error = np.linalg.norm(
        (predictions[:, :2] - targets[:, :2]) * FRAME_SIZE, axis=1
    )
    radius_error = np.linalg.norm(
        (predictions[:, 2:4] - targets[:, 2:4]) * FRAME_SIZE, axis=1
    )
    return {
        "n": int(len(targets)),
        "center_mae_px": float(np.mean(center_error)),
        "center_median_px": float(np.median(center_error)),
        "center_pct_le_8px": float(np.mean(center_error <= 8.0)),
        "center_pct_le_16px": float(np.mean(center_error <= 16.0)),
        "radius_mae_px": float(np.mean(radius_error)),
        "radius_median_px": float(np.median(radius_error)),
        "radius_pct_le_8px": float(np.mean(radius_error <= 8.0)),
        "pred_radius_mean": [float(value) for value in np.mean(predictions[:, 2:4], axis=0)],
        "gt_radius_mean": [float(value) for value in np.mean(targets[:, 2:4], axis=0)],
        "pred_radius_std": [float(value) for value in np.std(predictions[:, 2:4], axis=0)],
    }


def main() -> None:
    """Parse arguments, evaluate all test zips, and write a JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    print("Input:", input_detail["shape"].tolist(), input_detail["dtype"], input_detail["quantization"])
    print("Outputs:", [(detail["name"], detail["shape"].tolist(), detail["dtype"], detail["quantization"]) for detail in output_details])

    model_size = int(input_detail["shape"][1])
    report: dict[str, Any] = {"model": str(args.model), "image_size": model_size, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name, model_size)
        predictions = _predict(interpreter, images)
        report["tests"][zip_name] = _metrics(predictions, targets)
        print(zip_name, json.dumps(report["tests"][zip_name], indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
