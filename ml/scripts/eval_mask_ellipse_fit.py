#!/usr/bin/env python3
"""Evaluate classical ellipse fitting on an int8 neural mask output."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip, _metrics


def predict_contract(model: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode mask, fine heatmap, and geometry from known candidate contracts."""
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    rows: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        rows.append((raw - out_zero) * out_scale)
    values = np.asarray(rows, dtype=np.float32)
    # The three contracts differ only in auxiliary heads; all use a 96x96 mask
    # and fine heatmap, so this decoder keeps the comparison architecture-free.
    mask_values = 96 * 96
    mask = values[:, :mask_values].reshape(-1, 96, 96, 1)
    if values.shape[1] == 36868:
        heat = values[:, mask_values:2 * mask_values].reshape(-1, 96, 96, 1)
        geometry = values[:, -4:]
    elif values.shape[1] == 20740:
        fine_start = mask_values + 48 * 48
        heat = values[:, fine_start:fine_start + mask_values].reshape(-1, 96, 96, 1)
        geometry = values[:, -4:]
    else:
        heat = values[:, mask_values:2 * mask_values].reshape(-1, 96, 96, 1)
        geometry = values[:, -4:]
    return mask, heat, geometry


def heat_centers(heatmaps: np.ndarray) -> np.ndarray:
    """Decode fine heatmap centers at grid-cell centers."""
    centers = np.zeros((len(heatmaps), 2), dtype=np.float32)
    for index, heatmap in enumerate(heatmaps[..., 0]):
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        centers[index] = [(x + 0.5) / 96.0, (y + 0.5) / 96.0]
    return centers


def fit_masks(masks: np.ndarray, threshold: float, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the largest binary contour and fall back when it is not elliptical."""
    predictions = fallback.copy()
    valid = np.zeros(len(masks), dtype=bool)
    for index, mask in enumerate(masks[..., 0]):
        binary = (mask >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5 or cv2.contourArea(contour) < 2.0:
            continue
        (cx, cy), (major, minor), angle = cv2.fitEllipse(contour)
        predictions[index, :2] = [cx / 96.0, cy / 96.0]
        predictions[index, 2:4] = [max(major, minor) / (2.0 * 96.0), min(major, minor) / (2.0 * 96.0)]
        valid[index] = True
    return predictions, valid


def main() -> None:
    """Sweep mask thresholds on all test sets and report fitted ellipses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name, 384)
        images = tf.image.resize(images, (384, 384)).numpy()
        masks, heatmaps, geometry = predict_contract(args.model, images)
        fallback = geometry.copy()
        fallback[:, :2] = heat_centers(heatmaps)
        for threshold in (0.10, 0.20, 0.30, 0.40, 0.50):
            predictions, valid = fit_masks(masks, threshold, fallback)
            predictions = np.concatenate([predictions, np.ones((len(predictions), 1), dtype=np.float32)], axis=1)
            metrics = _metrics(predictions, targets)
            print(zip_name, f"threshold={threshold:.2f}", f"valid={valid.mean():.3f}", metrics, flush=True)


if __name__ == "__main__":
    main()
