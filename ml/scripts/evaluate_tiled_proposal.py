#!/usr/bin/env python3
"""Evaluate shared-weight tiled proposal inference for tiny gauges."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_coarse_fine_ellipse_224 import stage1_decode
from train_ellipse_mask_640_center import decode_masks, predict_int8

MODEL = Path("artifacts/gauge_ellipse_mask_center_scaleconf_384_aux_v1/model_int8.tflite")
IMAGE_SIZE = 384


def make_tile(image: np.ndarray, x: float, y: float, side: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract a normalized square tile and return its source coordinates."""
    height, width = image.shape[:2]
    ix, iy, pixels = int(round(x * width)), int(round(y * height)), int(round(side * max(width, height)))
    pixels = max(pixels, 2)
    canvas = np.zeros((pixels, pixels), dtype=np.float32)
    sx1, sy1 = max(0, ix), max(0, iy)
    sx2, sy2 = min(width, ix + pixels), min(height, iy + pixels)
    canvas[sy1 - iy:sy2 - iy, sx1 - ix:sx2 - ix] = image[sy1:sy2, sx1:sx2, 0]
    return cv2.resize(canvas, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)[..., None], np.asarray([ix / width, iy / height, (ix + pixels) / width, (iy + pixels) / height], dtype=np.float32)


def tiled_predictions(images: np.ndarray) -> np.ndarray:
    """Run one shared model over full frame and overlapping high-resolution tiles."""
    crops: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    owners: list[int] = []
    # why: the same weights see the entire scene and four overlapping views;
    # the tiles preserve pixels for tiny gauges without domain-specific heads.
    for index, image in enumerate(images):
        crops.append(image)
        sources.append(np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
        owners.append(index)
        for x, y in ((0.0, 0.0), (0.4, 0.0), (0.0, 0.4), (0.4, 0.4)):
            crop, source = make_tile(image, x, y, 0.6)
            crops.append(crop)
            sources.append(source)
            owners.append(index)
    masks, heatmaps, _, geometry, scale = predict_int8(MODEL, np.asarray(crops, dtype=np.float32))
    proposals = decode_masks(masks)
    candidates: list[list[np.ndarray]] = [[] for _ in images]
    scores: list[list[float]] = [[] for _ in images]
    for row, (owner, source) in enumerate(zip(owners, sources)):
        proposal = proposals[row].copy()
        heat = float(np.max(heatmaps[row, ..., 0]))
        if float(scale[row, 0]) >= 0.5 and heat >= 0.55:
            y, x = np.unravel_index(np.argmax(heatmaps[row, ..., 0]), heatmaps[row, ..., 0].shape)
            proposal[:2] = [(x + 0.5) / heatmaps.shape[1], (y + 0.5) / heatmaps.shape[2]]
            proposal[2:4] = geometry[row, 2:4] * np.asarray([0.487, 0.368], dtype=np.float32)
        sx, sy = source[2] - source[0], source[3] - source[1]
        mapped = np.asarray([source[0] + proposal[0] * sx, source[1] + proposal[1] * sy, proposal[2] * sx, proposal[3] * sy], dtype=np.float32)
        candidates[owner].append(mapped)
        scores[owner].append(heat * (0.5 + float(scale[row, 0])))
    return np.asarray([rows[int(np.argmax(scores[index]))] for index, rows in enumerate(candidates)], dtype=np.float32)


def main() -> None:
    """Score full-frame plus tiled shared-weight proposals on all tests."""
    report: dict[str, object] = {"tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        tiled = tiled_predictions(images)
        predictions = np.concatenate([tiled, np.ones((len(tiled), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    Path("artifacts/tiled_proposal_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
