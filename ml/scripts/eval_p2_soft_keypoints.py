#!/usr/bin/env python3
"""Evaluate soft-argmax decoding for the five-keypoint P2 ellipse model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_mask_640_center import predict_int8


def decode(heatmaps: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    """Compute sub-cell keypoints using foreground-weighted spatial moments."""
    size = heatmaps.shape[1]
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    points = np.zeros((len(heatmaps), 5, 2), dtype=np.float32)
    for i, maps in enumerate(heatmaps):
        for k in range(5):
            weights = np.maximum(maps[..., k] - 0.10, 0.0)
            total = float(weights.sum()) + 1e-6
            points[i, k] = [(weights * xx).sum() / total, (weights * yy).sum() / total]
    output = geometry.copy()
    output[:, :2] = points[:, 0]
    output[:, 2] = np.maximum((points[:, 2, 0] - points[:, 1, 0]) * 0.5, 1e-3)
    output[:, 3] = np.maximum((points[:, 4, 1] - points[:, 3, 1]) * 0.5, 1e-3)
    return output


def main() -> None:
    """Run soft-argmax inference on all three independent test archives."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, object] = {"model": str(args.model), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name, 320)
        _, heatmaps, geometry = predict_int8(args.model, tf.image.resize(images, (320, 320)).numpy())
        predictions = np.concatenate([decode(heatmaps, geometry), np.ones((len(images), 1), np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, targets)
        print(zip_name, json.dumps(report["tests"][zip_name], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
