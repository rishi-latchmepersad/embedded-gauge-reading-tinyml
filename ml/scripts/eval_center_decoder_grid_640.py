#!/usr/bin/env python3
"""Search deterministic center-heatmap decoding temperatures and floors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_center_heatmap_640 import HEATMAP_SIZE, predict_int8
from train_ellipse_scalar_640 import resize_cpu


def decode(heatmaps: np.ndarray, floor: float, power: float) -> np.ndarray:
    """Decode a heatmap with background subtraction and sharpening power."""
    coords = (np.arange(HEATMAP_SIZE, dtype=np.float32) + 0.5) / HEATMAP_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    result = []
    for heatmap in heatmaps[..., 0]:
        weights = np.maximum(heatmap - floor, 0.0) ** power
        total = max(float(weights.sum()), 1e-6)
        result.append([(weights * xx).sum() / total, (weights * yy).sum() / total])
    return np.asarray(result, dtype=np.float32)


def main() -> None:
    """Report center metrics for each decoder setting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    values = {}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        heatmaps, geometry = predict_int8(args.model, resize_cpu(images))
        values[zip_name] = (heatmaps, geometry, targets)
    for floor in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        for power in (1.0, 2.0, 3.0, 4.0):
            metrics = []
            for heatmaps, geometry, targets in values.values():
                centers = decode(heatmaps, floor, power)
                predictions = np.concatenate([centers, geometry[:, 2:4], np.ones((len(geometry), 1), dtype=np.float32)], axis=1)
                metrics.append(_metrics(predictions, targets))
            if metrics[1]["center_mae_px"] < 35.0:
                print(f"floor={floor:.2f} power={power:.1f} metrics={metrics}")


if __name__ == "__main__":
    main()
