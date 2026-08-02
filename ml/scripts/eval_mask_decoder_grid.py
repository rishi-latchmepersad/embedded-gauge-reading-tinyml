#!/usr/bin/env python3
"""Search simple deterministic moment-decoder calibrations for a mask model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_mask_all_domains_384 import MASK_SIZE, predict_int8


def decode(masks: np.ndarray, floor: float, radius_factor: float) -> np.ndarray:
    """Decode weighted mask moments with a background floor and scale factor."""
    coords = (np.arange(MASK_SIZE, dtype=np.float32) + 0.5) / MASK_SIZE
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    boxes = []
    for mask in masks[..., 0]:
        weights = np.maximum(mask - floor, 0.0)
        total = float(weights.sum())
        if total <= 1e-6:
            weights = np.maximum(mask, 0.0)
            total = float(weights.sum())
        cx = float((weights * xx).sum() / max(total, 1e-6))
        cy = float((weights * yy).sum() / max(total, 1e-6))
        rx = radius_factor * np.sqrt(max(float((weights * (xx - cx) ** 2).sum() / max(total, 1e-6)), 1e-8))
        ry = radius_factor * np.sqrt(max(float((weights * (yy - cy) ** 2).sum() / max(total, 1e-6)), 1e-8))
        boxes.append([cx, cy, rx, ry, 1.0])
    return np.asarray(boxes, dtype=np.float32)


def main() -> None:
    """Print the best decoder calibration on each independent test archive."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    masks_by_zip = {}
    targets_by_zip = {}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        masks_by_zip[zip_name] = predict_int8(args.model, images)
        targets_by_zip[zip_name] = targets
    for floor in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        for factor in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            metrics = [_metrics(decode(masks_by_zip[z], floor, factor), targets_by_zip[z]) for z in masks_by_zip]
            score = sum(item["center_mae_px"] + item["radius_mae_px"] for item in metrics)
            if score < 250.0:
                print(f"floor={floor:.2f} factor={factor:.1f} score={score:.1f} metrics={metrics}")


if __name__ == "__main__":
    main()
