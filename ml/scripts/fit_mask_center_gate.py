#!/usr/bin/env python3
"""Fit a non-parametric center decoder gate on clean validation archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip
from train_ellipse_mask_640_center import IMAGE_SIZE, decode_masks, predict_int8
from train_ellipse_robust_384 import load_zips


def collect(model: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect mask/heatmap centers and features from clean validation data."""
    all_targets: list[np.ndarray] = []
    all_moment: list[np.ndarray] = []
    all_heat: list[np.ndarray] = []
    all_features: list[np.ndarray] = []
    # Keep validation fast and domain-relevant: train_2 represents tiny faces
    # and board_captures_1 represents the board-style large/medium faces.
    x1, y1 = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))
    x2, y2 = load_zips(["train_2.zip"], labels=("GaugeFace",))
    for images, targets in ((x1, y1), (x2, y2)):
        resized = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        masks, heatmaps, offsets, geometry = predict_int8(model, resized)
        moment = decode_masks(masks)
        heat = np.zeros((len(images), 2), dtype=np.float32)
        peaks = np.max(heatmaps[..., 0], axis=(1, 2))
        for index, hm in enumerate(heatmaps[..., 0]):
            yy, xx = np.unravel_index(np.argmax(hm), hm.shape)
            heat[index] = [(xx + 0.5) / hm.shape[1], (yy + 0.5) / hm.shape[0]]
        all_targets.append(targets[:, :2])
        all_moment.append(moment[:, :2])
        all_heat.append(heat)
        all_features.append(np.stack([np.mean(geometry[:, 2:4], 1), peaks], axis=1))
    return tuple(np.concatenate(values) for values in (all_targets, all_moment, all_heat, all_features))


def main() -> None:
    """Search geometry/confidence thresholds and save the clean validation rule."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target, moment, heat, features = collect(args.model)
    best: tuple[float, float, float] | None = None
    for radius_threshold in np.arange(0.15, 0.31, 0.01):
        for heat_threshold in np.arange(0.0, 0.81, 0.05):
            use_heat = (features[:, 0] < radius_threshold) & (features[:, 1] >= heat_threshold)
            prediction = np.where(use_heat[:, None], heat, moment)
            error = np.linalg.norm((prediction - target) * 640.0, axis=1)
            score = float(np.mean(error))
            if best is None or score < best[0]:
                best = (score, float(radius_threshold), float(heat_threshold))
    assert best is not None
    report = {"validation_center_mae_px": best[0], "tiny_radius_threshold": best[1], "heat_threshold": best[2], "n": len(target)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
