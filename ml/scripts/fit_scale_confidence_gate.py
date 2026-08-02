#!/usr/bin/env python3
"""Fit a clean validation threshold for the learned tiny-scale confidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip
from train_ellipse_mask_640_center import decode_masks, predict_int8
from train_ellipse_robust_384 import load_zips


def collect(model: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect clean targets, moment centers, heatmap centers, and confidence."""
    targets_all: list[np.ndarray] = []
    moments_all: list[np.ndarray] = []
    heat_all: list[np.ndarray] = []
    confidence_all: list[np.ndarray] = []
    board_images, board_targets = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    for images, targets in ((board_images, board_targets), (tiny_images, tiny_targets)):
        resized = tf.image.resize(images, (384, 384)).numpy()
        masks, heatmaps, offsets, geometry, confidence = predict_int8(model, resized)
        moments = decode_masks(masks)[:, :2]
        heat_centers = []
        for heatmap in heatmaps[..., 0]:
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            heat_centers.append([(x + 0.5) / 96.0, (y + 0.5) / 96.0])
        targets_all.append(targets[:, :2])
        moments_all.append(moments)
        heat_all.append(np.asarray(heat_centers, dtype=np.float32))
        confidence_all.append(confidence[:, 0])
    return tuple(np.concatenate(values) for values in (targets_all, moments_all, heat_all, confidence_all))


def main() -> None:
    """Select the confidence threshold minimizing clean validation center error."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target, moment, heat, confidence = collect(args.model)
    best: tuple[float, float] | None = None
    for threshold in np.arange(0.05, 0.96, 0.01):
        use_heat = confidence >= threshold
        prediction = np.where(use_heat[:, None], heat, moment)
        error = np.linalg.norm((prediction - target) * 384.0, axis=1)
        score = float(np.mean(error))
        if best is None or score < best[0]:
            best = score, float(threshold)
    assert best is not None
    report = {"validation_center_mae_px": best[0], "scale_threshold": best[1], "n": len(target)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
