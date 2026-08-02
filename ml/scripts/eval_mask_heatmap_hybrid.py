#!/usr/bin/env python3
"""Evaluate a scale-aware hybrid decoder for the mask/heatmap model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_mask_640_center import IMAGE_SIZE, decode_masks, predict_int8


def main() -> None:
    """Select heatmap centers only for predicted tiny faces and report metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny-radius", type=float, default=0.18)
    parser.add_argument("--radius-source", choices=("mask", "geometry"), default="mask")
    parser.add_argument("--tiny-radius-scale", type=float, nargs=2, default=(0.487, 0.368))
    parser.add_argument("--heat-threshold", type=float, default=0.55)
    parser.add_argument("--scale-threshold", type=float, default=0.5)
    args = parser.parse_args()
    report: dict[str, object] = {"model": str(args.model), "tiny_radius": args.tiny_radius, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name, IMAGE_SIZE)
        images = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        masks, heatmaps, offsets, geometry, scale_confidence = predict_int8(args.model, images)
        moment = decode_masks(masks)
        prediction = moment.copy()
        if args.radius_source == "geometry":
            prediction[:, 2:4] = geometry[:, 2:4]
        for index, heatmap in enumerate(heatmaps[..., 0]):
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            # The auxiliary geometry head is a better scale gate than mask
            # moments, whose radius estimate is biased when tiny masks blur.
            estimated_radius = float(scale_confidence[index, 0])
            peak = float(np.max(heatmap))
            if estimated_radius >= args.scale_threshold and peak >= args.heat_threshold:
                prediction[index, :2] = [(x + 0.5) / heatmap.shape[1],
                                         (y + 0.5) / heatmap.shape[0]]
                prediction[index, 2:4] = geometry[index, 2:4] * np.asarray(args.tiny_radius_scale, dtype=np.float32)
        prediction = np.concatenate([prediction, np.ones((len(prediction), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(prediction, targets)
        print(zip_name, json.dumps(report["tests"][zip_name], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
