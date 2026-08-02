#!/usr/bin/env python3
"""Render predicted and ground-truth ellipse overlays for visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip
from train_ellipse_mask_640_center import decode_masks, predict_int8


def main() -> None:
    """Save color overlays using the deployment-style hybrid decoder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--zip", default="test_2.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images, targets = _load_zip(args.zip, 384)
    images = tf.image.resize(images, (384, 384)).numpy()
    masks, heatmaps, offsets, geometry, scale_confidence = predict_int8(args.model, images)
    moments = decode_masks(masks)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (image, target) in enumerate(zip(images, targets)):
        prediction = moments[index].copy()
        # why: the validation-selected rule uses the heatmap for tiny faces,
        # while keeping the learned geometry radius calibrated for them.
        if float(scale_confidence[index, 0]) >= 0.5:
            y, x = np.unravel_index(np.argmax(heatmaps[index, ..., 0]), heatmaps[index, ..., 0].shape)
            prediction[:2] = [(x + 0.5) / 96.0, (y + 0.5) / 96.0]
            prediction[2:4] = geometry[index, 2:4] * np.asarray([0.487, 0.368], dtype=np.float32)
        canvas = cv2.cvtColor(np.clip(image[..., 0] * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for ellipse, color, label in ((target[:4], (0, 220, 0), "gt"), (prediction, (0, 0, 255), "pred")):
            center = tuple(np.round(ellipse[:2] * 384.0).astype(int))
            axes = tuple(np.maximum(1, np.round(ellipse[2:4] * 384.0).astype(int)))
            cv2.ellipse(canvas, center, axes, 0.0, 0.0, 360.0, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (center[0] + 4, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(args.output / f"{index:02d}.png"), canvas)
    print(f"wrote {len(images)} overlays to {args.output}")


if __name__ == "__main__":
    main()
