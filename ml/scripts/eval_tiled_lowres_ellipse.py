#!/usr/bin/env python3
"""Evaluate a single 384 model with optional overlapping small-face tiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_mask_all_domains_384 import soft_box_from_masks, predict_int8


def remap_box(box: np.ndarray, x0: float, y0: float, size: float) -> np.ndarray:
    """Map a tile-normalized ellipse back into full-frame coordinates."""
    mapped = box.copy()
    mapped[:2] = np.asarray([x0, y0]) + size * box[:2]
    mapped[2:4] = size * box[2:4]
    return mapped


def main() -> None:
    """Search tile sizes and select the highest-evidence candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tile-size", type=float, default=0.60)
    args = parser.parse_args()
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        full_masks = predict_int8(args.model, images)
        full_boxes = np.concatenate([soft_box_from_masks(full_masks), np.ones((len(images), 1), dtype=np.float32)], axis=1)
        full_scale = np.max(full_boxes[:, 2:4], axis=1)
        predictions = full_boxes.copy()
        for index, image in enumerate(images):
            if full_scale[index] >= 0.25:
                continue
            size = args.tile_size
            candidates: list[tuple[float, np.ndarray]] = []
            for x0, y0 in ((0.0, 1.0 - size), (1.0 - size, 1.0 - size), (0.0, 0.0), (1.0 - size, 0.0)):
                crop = image[int(y0 * 384) : int((y0 + size) * 384), int(x0 * 384) : int((x0 + size) * 384)]
                crop = tf.image.resize(crop[None], [384, 384]).numpy()[0]
                mask = predict_int8(args.model, crop[None])[0]
                box = soft_box_from_masks(mask[None])[0]
                score = float(np.maximum(mask[..., 0] - 0.1, 0.0).sum())
                candidates.append((score, remap_box(np.r_[box, 1.0], x0, y0, size)))
            predictions[index] = max(candidates, key=lambda item: item[0])[1]
        print(zip_name, _metrics(predictions, targets), "tiled", int(np.sum(full_scale < 0.25)))


if __name__ == "__main__":
    main()
