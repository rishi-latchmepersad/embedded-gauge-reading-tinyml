#!/usr/bin/env python3
"""Compare heatmap-only and heatmap-plus-offset center decoders."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_mask_640_center import IMAGE_SIZE, decode_masks, predict_int8


def main() -> None:
    """Report center and radius metrics for both center decoding variants."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name, IMAGE_SIZE)
        images = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        masks, heatmaps, offsets, geometry = predict_int8(args.model, images)
        moment = decode_masks(masks)
        for name in ("heatmap",):
            prediction = geometry.copy()
            for index, heatmap in enumerate(heatmaps[..., 0]):
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                prediction[index, :2] = [(x + 0.5) / heatmap.shape[1],
                                          (y + 0.5) / heatmap.shape[0]]
            # Keep the comparison focused on center decoding; use mask moments
            # for radius because the geometry head is intentionally uncalibrated.
            prediction[:, 2:4] = moment[:, 2:4]
            prediction = np.concatenate([prediction, np.ones((len(prediction), 1), dtype=np.float32)], axis=1)
            print(zip_name, name, _metrics(prediction, targets), flush=True)


if __name__ == "__main__":
    main()
