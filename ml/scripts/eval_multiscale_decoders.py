#!/usr/bin/env python3
"""Compare coarse, fine, and fused center decoders for the multiscale model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_mask_640_center import IMAGE_SIZE, decode_masks, predict_int8


def decode_heatmap(heatmaps: np.ndarray) -> np.ndarray:
    """Decode heatmap peaks into normalized x/y coordinates."""
    decoded = np.zeros((len(heatmaps), 2), dtype=np.float32)
    for index, heatmap in enumerate(heatmaps[..., 0]):
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        decoded[index] = [(x + 0.5) / heatmap.shape[1], (y + 0.5) / heatmap.shape[0]]
    return decoded


def main() -> None:
    """Evaluate fine, coarse, average, and confidence-routed centers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name, IMAGE_SIZE)
        images = tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        masks, coarse, fine, geometry = predict_int8(args.model, images)
        moment = decode_masks(masks)
        coarse_xy = decode_heatmap(coarse)
        fine_xy = decode_heatmap(fine)
        coarse_peak = np.max(coarse[..., 0], axis=(1, 2))
        fine_peak = np.max(fine[..., 0], axis=(1, 2))
        candidates = {
            "fine": fine_xy,
            "coarse": coarse_xy,
            "average": 0.5 * (fine_xy + coarse_xy),
            # why: coarse supervision should be trusted when fine confidence
            # collapses, which is the observed tiny-gauge failure mode.
            "coarse_when_fine_weak": np.where((fine_peak < 0.55)[:, None], coarse_xy, fine_xy),
        }
        for name, centers in candidates.items():
            prediction = geometry.copy()
            prediction[:, :2] = centers
            prediction = np.concatenate([prediction, np.ones((len(prediction), 1), dtype=np.float32)], axis=1)
            print(zip_name, name, _metrics(prediction, targets), flush=True)


if __name__ == "__main__":
    main()
