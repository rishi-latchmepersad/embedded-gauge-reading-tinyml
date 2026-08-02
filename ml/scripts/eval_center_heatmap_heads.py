#!/usr/bin/env python3
"""Compare center heatmap and scalar geometry heads independently."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_center_heatmap_640 import decode_centers, predict_int8
from train_ellipse_scalar_640 import resize_cpu


def main() -> None:
    """Print head-specific metrics for the high-resolution model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        heatmaps, geometry = predict_int8(args.model, resize_cpu(images))
        heat_predictions = np.concatenate([decode_centers(heatmaps), geometry[:, 2:4], np.ones((len(geometry), 1), dtype=np.float32)], axis=1)
        scalar_predictions = np.concatenate([geometry, np.ones((len(geometry), 1), dtype=np.float32)], axis=1)
        print(zip_name, "heatmap", _metrics(heat_predictions, targets), "scalar", _metrics(scalar_predictions, targets))


if __name__ == "__main__":
    main()
