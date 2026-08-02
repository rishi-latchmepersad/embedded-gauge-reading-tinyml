#!/usr/bin/env python3
"""Evaluate mask-derived centers combined with scalar multitask radii."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_heatmap_multitask_384 import predict_contract
from train_ellipse_mask_all_domains_384 import soft_box_from_masks


def main() -> None:
    """Print hybrid geometry metrics for all held-out archives."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        masks, geometry = predict_contract(args.model, images)
        centers = soft_box_from_masks(masks)[:, :2]
        predictions = np.concatenate([centers, geometry[:, 2:4], np.ones((len(geometry), 1), dtype=np.float32)], axis=1)
        print(zip_name, _metrics(predictions, targets))


if __name__ == "__main__":
    main()
