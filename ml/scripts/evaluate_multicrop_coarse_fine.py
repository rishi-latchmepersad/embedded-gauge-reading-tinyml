#!/usr/bin/env python3
"""Evaluate robust multi-crop fusion for the coarse-to-fine ellipse model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_coarse_fine_ellipse_224 import (
    STAGE1_MODEL,
    crop_image,
    predict_local,
    stage1_decode,
)

REFINER = Path("artifacts/coarse_fine_ellipse_224_v1/model_int8.tflite")


def multicrop_predictions(refiner: Path, stage1: Path, images: np.ndarray, center_blend: float) -> np.ndarray:
    """Run six proposal-centered crops and fuse mapped ellipse estimates."""
    proposals = stage1_decode(stage1, images)
    all_crops: list[np.ndarray] = []
    all_sources: list[np.ndarray] = []
    all_indices: list[int] = []
    # why: modest translation and scale perturbations expose local model
    # sensitivity while keeping every crop anchored to the same global face.
    variants = ((0.0, 0.90), (0.0, 1.00), (0.0, 1.10), (-0.04, 1.00), (0.04, 1.00), (0.0, 1.25))
    for index, (image, proposal) in enumerate(zip(images, proposals)):
        base_side = float(np.clip(2.2 * max(proposal[2], proposal[3]), 0.18, 1.4))
        for shift, scale in variants:
            side = float(np.clip(base_side * scale, 0.18, 1.4))
            box = np.asarray([proposal[0] - side / 2 + shift * base_side, proposal[1] - side / 2, proposal[0] + side / 2 + shift * base_side, proposal[1] + side / 2], dtype=np.float32)
            crop, source = crop_image(image, box)
            all_crops.append(crop)
            all_sources.append(source)
            all_indices.append(index)
    local = predict_local(refiner, np.asarray(all_crops, dtype=np.float32))
    grouped: list[np.ndarray] = []
    for index, proposal in enumerate(proposals):
        mapped = []
        for value, source, owner in zip(local, all_sources, all_indices):
            if owner != index:
                continue
            sx, sy = source[2] - source[0], source[3] - source[1]
            mapped.append([source[0] + value[0] * sx, source[1] + value[1] * sy, value[2] * sx, value[3] * sy])
        values = np.asarray(mapped, dtype=np.float32)
        # why: median fusion is robust to one crop locking onto a tick mark or
        # a reflection instead of the face boundary.
        fused = np.median(values, axis=0)
        fused[:2] = (1.0 - center_blend) * proposal[:2] + center_blend * fused[:2]
        grouped.append(fused)
    return np.asarray(grouped, dtype=np.float32)


def main() -> None:
    """Score multi-crop center/radius fusion on all held-out test sets."""
    report: dict[str, object] = {"tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        rows: dict[str, object] = {}
        for blend in (0.0, 0.15, 0.25, 0.35, 0.50):
            predictions = multicrop_predictions(REFINER, STAGE1_MODEL, images, blend)
            rows[f"center_blend_{blend:.2f}"] = _metrics(np.concatenate([predictions, np.ones((len(predictions), 1), dtype=np.float32)], axis=1), targets)
        report["tests"][zip_name] = rows
        print(zip_name, json.dumps(rows), flush=True)
    Path("artifacts/multicrop_coarse_fine_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
