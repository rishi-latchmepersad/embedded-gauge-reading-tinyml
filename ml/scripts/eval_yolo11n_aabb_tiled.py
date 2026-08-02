#!/usr/bin/env python3
"""Evaluate a full-frame plus overlapping-tile AABB proposal pyramid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from eval_ellipse_all_test_sets import _load_zip, _metrics

GPU_MEMORY_FRACTION = 15_000 / (16 * 1024)

def as_uint8_rgb(images: np.ndarray) -> np.ndarray:
    """Convert the shared normalized grayscale loader to detector images."""
    return np.repeat(np.clip(images * 255.0, 0.0, 255.0).astype(np.uint8), 3, axis=-1)


def infer_candidates(model: YOLO, images: np.ndarray, conf: float) -> list[list[float]]:
    """Return one normalized full-image box proposal per image."""
    results = model.predict(
        list(images), imgsz=640, batch=1,
        device=0 if torch.cuda.is_available() else "cpu", conf=conf, verbose=False,
    )
    proposals: list[list[float]] = []
    height, width = images.shape[1:3]
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            proposals.append([0.5, 0.5, 0.25, 0.25, 0.0])
            continue
        box = result.boxes.xyxy[0].detach().cpu().numpy()
        center = (box[:2] + box[2:]) / 2.0
        radii = (box[2:] - box[:2]) / 2.0
        proposals.append([center[0] / width, center[1] / height, radii[0] / width, radii[1] / height, float(result.boxes.conf[0].detach().cpu().item())])
    return proposals


def tiled_candidates(model: YOLO, images: np.ndarray, conf: float) -> list[list[float]]:
    """Detect on four overlapping tiles and map the best tile to full-frame coordinates."""
    tile_size = images.shape[1] * 2 // 3
    starts = (0, images.shape[1] - tile_size)
    tiles: list[np.ndarray] = []
    metadata: list[tuple[int, int]] = []
    for image_index, image in enumerate(images):
        for top in starts:
            for left in starts:
                tiles.append(image[top : top + tile_size, left : left + tile_size])
                metadata.append((image_index, top))
                # Store left in a parallel encoding to keep the metadata typed
                # without introducing a per-tile object allocation.
                metadata[-1] = (image_index, top * images.shape[1] + left)
    tile_results = model.predict(
        # why: tiled inference multiplies the source count; batch four keeps
        # peak activation memory bounded on the 16 GB host GPU.
        list(tiles), imgsz=640, batch=1,
        device=0 if torch.cuda.is_available() else "cpu", conf=conf, verbose=False,
    )
    height, width = images.shape[1:3]
    best = [[0.5, 0.5, 0.25, 0.25, 0.0] for _ in images]
    for result, (image_index, encoded) in zip(tile_results, metadata):
        top, left = divmod(encoded, width)
        if result.boxes is None or len(result.boxes) == 0:
            continue
        box = result.boxes.xyxy[0].detach().cpu().numpy()
        confidence = float(result.boxes.conf[0].detach().cpu().item())
        if confidence <= best[image_index][4]:
            continue
        center = (box[:2] + box[2:]) / 2.0 + np.asarray([left, top], dtype=np.float32)
        radii = (box[2:] - box[:2]) / 2.0
        best[image_index] = [center[0] / width, center[1] / height, radii[0] / width, radii[1] / height, confidence]
    return best


def main() -> None:
    """Compare full-frame, tiled, and confidence-routed proposal pyramids."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--test", choices=("test_1.zip", "test_2.zip", "test_3.zip"), default=None)
    args = parser.parse_args()
    # why: keep evaluation consistent with the project's 15 GB host-GPU cap.
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    model = YOLO(str(args.model))
    report: dict[str, object] = {"model": str(args.model), "conf": args.conf, "tests": {}}
    zip_names = (args.test,) if args.test is not None else ("test_1.zip", "test_2.zip", "test_3.zip")
    for zip_name in zip_names:
        images, targets = _load_zip(zip_name, image_size=384)
        rgb = as_uint8_rgb(images)
        full = np.asarray(infer_candidates(model, rgb, args.conf), dtype=np.float32)
        tiled = np.asarray(tiled_candidates(model, rgb, args.conf), dtype=np.float32)
        # The confidence route is a scale route, not a domain-specific route:
        # the tile pyramid wins only when it produces a more confident proposal.
        routed = np.where((tiled[:, 4] > full[:, 4])[:, None], tiled, full)
        metrics = {"full": _metrics(full, targets), "tiled": _metrics(tiled, targets), "routed": _metrics(routed, targets)}
        report["tests"][zip_name] = metrics
        print(zip_name, json.dumps(metrics), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
