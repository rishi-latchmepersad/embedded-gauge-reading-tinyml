#!/usr/bin/env python3
"""Evaluate YOLO11 OBB centroids as normalized ellipse proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from eval_ellipse_all_test_sets import _load_zip, _metrics

GPU_MEMORY_FRACTION = 15_000 / (16 * 1024)

def main() -> None:
    """Run OBB inference on every held-out test archive."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    model = YOLO(str(args.model))
    reports: dict[str, object] = {"model": str(args.model), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name, image_size=384)
        rgb = np.repeat(np.clip(images * 255.0, 0.0, 255.0).astype(np.uint8), 3, axis=-1)
        results = model.predict(list(rgb), imgsz=640, batch=1, device=0 if torch.cuda.is_available() else "cpu", conf=0.001, verbose=False)
        predictions: list[list[float]] = []
        for result in results:
            if result.obb is None or len(result.obb) == 0:
                predictions.append([0.5, 0.5, 0.25, 0.25, 0.0])
                continue
            corners = result.obb.xyxyxyxy[0].detach().cpu().numpy()
            center = corners.mean(axis=0)
            radii = (corners.max(axis=0) - corners.min(axis=0)) / 2.0
            confidence = float(result.obb.conf[0].detach().cpu().item())
            predictions.append([center[0] / 384.0, center[1] / 384.0, radii[0] / 384.0, radii[1] / 384.0, confidence])
        metrics = _metrics(np.asarray(predictions, dtype=np.float32), targets)
        reports["tests"][zip_name] = metrics
        print(zip_name, json.dumps(metrics), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
