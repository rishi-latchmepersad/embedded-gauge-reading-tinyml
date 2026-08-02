#!/usr/bin/env python3
"""Evaluate YOLO pose center keypoints and AABB radii on held-out sets."""

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
    """Run pose inference and score the explicit keypoint center."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.05)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    model = YOLO(str(args.model))
    report: dict[str, object] = {"model": str(args.model), "conf": args.conf, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name, image_size=384)
        rgb = np.repeat(np.clip(images * 255.0, 0.0, 255.0).astype(np.uint8), 3, axis=-1)
        results = model.predict(list(rgb), imgsz=640, batch=1, device=0 if torch.cuda.is_available() else "cpu", conf=args.conf, verbose=False)
        predictions: list[list[float]] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
                predictions.append([0.5, 0.5, 0.25, 0.25, 0.0])
                continue
            box = result.boxes.xyxy[0].detach().cpu().numpy()
            point = result.keypoints.xy[0, 0].detach().cpu().numpy()
            center = point
            radii = (box[2:] - box[:2]) / 2.0
            confidence = float(result.boxes.conf[0].detach().cpu().item())
            predictions.append([center[0] / 384.0, center[1] / 384.0, radii[0] / 384.0, radii[1] / 384.0, confidence])
        metrics = _metrics(np.asarray(predictions, dtype=np.float32), targets)
        report["tests"][zip_name] = metrics
        print(zip_name, json.dumps(metrics), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
