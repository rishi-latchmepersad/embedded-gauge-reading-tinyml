#!/usr/bin/env python3
"""Train a compact YOLO11 pose detector whose keypoint is the face center."""

from __future__ import annotations

import argparse
import resource
from pathlib import Path

import torch
from ultralytics import YOLO

GPU_MEMORY_FRACTION = 15_000 / (16 * 1024)

def main() -> None:
    """Train the pose detector within the project resource limits."""
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data/yolo_pose_labelled_640/dataset.yaml")
    parser.add_argument("--output", type=Path, default=root / "artifacts/yolo11n_pose_labelled_640")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (55 * 1024**3, 55 * 1024**3))
    if torch.cuda.is_available():
        # why: PyTorch takes a fraction of the 16 GiB device, so this is the
        # closest equivalent to the project's explicit 15,000 MB ceiling.
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    model = YOLO("yolo11n-pose.pt")
    model.train(data=str(args.data), epochs=args.epochs, imgsz=640, batch=args.batch, device=0 if torch.cuda.is_available() else "cpu", workers=0, project=str(args.output), name="train", exist_ok=True, pretrained=True, patience=8, close_mosaic=3, cos_lr=True, warmup_epochs=1, degrees=8.0, translate=0.08, scale=0.35, shear=2.0, perspective=0.0, fliplr=0.5, mosaic=0.5, mixup=0.05, cache=False, amp=True, val=True, save=True)
    print("best", args.output / "train/weights/best.pt", flush=True)


if __name__ == "__main__":
    main()
