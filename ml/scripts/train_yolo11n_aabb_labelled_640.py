#!/usr/bin/env python3
"""Train a small pretrained YOLO11 axis-aligned gauge detector at 640px."""

from __future__ import annotations

import argparse
import resource
from pathlib import Path

import torch
from ultralytics import YOLO

GPU_MEMORY_FRACTION = 15_000 / (16 * 1024)

def main() -> None:
    """Train and validate the AABB detector under the requested resource caps."""
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data/yolo_aabb_labelled_640/dataset.yaml")
    parser.add_argument("--output", type=Path, default=root / "artifacts/yolo11n_aabb_labelled_640")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    # why: bound virtual address space so a loader or augmentation regression
    # cannot consume the user's requested 50 GB host-RAM ceiling.
    resource.setrlimit(resource.RLIMIT_AS, (55 * 1024**3, 55 * 1024**3))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=640, batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu", workers=0,
        project=str(args.output), name="train", exist_ok=True, pretrained=True,
        patience=8, close_mosaic=3, cos_lr=True, warmup_epochs=1,
        degrees=8.0, translate=0.08, scale=0.35, shear=2.0,
        perspective=0.0, fliplr=0.5, mosaic=0.5, mixup=0.05,
        cache=False, amp=True, val=True, save=True,
    )
    print("best", args.output / "train/weights/best.pt", flush=True)


if __name__ == "__main__":
    main()
