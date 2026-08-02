#!/usr/bin/env python3
"""Train a small pretrained YOLO11 OBB gauge detector at 640px."""

from __future__ import annotations

import argparse
import resource
from pathlib import Path

import torch
from ultralytics import YOLO

GPU_MEMORY_FRACTION = 15_000 / (16 * 1024)

def main() -> None:
    """Train and validate the YOLO OBB detector under bounded resources."""
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data/yolo_obb_labelled_640b/dataset.yaml")
    parser.add_argument("--output", type=Path, default=root / "artifacts/yolo11n_obb_labelled_640")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    # why: the user requested a hard host-RAM ceiling for long experiments.
    resource.setrlimit(resource.RLIMIT_AS, (55 * 1024**3, 55 * 1024**3))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
    model = YOLO("yolo11n-obb.pt")
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=640, batch=args.batch,
        # why: WSL's restricted IPC namespace rejects Ultralytics worker
        # sockets; single-process loading is stable and stays within the RAM cap.
        device=0 if torch.cuda.is_available() else "cpu", workers=0,
        project=str(args.output), name="train", exist_ok=True, pretrained=True,
        patience=20, close_mosaic=10, cos_lr=True, warmup_epochs=3,
        degrees=8.0, translate=0.08, scale=0.35, shear=2.0,
        perspective=0.0, fliplr=0.5, mosaic=0.5, mixup=0.05,
        cache=False, amp=True, val=True, save=True,
    )
    print("best", args.output / "train/weights/best.pt", flush=True)


if __name__ == "__main__":
    main()
