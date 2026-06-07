"""
Train YOLO11n-OBB on the 320×320 gauge-face dataset.

Usage:
    poetry run python scripts/train_yolo_obb_320.py

Output:
    ml/artifacts/yolo_obb_320/train{run}/  (Ultralytics auto-names runs)
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parents[1] / "data" / "yolo_obb_320" / "dataset.yaml"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "yolo_obb_320"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Tunables — adjust these between rounds
EPOCHS = 300
PATIENCE = 30  # early-stop epochs
BATCH = 16
LR0 = 0.01
IMGSZ = 320
WORKERS = 8  # tune to your core count


def main() -> None:
    # Load a pretrained YOLO11n-OBB model
    model = YOLO("yolo11n-obb.pt")

    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        imgsz=IMGSZ,
        lr0=LR0,
        device="cuda",
        workers=WORKERS,
        project=str(ARTIFACTS),
        name="train",
        exist_ok=True,
        pretrained=True,
        # OBB-specific
        close_mosaic=10,
        cos_lr=True,
        warmup_epochs=3,
        # Augmentation — moderate since our dataset is small
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.0,
        erasing=0.4,
        # Save / export
        save=True,
        val=True,
    )

    print(f"Training complete. Best model at {ARTIFACTS / 'train' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
