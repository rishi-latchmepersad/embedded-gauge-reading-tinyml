"""Train an auxiliary four-landmark OBB model for generalization pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_YAML = REPO_ROOT / "ml/data/external/gauge_meter_detection_v2_yolov8_obb/data_local.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "ml/yolo11n-obb.pt"
DEFAULT_PROJECT = REPO_ROOT / "ml/artifacts/external_landmark_obb"


def configure_gpu() -> None:
    """Cap TensorFlow and PyTorch GPU usage before model initialization."""

    # why: the WSL workstation has a 4 GB GPU and needs headroom for the desktop.
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3900)]
        )

    # why: Ultralytics uses PyTorch, so apply a matching fraction cap to its allocator.
    import torch

    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(3900 * 1024**2 / total_memory)


def main() -> None:
    """Run auxiliary landmark pretraining; do not treat its output as deployable."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_YAML,
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="yolo11n_landmark_pretrain")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_gpu()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        cache=False,
        plots=True,
    )


if __name__ == "__main__":
    main()
