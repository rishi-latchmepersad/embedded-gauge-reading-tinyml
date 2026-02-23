"""Run one end-to-end training job and save artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

# Add `ml/src` to sys.path so this script works even before `poetry install`.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import TrainConfig, train


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a reproducible training run."""
    parser = argparse.ArgumentParser(description="Train gauge value regressor.")
    parser.add_argument("--gauge-id", type=str, default="littlegood_home_temp_gauge_c")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--model-family",
        type=str,
        choices=["compact", "mobilenet_v2"],
        default="mobilenet_v2",
        help="Select model architecture family.",
    )
    parser.add_argument(
        "--mobilenet-backbone-trainable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable end-to-end MobileNetV2 fine-tuning (default: enabled).",
    )
    parser.add_argument(
        "--no-mobilenet-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights for MobileNetV2.",
    )
    parser.add_argument(
        "--mobilenet-warmup-epochs",
        type=int,
        default=8,
        help="Warmup epochs with frozen MobileNetV2 backbone before fine-tuning.",
    )
    parser.add_argument("--strict-labels", action="store_true")
    parser.add_argument("--crop-pad-ratio", type=float, default=0.25)
    parser.add_argument("--no-augment-training", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="gpu",
        help="Select accelerator mode. Default is 'gpu' for the best-performing setup.",
    )
    parser.add_argument(
        "--no-gpu-memory-growth",
        action="store_true",
        help="Disable TensorFlow GPU memory growth.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable Keras mixed_float16 policy (best on Tensor Core GPUs).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run folder name. Defaults to timestamp.",
    )
    return parser.parse_args()


def to_serializable_history(history: dict[str, list[Any]]) -> dict[str, list[float]]:
    """Convert Keras history values to plain float lists for JSON."""
    return {key: [float(v) for v in values] for key, values in history.items()}


def main() -> None:
    """Train model, then persist model + metrics + history."""
    args = parse_args()

    # Create a unique run folder so repeated runs do not overwrite each other.
    run_name: str = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir: Path = args.artifacts_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build one immutable config object so run settings are explicit and saved.
    config = TrainConfig(
        gauge_id=args.gauge_id,
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        model_family=args.model_family,
        mobilenet_pretrained=not args.no_mobilenet_pretrained,
        mobilenet_backbone_trainable=args.mobilenet_backbone_trainable,
        mobilenet_warmup_epochs=args.mobilenet_warmup_epochs,
        strict_labels=args.strict_labels,
        crop_pad_ratio=args.crop_pad_ratio,
        augment_training=not args.no_augment_training,
        device=args.device,
        gpu_memory_growth=not args.no_gpu_memory_growth,
        mixed_precision=args.mixed_precision,
    )

    # Execute training and evaluation.
    result = train(config)

    # Save trained model in Keras format.
    model_path: Path = run_dir / "model.keras"
    result.model.save(model_path)

    # Save structured run outputs for analysis and reproducibility.
    history_path: Path = run_dir / "history.json"
    metrics_path: Path = run_dir / "metrics.json"

    history_payload: dict[str, list[float]] = to_serializable_history(
        result.history.history
    )
    metrics_payload: dict[str, Any] = {
        "config": asdict(config),
        "label_summary": asdict(result.label_summary),
        "test_metrics": result.test_metrics,
        "model_path": str(model_path),
    }

    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Print a concise run summary for quick feedback in terminal.
    print(f"Run directory: {run_dir}")
    print(f"Model saved: {model_path}")
    print(f"Label summary: {result.label_summary}")
    print(f"Test metrics: {result.test_metrics}")


if __name__ == "__main__":
    main()
