"""Run one end-to-end training job with range-aware sampling and linear output.

This script trains a MobileNetV2 model with:
- Range-aware sampling: oversamples cold/hot tail regions for better range coverage
- Linear output head: unbounded regression without saturating activation
- Enhanced augmentation: crop jitter and brightness/exposure matching board reality
- Post-training calibration: affine fit to map linear outputs to calibrated values

Usage:
    poetry run python scripts/run_training_range_aware.py \\
        --epochs 50 \\
        --batch-size 16 \\
        --range-aware-sampling \\
        --linear-output \\
        --artifacts-dir artifacts/training/range_aware_experiment
"""

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
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import TrainConfig, train
from embedded_gauge_reading_tinyml.presets import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLI_DEVICE,
    DEFAULT_CROP_PAD_RATIO,
    DEFAULT_EPOCHS,
    DEFAULT_GAUGE_ID,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MOBILENET_ALPHA,
    DEFAULT_MOBILENET_BACKBONE_TRAINABLE,
    DEFAULT_MOBILENET_HEAD_DROPOUT,
    DEFAULT_MOBILENET_HEAD_UNITS,
    DEFAULT_MOBILENET_WARMUP_EPOCHS,
    DEFAULT_INTERVAL_BIN_WIDTH,
    DEFAULT_INTERPOLATION_PAIR_SCALE,
    DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    DEFAULT_KEYPOINT_HEATMAP_SIZE,
    DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
    DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT,
    DEFAULT_GEOMETRY_UNCERTAINTY_LOSS_WEIGHT,
    DEFAULT_GEOMETRY_UNCERTAINTY_LOW_QUANTILE,
    DEFAULT_GEOMETRY_UNCERTAINTY_HIGH_QUANTILE,
    DEFAULT_ORDINAL_LOSS_WEIGHT,
    DEFAULT_ORDINAL_THRESHOLD_STEP,
    DEFAULT_SWEEP_FRACTION_LOSS_WEIGHT,
    DEFAULT_MODEL_FAMILY,
    DEFAULT_SEED,
    DEFAULT_EDGE_FOCUS_STRENGTH,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a reproducible training run."""
    parser = argparse.ArgumentParser(
        description="Train gauge value regressor with range-aware sampling."
    )
    parser.add_argument("--gauge-id", type=str, default=DEFAULT_GAUGE_ID)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--model-family",
        type=str,
        choices=[
            "compact",
            "mobilenet_v2",
            "mobilenet_v2_tiny",
        ],
        default="mobilenet_v2",
        help="Select model architecture family.",
    )
    parser.add_argument(
        "--mobilenet-alpha",
        type=float,
        default=DEFAULT_MOBILENET_ALPHA,
        help="MobileNetV2 width multiplier. Smaller values shrink the model.",
    )
    parser.add_argument(
        "--mobilenet-head-units",
        type=int,
        default=DEFAULT_MOBILENET_HEAD_UNITS,
        help="Dense layer width used after the MobileNetV2 backbone.",
    )
    parser.add_argument(
        "--mobilenet-head-dropout",
        type=float,
        default=DEFAULT_MOBILENET_HEAD_DROPOUT,
        help="Dropout rate for the MobileNetV2 head.",
    )
    parser.add_argument(
        "--mobilenet-backbone-trainable",
        action="store_true",
        default=True,
        help="Train the MobileNetV2 backbone (default: True).",
    )
    parser.add_argument(
        "--mobilenet-pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet pretrained weights (default: True).",
    )
    parser.add_argument(
        "--edge-focus-strength",
        type=float,
        default=DEFAULT_EDGE_FOCUS_STRENGTH,
        help="Weight strength for edge focus (0.0 to disable).",
    )
    parser.add_argument(
        "--range-aware-sampling",
        action="store_true",
        default=False,
        help="Enable range-aware sampling with cold/hot tail oversampling.",
    )
    parser.add_argument(
        "--cold-tail-fraction",
        type=float,
        default=0.15,
        help="Fraction of range at cold end to oversample (default: 0.15).",
    )
    parser.add_argument(
        "--hot-tail-fraction",
        type=float,
        default=0.15,
        help="Fraction of range at hot end to oversample (default: 0.15).",
    )
    parser.add_argument(
        "--oversampling-factor",
        type=float,
        default=3.0,
        help="How much more to weight tail samples (default: 3x).",
    )
    parser.add_argument(
        "--linear-output",
        action="store_true",
        default=False,
        help="Use linear output head (no saturating activation).",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="Path to validation manifest CSV.",
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        default=None,
        help="Path to test manifest CSV.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Directory to save training artifacts.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run (used in artifact paths).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default=DEFAULT_CLI_DEVICE,
        help="Device to use for training.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of data to use for validation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Fraction of data to use for testing.",
    )

    return parser.parse_args()


def main() -> None:
    """Run one end-to-end training job with range-aware sampling."""
    args = parse_args()

    # Build config from CLI arguments
    config = TrainConfig(
        gauge_id=args.gauge_id,
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        crop_pad_ratio=DEFAULT_CROP_PAD_RATIO,
        augment_training=True,  # Always enable augmentation for this experiment
        device=args.device,
        gpu_memory_growth=True,
        mixed_precision=False,
        edge_focus_strength=args.edge_focus_strength,
        model_family=args.model_family,
        mobilenet_pretrained=args.mobilenet_pretrained,
        mobilenet_backbone_trainable=args.mobilenet_backbone_trainable,
        mobilenet_alpha=args.mobilenet_alpha,
        mobilenet_head_units=args.mobilenet_head_units,
        mobilenet_head_dropout=args.mobilenet_head_dropout,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        range_aware_sampling=args.range_aware_sampling,
        cold_tail_fraction=args.cold_tail_fraction,
        hot_tail_fraction=args.hot_tail_fraction,
        oversampling_factor=args.oversampling_factor,
        linear_output=args.linear_output,
    )

    # Build artifact directory
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        # Default to artifacts/training with run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or f"range_aware_{timestamp}"
        artifacts_dir = REPO_ROOT / "artifacts" / "training" / run_name

    # Ensure artifacts directory exists
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save config to artifacts
    config_path = artifacts_dir / "config.json"
    config_dict = asdict(config)
    # Convert Path objects to strings for JSON serialization
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"[TRAIN] Starting training run: {run_name}")
    print(f"[TRAIN] Artifacts will be saved to: {artifacts_dir}")
    print(f"[TRAIN] Config: {json.dumps(config_dict, indent=2)}")

    # Run training
    result = train(config)

    # Save model
    model_path = artifacts_dir / "model.keras"
    result.model.save(model_path)
    print(f"[TRAIN] Saved model to: {model_path}")

    # Save training history
    history_path = artifacts_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(result.history.history, f, indent=2)
    print(f"[TRAIN] Saved history to: {history_path}")

    # Save test metrics
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result.test_metrics, f, indent=2)
    print(f"[TRAIN] Saved metrics to: {metrics_path}")

    # Print summary
    print("\n[TRAIN] Training complete!")
    print(f"[TRAIN] Test MAE: {result.test_metrics.get('mae', 'N/A')}C")
    if "calibrated_mae" in result.test_metrics:
        print(f"[TRAIN] Calibrated MAE: {result.test_metrics['calibrated_mae']}C")
        print(f"[TRAIN] Calibration slope: {result.test_metrics['calibration_slope']}")
        print(f"[TRAIN] Calibration bias: {result.test_metrics['calibration_bias']}")
    print(f"[TRAIN] Baseline MAE: {result.baseline_test_mae}C")


if __name__ == "__main__":
    main()
