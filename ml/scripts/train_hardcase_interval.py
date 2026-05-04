"""Train MobileNetV2-interval model with aggressive hard-case focus.

This script trains the strongest known configuration for getting <5C MAE on
hard cases while keeping the MobileNetV2 backbone for STM32N6 deployment.

Key features:
- MobileNetV2-interval head (coarse 5C bins + residual correction)
- Full backbone trainable from scratch (ImageNet warm-start)
- Range-aware sampling: 3x oversampling of cold (-30C) and hot (45-50C) tails
- Hard-case repeat: 12x repetition of known difficult captures
- Long training: 80 epochs with cosine decay
- Aggressive augmentation + edge focus
- Unified dataset: 409 images covering full -30C to 50C range

Usage (from ml/ directory in WSL):
    bash scripts/wsl_ml.sh train-hardcase-interval

Or directly:
    poetry run python scripts/train_hardcase_interval.py
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
    DEFAULT_GAUGE_ID,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MOBILENET_ALPHA,
    DEFAULT_MOBILENET_HEAD_DROPOUT,
    DEFAULT_MOBILENET_HEAD_UNITS,
    DEFAULT_MOBILENET_WARMUP_EPOCHS,
    DEFAULT_INTERVAL_BIN_WIDTH,
    DEFAULT_SEED,
    DEFAULT_EDGE_FOCUS_STRENGTH,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the hard-case focused training run."""
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2-interval with aggressive hard-case focus."
    )
    parser.add_argument("--gauge-id", type=str, default=DEFAULT_GAUGE_ID)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--mobilenet-alpha",
        type=float,
        default=DEFAULT_MOBILENET_ALPHA,
        help="MobileNetV2 width multiplier.",
    )
    parser.add_argument(
        "--mobilenet-head-units",
        type=int,
        default=DEFAULT_MOBILENET_HEAD_UNITS,
        help="Dense layer width after MobileNetV2 backbone.",
    )
    parser.add_argument(
        "--mobilenet-head-dropout",
        type=float,
        default=DEFAULT_MOBILENET_HEAD_DROPOUT,
        help="Dropout rate in the regression head.",
    )
    parser.add_argument(
        "--interval-bin-width",
        type=float,
        default=DEFAULT_INTERVAL_BIN_WIDTH,
        help="Width of coarse temperature bins (default 5C).",
    )
    parser.add_argument(
        "--edge-focus-strength",
        type=float,
        default=1.5,
        help="Weight on extreme values. 0 disables, 1.5 is aggressive.",
    )
    parser.add_argument(
        "--range-aware-sampling",
        action="store_true",
        default=True,
        help="Oversample cold/hot tails (default: enabled).",
    )
    parser.add_argument(
        "--cold-tail-fraction",
        type=float,
        default=0.20,
        help="Fraction of range at cold end to oversample (default: 0.20).",
    )
    parser.add_argument(
        "--hot-tail-fraction",
        type=float,
        default=0.20,
        help="Fraction of range at hot end to oversample (default: 0.20).",
    )
    parser.add_argument(
        "--oversampling-factor",
        type=float,
        default=4.0,
        help="How much more to weight tail samples (default: 4x).",
    )
    parser.add_argument(
        "--hard-case-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv",
        help="CSV of hard cases to upweight.",
    )
    parser.add_argument(
        "--hard-case-repeat",
        type=int,
        default=12,
        help="Repeat count for hard-case rows (default: 12).",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Optional pinned validation manifest.",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=None,
        help="Optional pinned test manifest.",
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
        help="Run folder name. Defaults to timestamp.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default=DEFAULT_CLI_DEVICE,
    )
    parser.add_argument(
        "--no-gpu-memory-growth",
        action="store_true",
        help="Disable TensorFlow GPU memory growth.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed_float16 policy.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the hard-case focused MobileNetV2-interval model."""
    args = parse_args()

    run_name: str = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir: Path = args.artifacts_dir / f"hardcase_interval_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build immutable config with all hard-case optimizations enabled
    config = TrainConfig(
        gauge_id=args.gauge_id,
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        val_fraction=DEFAULT_VAL_FRACTION,
        test_fraction=DEFAULT_TEST_FRACTION,
        strict_labels=False,
        crop_pad_ratio=DEFAULT_CROP_PAD_RATIO,
        augment_training=True,
        device=args.device,
        gpu_memory_growth=not args.no_gpu_memory_growth,
        mixed_precision=args.mixed_precision,
        edge_focus_strength=args.edge_focus_strength,
        model_family="mobilenet_v2_interval",
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=True,
        mobilenet_warmup_epochs=DEFAULT_MOBILENET_WARMUP_EPOCHS,
        mobilenet_alpha=args.mobilenet_alpha,
        mobilenet_head_units=args.mobilenet_head_units,
        mobilenet_head_dropout=args.mobilenet_head_dropout,
        hard_case_manifest=str(args.hard_case_manifest),
        hard_case_repeat=args.hard_case_repeat,
        val_manifest=str(args.val_manifest) if args.val_manifest else None,
        test_manifest=str(args.test_manifest) if args.test_manifest else None,
        range_aware_sampling=args.range_aware_sampling,
        cold_tail_fraction=args.cold_tail_fraction,
        hot_tail_fraction=args.hot_tail_fraction,
        oversampling_factor=args.oversampling_factor,
        interval_bin_width=args.interval_bin_width,
    )

    print("=" * 60)
    print("HARDCASE INTERVAL TRAINING RUN")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Model family:  {config.model_family}")
    print(f"Backbone:      MobileNetV2 (alpha={config.mobilenet_alpha})")
    print(f"Backbone trainable: {config.mobilenet_backbone_trainable}")
    print(f"Epochs:        {config.epochs}")
    print(f"Batch size:    {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Image size:    {config.image_height}x{config.image_width}")
    print(f"Interval bins: {config.interval_bin_width}C")
    print(f"Edge focus:    {config.edge_focus_strength}")
    print(f"Range-aware:   {config.range_aware_sampling}")
    print(f"  Cold tail:   {config.cold_tail_fraction} @ {config.oversampling_factor}x")
    print(f"  Hot tail:    {config.hot_tail_fraction} @ {config.oversampling_factor}x")
    print(f"Hard cases:    {config.hard_case_manifest}")
    print(f"  Repeat:      {config.hard_case_repeat}x")
    print("=" * 60)

    # Save config for reproducibility
    config_path = run_dir / "config.json"
    config_dict = asdict(config)
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[TRAIN] Config saved to {config_path}")

    # Run training
    result = train(config)

    # Save artifacts
    model_path = run_dir / "model.keras"
    result.model.save(model_path)
    print(f"[TRAIN] Model saved to {model_path}")

    history_path = run_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(result.history.history, f, indent=2)
    print(f"[TRAIN] History saved to {history_path}")

    metrics_path = run_dir / "metrics.json"
    metrics = dict(result.test_metrics)
    metrics["config"] = config_dict
    metrics["label_summary"] = asdict(result.label_summary)
    metrics["baseline_test_mae"] = result.baseline_test_mae
    metrics["dropped_out_of_sweep"] = result.dropped_out_of_sweep
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[TRAIN] Metrics saved to {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Test MAE:      {result.test_metrics.get('mae', 'N/A'):.2f}C")
    print(f"Test RMSE:     {result.test_metrics.get('rmse', 'N/A'):.2f}C")
    print(f"Baseline MAE:  {result.baseline_test_mae:.2f}C")
    print(f"Model path:    {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
