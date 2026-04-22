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
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a reproducible training run."""
    parser = argparse.ArgumentParser(description="Train gauge value regressor.")
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
            "compact_direction",
            "compact_interval",
            "compact_geometry",
            "mobilenet_v2",
            "mobilenet_v2_tiny",
            "mobilenet_v2_direction",
            "mobilenet_v2_fraction",
            "mobilenet_v2_detector",
            "mobilenet_v2_geometry",
            "mobilenet_v2_geometry_uncertainty",
            "mobilenet_v2_rectifier",
            "mobilenet_v2_keypoint",
            "mobilenet_v2_interval",
            "mobilenet_v2_ordinal",
        ],
        default=DEFAULT_MODEL_FAMILY,
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
        help="Dropout rate applied in the MobileNetV2 regression head.",
    )
    parser.add_argument(
        "--mobilenet-backbone-trainable",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MOBILENET_BACKBONE_TRAINABLE,
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
        default=DEFAULT_MOBILENET_WARMUP_EPOCHS,
        help="Warmup epochs with frozen MobileNetV2 backbone before fine-tuning.",
    )
    parser.add_argument("--strict-labels", action="store_true")
    parser.add_argument(
        "--crop-pad-ratio", type=float, default=DEFAULT_CROP_PAD_RATIO
    )
    parser.add_argument("--no-augment-training", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default=DEFAULT_CLI_DEVICE,
        help="Select accelerator mode. Default is 'gpu' for the best-performing setup.",
    )
    parser.add_argument(
        "--no-gpu-memory-growth",
        action="store_true",
        help="Disable TensorFlow GPU memory growth.",
    )
    parser.add_argument(
        "--hard-case-manifest",
        type=Path,
        default=None,
        help="Optional CSV manifest of extra hard board captures to upweight.",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Optional CSV manifest pinned as the validation set. Overrides random val split.",
    )
    parser.add_argument(
        "--hard-case-repeat",
        type=int,
        default=0,
        help="Repeat count for each hard-case row when fine-tuning.",
    )
    parser.add_argument(
        "--edge-focus-strength",
        type=float,
        default=DEFAULT_EDGE_FOCUS_STRENGTH,
        help="Additional weight placed on extreme gauge values (0 disables).",
    )
    parser.add_argument(
        "--rectifier-model-path",
        type=Path,
        default=None,
        help="Optional rectifier model used to generate rectified scalar crops.",
    )
    parser.add_argument(
        "--precomputed-crop-boxes",
        type=Path,
        default=None,
        help="CSV of precomputed rectifier crop boxes (image_path,value,x0,y0,x1,y1). "
             "Faster alternative to --rectifier-model-path; skips in-process rectifier inference.",
    )
    parser.add_argument(
        "--rectifier-crop-scale",
        type=float,
        default=1.5,
        help="Scale factor applied when generating rectified scalar crops.",
    )
    parser.add_argument(
        "--monotonic-pair-strength",
        type=float,
        default=0.0,
        help="Extra loss weight that penalizes ordering violations inside a batch.",
    )
    parser.add_argument(
        "--monotonic-pair-margin",
        type=float,
        default=0.0,
        help="Minimum prediction gap encouraged between ordered batch samples.",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="MixUp Beta distribution alpha for scalar regression training.",
    )
    parser.add_argument(
        "--interval-bin-width",
        type=float,
        default=DEFAULT_INTERVAL_BIN_WIDTH,
        help="Width of the coarse temperature bins used by the hybrid interval head.",
    )
    parser.add_argument(
        "--ordinal-threshold-step",
        type=float,
        default=DEFAULT_ORDINAL_THRESHOLD_STEP,
        help="Spacing between ordinal thresholds for the ordered temperature head.",
    )
    parser.add_argument(
        "--ordinal-loss-weight",
        type=float,
        default=DEFAULT_ORDINAL_LOSS_WEIGHT,
        help="Loss weight applied to the ordinal threshold branch.",
    )
    parser.add_argument(
        "--sweep-fraction-loss-weight",
        type=float,
        default=DEFAULT_SWEEP_FRACTION_LOSS_WEIGHT,
        help="Loss weight applied to the sweep-fraction branch.",
    )
    parser.add_argument(
        "--keypoint-heatmap-size",
        type=int,
        default=DEFAULT_KEYPOINT_HEATMAP_SIZE,
        help="Resolution of the keypoint heatmap supervision grid.",
    )
    parser.add_argument(
        "--keypoint-heatmap-loss-weight",
        type=float,
        default=DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
        help="Loss weight applied to the keypoint heatmap branch.",
    )
    parser.add_argument(
        "--keypoint-coord-loss-weight",
        type=float,
        default=DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
        help="Loss weight applied to the keypoint coordinate branch.",
    )
    parser.add_argument(
        "--geometry-value-loss-weight",
        type=float,
        default=DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT,
        help="Loss weight applied to the derived scalar value branch for geometry models.",
    )
    parser.add_argument(
        "--geometry-uncertainty-loss-weight",
        type=float,
        default=DEFAULT_GEOMETRY_UNCERTAINTY_LOSS_WEIGHT,
        help="Loss weight applied to the lower/upper uncertainty branches.",
    )
    parser.add_argument(
        "--geometry-uncertainty-low-quantile",
        type=float,
        default=DEFAULT_GEOMETRY_UNCERTAINTY_LOW_QUANTILE,
        help="Lower quantile used for the geometry uncertainty interval.",
    )
    parser.add_argument(
        "--geometry-uncertainty-high-quantile",
        type=float,
        default=DEFAULT_GEOMETRY_UNCERTAINTY_HIGH_QUANTILE,
        help="Upper quantile used for the geometry uncertainty interval.",
    )
    parser.add_argument(
        "--interpolation-pair-strength",
        type=float,
        default=0.0,
        help="Extra loss weight that penalizes local slope mismatches inside a batch.",
    )
    parser.add_argument(
        "--interpolation-pair-scale",
        type=float,
        default=DEFAULT_INTERPOLATION_PAIR_SCALE,
        help="Temperature scale that controls how far interpolation pairs reach.",
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Optional warm-start Keras model to fine-tune instead of building from scratch.",
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
        mobilenet_alpha=args.mobilenet_alpha,
        mobilenet_head_units=args.mobilenet_head_units,
        mobilenet_head_dropout=args.mobilenet_head_dropout,
        hard_case_manifest=str(args.hard_case_manifest) if args.hard_case_manifest else None,
        hard_case_repeat=args.hard_case_repeat,
        val_manifest=str(args.val_manifest) if args.val_manifest else None,
        init_model_path=str(args.init_model) if args.init_model else None,
        strict_labels=args.strict_labels,
        crop_pad_ratio=args.crop_pad_ratio,
        augment_training=not args.no_augment_training,
        device=args.device,
        gpu_memory_growth=not args.no_gpu_memory_growth,
        mixed_precision=args.mixed_precision,
        edge_focus_strength=args.edge_focus_strength,
        rectifier_model_path=str(args.rectifier_model_path) if args.rectifier_model_path else None,
        rectifier_crop_scale=args.rectifier_crop_scale,
        precomputed_crop_boxes_path=str(args.precomputed_crop_boxes) if args.precomputed_crop_boxes else None,
        monotonic_pair_strength=args.monotonic_pair_strength,
        monotonic_pair_margin=args.monotonic_pair_margin,
        mixup_alpha=args.mixup_alpha,
        interval_bin_width=args.interval_bin_width,
        interpolation_pair_strength=args.interpolation_pair_strength,
        interpolation_pair_scale=args.interpolation_pair_scale,
        ordinal_threshold_step=args.ordinal_threshold_step,
        ordinal_loss_weight=args.ordinal_loss_weight,
        sweep_fraction_loss_weight=args.sweep_fraction_loss_weight,
        keypoint_heatmap_size=args.keypoint_heatmap_size,
        keypoint_heatmap_loss_weight=args.keypoint_heatmap_loss_weight,
        keypoint_coord_loss_weight=args.keypoint_coord_loss_weight,
        geometry_value_loss_weight=args.geometry_value_loss_weight,
        geometry_uncertainty_loss_weight=args.geometry_uncertainty_loss_weight,
        geometry_uncertainty_low_quantile=args.geometry_uncertainty_low_quantile,
        geometry_uncertainty_high_quantile=args.geometry_uncertainty_high_quantile,
    )

    print(
        "[RUN] Training job: "
        f"model_family={config.model_family} "
        f"image_size={config.image_height}x{config.image_width} "
        f"epochs={config.epochs} "
        f"batch_size={config.batch_size} "
        f"device={config.device}"
    )
    print(
        "[RUN] MobileNetV2 options: "
        f"pretrained={config.mobilenet_pretrained} "
        f"backbone_trainable={config.mobilenet_backbone_trainable} "
        f"warmup_epochs={config.mobilenet_warmup_epochs} "
        f"alpha={config.mobilenet_alpha} "
        f"head_units={config.mobilenet_head_units} "
        f"head_dropout={config.mobilenet_head_dropout} "
        f"init_model={config.init_model_path}"
    )
    print(
        "[RUN] Monotonic regularizer: "
        f"pair_strength={config.monotonic_pair_strength} "
        f"pair_margin={config.monotonic_pair_margin}"
    )
    print(
        "[RUN] MixUp / interval: "
        f"mixup_alpha={config.mixup_alpha} "
        f"interval_bin_width={config.interval_bin_width}"
    )
    print(
        "[RUN] Interpolation pairs: "
        f"pair_strength={config.interpolation_pair_strength} "
        f"pair_scale={config.interpolation_pair_scale}"
    )
    print(
        "[RUN] Ordinal head: "
        f"threshold_step={config.ordinal_threshold_step} "
        f"loss_weight={config.ordinal_loss_weight}"
    )
    print(
        "[RUN] Sweep fraction head: "
        f"loss_weight={config.sweep_fraction_loss_weight}"
    )
    head_label = (
        "Rectifier head"
        if config.model_family == "mobilenet_v2_rectifier"
        else
        "Geometry head"
        if config.model_family in {
            "mobilenet_v2_detector",
            "mobilenet_v2_geometry",
            "mobilenet_v2_geometry_uncertainty",
        }
        else "Keypoint head"
    )
    print(
        f"[RUN] {head_label}: "
        f"heatmap_size={config.keypoint_heatmap_size} "
        f"heatmap_loss_weight={config.keypoint_heatmap_loss_weight} "
        f"coord_loss_weight={config.keypoint_coord_loss_weight} "
        f"value_loss_weight={config.geometry_value_loss_weight}"
    )
    if config.model_family == "mobilenet_v2_geometry_uncertainty":
        print(
            "[RUN] Geometry uncertainty head: "
            f"loss_weight={config.geometry_uncertainty_loss_weight} "
            f"low_q={config.geometry_uncertainty_low_quantile} "
            f"high_q={config.geometry_uncertainty_high_quantile}"
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
