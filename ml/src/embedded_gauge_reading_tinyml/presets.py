"""Shared preset values for ML training and launch scripts."""

from __future__ import annotations

from typing import Literal


# Dataset and crop settings.
DEFAULT_GAUGE_ID: str = "littlegood_home_temp_gauge_c"
DEFAULT_IMAGE_HEIGHT: int = 224
DEFAULT_IMAGE_WIDTH: int = 224
DEFAULT_CROP_PAD_RATIO: float = 0.25

# Training setup chosen from the strongest known MobileNetV2 preset.
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_EPOCHS: int = 40
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_SEED: int = 21
DEFAULT_VAL_FRACTION: float = 0.15
DEFAULT_TEST_FRACTION: float = 0.15
DEFAULT_STRICT_LABELS: bool = False
DEFAULT_AUGMENT_TRAINING: bool = True
DEFAULT_MODEL_FAMILY: Literal[
    "compact",
    "mobilenet_v2",
    "mobilenet_v2_tiny",
] = "mobilenet_v2"
DEFAULT_MOBILENET_PRETRAINED: bool = True
DEFAULT_MOBILENET_BACKBONE_TRAINABLE: bool = True
DEFAULT_MOBILENET_WARMUP_EPOCHS: int = 8
# MobileNetV2 head settings kept separate so tiny variants can reuse the same
# trainer with a narrower backbone and a lighter classification head.
DEFAULT_MOBILENET_ALPHA: float = 1.0
DEFAULT_MOBILENET_HEAD_UNITS: int = 128
DEFAULT_MOBILENET_HEAD_DROPOUT: float = 0.2
DEFAULT_INTERVAL_BIN_WIDTH: float = 5.0
DEFAULT_INTERPOLATION_PAIR_SCALE: float = 12.0
DEFAULT_ORDINAL_THRESHOLD_STEP: float = 2.5
DEFAULT_ORDINAL_LOSS_WEIGHT: float = 0.5
DEFAULT_SWEEP_FRACTION_LOSS_WEIGHT: float = 0.5
DEFAULT_KEYPOINT_HEATMAP_SIZE: int = 28
DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT: float = 0.5
DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT: float = 1.0
DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT: float = 0.0
DEFAULT_GEOMETRY_UNCERTAINTY_LOSS_WEIGHT: float = 0.25
DEFAULT_GEOMETRY_UNCERTAINTY_LOW_QUANTILE: float = 0.10
DEFAULT_GEOMETRY_UNCERTAINTY_HIGH_QUANTILE: float = 0.90

# Tiny MobileNetV2 preset intended for the STM32N6 memory budget.
DEFAULT_MOBILENET_TINY_ALPHA: float = 0.35
DEFAULT_MOBILENET_TINY_HEAD_UNITS: int = 64
DEFAULT_MOBILENET_TINY_HEAD_DROPOUT: float = 0.15

# Runtime defaults.
DEFAULT_LIBRARY_DEVICE: Literal["auto", "cpu", "gpu"] = "auto"
DEFAULT_CLI_DEVICE: Literal["auto", "cpu", "gpu"] = "gpu"
DEFAULT_GPU_MEMORY_GROWTH: bool = True
DEFAULT_MIXED_PRECISION: bool = False
DEFAULT_EDGE_FOCUS_STRENGTH: float = 0.75
