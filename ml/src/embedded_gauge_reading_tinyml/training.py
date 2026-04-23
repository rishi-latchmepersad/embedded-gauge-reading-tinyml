"""Training pipeline for gauge-value regression with dial ROI cropping."""

from __future__ import annotations

from dataclasses import replace
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Literal

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from embedded_gauge_reading_tinyml.dataset import Sample, load_dataset
from embedded_gauge_reading_tinyml.board_crop_compare import (
    load_rgb_image,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.gauge.processing import (
    GaugeSpec,
    load_gauge_specs,
    needle_fraction,
    needle_unit_xy_from_value,
    needle_value,
)
from embedded_gauge_reading_tinyml.labels import LabelSummary, summarize_label_sweep
from embedded_gauge_reading_tinyml.presets import (
    DEFAULT_AUGMENT_TRAINING,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CROP_PAD_RATIO,
    DEFAULT_EPOCHS,
    DEFAULT_GAUGE_ID,
    DEFAULT_GPU_MEMORY_GROWTH,
    DEFAULT_MOBILENET_ALPHA,
    DEFAULT_MOBILENET_HEAD_DROPOUT,
    DEFAULT_MOBILENET_HEAD_UNITS,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LIBRARY_DEVICE,
    DEFAULT_MOBILENET_BACKBONE_TRAINABLE,
    DEFAULT_MOBILENET_PRETRAINED,
    DEFAULT_MOBILENET_WARMUP_EPOCHS,
    DEFAULT_MIXED_PRECISION,
    DEFAULT_MODEL_FAMILY,
    DEFAULT_INTERVAL_BIN_WIDTH,
    DEFAULT_INTERPOLATION_PAIR_SCALE,
    DEFAULT_ORDINAL_LOSS_WEIGHT,
    DEFAULT_ORDINAL_THRESHOLD_STEP,
    DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    DEFAULT_KEYPOINT_HEATMAP_SIZE,
    DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
    DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT,
    DEFAULT_GEOMETRY_UNCERTAINTY_LOSS_WEIGHT,
    DEFAULT_GEOMETRY_UNCERTAINTY_LOW_QUANTILE,
    DEFAULT_GEOMETRY_UNCERTAINTY_HIGH_QUANTILE,
    DEFAULT_SWEEP_FRACTION_LOSS_WEIGHT,
    DEFAULT_SEED,
    DEFAULT_STRICT_LABELS,
    DEFAULT_EDGE_FOCUS_STRENGTH,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
)
from embedded_gauge_reading_tinyml.models import (
    GaugeValueFromKeypoints,
    SpatialSoftArgmax2D,
    build_compact_interval_model,
    build_compact_geometry_model,
    build_mobilenetv2_rectifier_model,
    build_mobilenetv2_obb_model,
    build_mobilenetv2_regression_model,
    build_mobilenetv2_direction_model,
    build_mobilenetv2_fraction_model,
    build_mobilenetv2_detector_model,
    build_mobilenetv2_geometry_model,
    build_mobilenetv2_geometry_uncertainty_model,
    build_mobilenetv2_keypoint_model,
    build_mobilenetv2_interval_model,
    build_mobilenetv2_ordinal_model,
    build_needle_direction_model,
    build_regression_model,
)


# Resolve ML/data paths from this file so training works from any cwd.
ML_ROOT: Path = Path(__file__).resolve().parents[2]
REPO_ROOT: Path = ML_ROOT.parent
LABELLED_DIR: Path = ML_ROOT / "data" / "labelled"
RAW_DIR: Path = ML_ROOT / "data" / "raw"


@dataclass(frozen=True)
class TrainConfig:
    """Config values for reproducible training runs.

    The defaults track the strongest known MobileNetV2 preset on this dataset,
    while callers can still override individual fields for ablations or the
    compact CNN baseline.
    """

    gauge_id: str = DEFAULT_GAUGE_ID
    image_height: int = DEFAULT_IMAGE_HEIGHT
    image_width: int = DEFAULT_IMAGE_WIDTH
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    seed: int = DEFAULT_SEED
    val_fraction: float = DEFAULT_VAL_FRACTION
    test_fraction: float = DEFAULT_TEST_FRACTION
    strict_labels: bool = DEFAULT_STRICT_LABELS
    crop_pad_ratio: float = DEFAULT_CROP_PAD_RATIO
    augment_training: bool = DEFAULT_AUGMENT_TRAINING
    device: Literal["auto", "cpu", "gpu"] = DEFAULT_LIBRARY_DEVICE
    gpu_memory_growth: bool = DEFAULT_GPU_MEMORY_GROWTH
    mixed_precision: bool = DEFAULT_MIXED_PRECISION
    edge_focus_strength: float = DEFAULT_EDGE_FOCUS_STRENGTH
    rectifier_model_path: str | None = None
    rectifier_crop_scale: float = 1.5
    precomputed_crop_boxes_path: str | None = None
    model_family: Literal[
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
        "mobilenet_v2_keypoint",
        "mobilenet_v2_obb",
        "mobilenet_v2_interval",
        "mobilenet_v2_ordinal",
    ] = (
        DEFAULT_MODEL_FAMILY
    )
    mobilenet_pretrained: bool = DEFAULT_MOBILENET_PRETRAINED
    mobilenet_backbone_trainable: bool = DEFAULT_MOBILENET_BACKBONE_TRAINABLE
    mobilenet_warmup_epochs: int = DEFAULT_MOBILENET_WARMUP_EPOCHS
    mobilenet_alpha: float = DEFAULT_MOBILENET_ALPHA
    mobilenet_head_units: int = DEFAULT_MOBILENET_HEAD_UNITS
    mobilenet_head_dropout: float = DEFAULT_MOBILENET_HEAD_DROPOUT
    hard_case_manifest: str | None = None
    hard_case_repeat: int = 0
    val_manifest: str | None = None
    test_manifest: str | None = None
    init_model_path: str | None = None
    monotonic_pair_strength: float = 0.0
    monotonic_pair_margin: float = 0.0
    mixup_alpha: float = 0.0
    interval_bin_width: float = DEFAULT_INTERVAL_BIN_WIDTH
    interpolation_pair_strength: float = 0.0
    interpolation_pair_scale: float = DEFAULT_INTERPOLATION_PAIR_SCALE
    ordinal_threshold_step: float = DEFAULT_ORDINAL_THRESHOLD_STEP
    ordinal_loss_weight: float = DEFAULT_ORDINAL_LOSS_WEIGHT
    sweep_fraction_loss_weight: float = DEFAULT_SWEEP_FRACTION_LOSS_WEIGHT
    keypoint_heatmap_size: int = DEFAULT_KEYPOINT_HEATMAP_SIZE
    keypoint_heatmap_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT
    keypoint_coord_loss_weight: float = DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT
    geometry_value_loss_weight: float = DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT
    geometry_uncertainty_loss_weight: float = (
        DEFAULT_GEOMETRY_UNCERTAINTY_LOSS_WEIGHT
    )
    geometry_uncertainty_low_quantile: float = (
        DEFAULT_GEOMETRY_UNCERTAINTY_LOW_QUANTILE
    )
    geometry_uncertainty_high_quantile: float = (
        DEFAULT_GEOMETRY_UNCERTAINTY_HIGH_QUANTILE
    )


@dataclass(frozen=True)
class TrainingExample:
    """One training row with image path, targets, and dial crop box."""

    image_path: str
    value: float
    crop_box_xyxy: tuple[float, float, float, float]
    needle_unit_xy: tuple[float, float]
    value_norm: float = 0.0
    center_xy: tuple[float, float] = (0.0, 0.0)
    tip_xy: tuple[float, float] = (0.0, 0.0)
    keypoint_heatmaps: np.ndarray | None = None
    keypoint_coords: np.ndarray | None = None
    obb_params: np.ndarray | None = None


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train/val/test example splits."""

    train_examples: list[TrainingExample]
    val_examples: list[TrainingExample]
    test_examples: list[TrainingExample]


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts returned after one training run."""

    model: keras.Model
    history: keras.callbacks.History
    label_summary: LabelSummary
    test_metrics: dict[str, float]
    baseline_test_mae: float
    dropped_out_of_sweep: int


def _validate_split_config(config: TrainConfig) -> None:
    """Guard against invalid split fractions."""
    if not (0.0 < config.val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1).")
    if not (0.0 < config.test_fraction < 1.0):
        raise ValueError("test_fraction must be in (0, 1).")
    if config.val_fraction + config.test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0.")
    if config.mobilenet_warmup_epochs < 0:
        raise ValueError("mobilenet_warmup_epochs must be >= 0.")
    if config.mobilenet_alpha <= 0.0:
        raise ValueError("mobilenet_alpha must be > 0.")
    if config.mobilenet_head_units <= 0:
        raise ValueError("mobilenet_head_units must be > 0.")
    if not (0.0 <= config.mobilenet_head_dropout < 1.0):
        raise ValueError("mobilenet_head_dropout must be in [0, 1).")
    if config.rectifier_crop_scale <= 0.0:
        raise ValueError("rectifier_crop_scale must be > 0.")
    if config.monotonic_pair_strength < 0.0:
        raise ValueError("monotonic_pair_strength must be >= 0.")
    if config.monotonic_pair_margin < 0.0:
        raise ValueError("monotonic_pair_margin must be >= 0.")
    if config.mixup_alpha < 0.0:
        raise ValueError("mixup_alpha must be >= 0.")
    if config.interval_bin_width <= 0.0:
        raise ValueError("interval_bin_width must be > 0.")
    if config.interpolation_pair_strength < 0.0:
        raise ValueError("interpolation_pair_strength must be >= 0.")
    if config.interpolation_pair_scale <= 0.0:
        raise ValueError("interpolation_pair_scale must be > 0.")
    if config.ordinal_threshold_step <= 0.0:
        raise ValueError("ordinal_threshold_step must be > 0.")
    if config.ordinal_loss_weight < 0.0:
        raise ValueError("ordinal_loss_weight must be >= 0.")
    if config.sweep_fraction_loss_weight < 0.0:
        raise ValueError("sweep_fraction_loss_weight must be >= 0.")
    if config.keypoint_heatmap_size < 4:
        raise ValueError("keypoint_heatmap_size must be >= 4.")
    if config.keypoint_heatmap_loss_weight < 0.0:
        raise ValueError("keypoint_heatmap_loss_weight must be >= 0.")
    if config.geometry_value_loss_weight < 0.0:
        raise ValueError("geometry_value_loss_weight must be >= 0.")
    if config.geometry_uncertainty_loss_weight < 0.0:
        raise ValueError("geometry_uncertainty_loss_weight must be >= 0.")
    if not (0.0 < config.geometry_uncertainty_low_quantile < 0.5):
        raise ValueError(
            "geometry_uncertainty_low_quantile must be in (0, 0.5)."
        )
    if not (0.5 < config.geometry_uncertainty_high_quantile < 1.0):
        raise ValueError(
            "geometry_uncertainty_high_quantile must be in (0.5, 1)."
        )
    if config.geometry_uncertainty_low_quantile >= config.geometry_uncertainty_high_quantile:
        raise ValueError(
            "geometry_uncertainty_low_quantile must be < geometry_uncertainty_high_quantile."
        )


def _configure_training_runtime(config: TrainConfig) -> None:
    """Configure TensorFlow runtime for CPU/GPU selection and memory behavior."""
    if config.device == "cpu":
        # Force CPU execution without paying the cost of an eager GPU probe.
        # Some WSL CUDA stacks stall while enumerating devices, so keep the
        # CPU path as light as possible.
        try:
            tf.config.set_visible_devices([], "GPU")
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Unable to force CPU mode because TensorFlow runtime is already initialized."
            ) from exc
        keras.mixed_precision.set_global_policy("float32")
        return

    # When memory growth is enabled, we need to enumerate GPUs so TensorFlow
    # can opt into the right allocator behavior. When it is disabled, skip the
    # probe entirely because some WSL GPU stacks stall here even though the rest
    # of the training job is fine.
    if config.gpu_memory_growth:
        gpus: list[tf.config.PhysicalDevice] = tf.config.list_physical_devices(
            "GPU"
        )
        if config.device == "gpu" and not gpus:
            raise ValueError(
                "device='gpu' was requested, but TensorFlow did not detect a GPU."
            )
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Best effort only: tests and REPL sessions may initialize runtime first.
                pass

    if config.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        keras.mixed_precision.set_global_policy("float32")


def _log_runtime_state(config: TrainConfig) -> None:
    """Print the TensorFlow runtime state we care about before training starts."""
    print(
        "[TRAIN] Runtime: "
        f"device={config.device} "
        f"gpu_memory_growth={config.gpu_memory_growth} "
        f"mixed_precision={keras.mixed_precision.global_policy().name}"
    )
    print("[TRAIN] TensorFlow device probe skipped to avoid WSL stalls.")


def _log_dataset_state(
    config: TrainConfig,
    *,
    label_summary: LabelSummary,
    split: DatasetSplit,
    dropped_out_of_sweep: int,
) -> None:
    """Print dataset and split sizes so long runs stay easy to monitor."""
    print(
        "[TRAIN] Dataset: "
        f"gauge_id={config.gauge_id} "
        f"labelled_samples={label_summary.total_samples} "
        f"in_sweep={label_summary.in_sweep} "
        f"out_of_sweep={label_summary.out_of_sweep} "
        f"dropped_after_filter={dropped_out_of_sweep}"
    )
    print(
        "[TRAIN] Split sizes: "
        f"train={len(split.train_examples)} "
        f"val={len(split.val_examples)} "
        f"test={len(split.test_examples)}"
    )


def _log_model_choice(config: TrainConfig) -> None:
    """Print the selected model family and input geometry before building it."""
    print(
        "[TRAIN] Model: "
        f"family={config.model_family} "
        f"image_size={config.image_height}x{config.image_width} "
        f"batch_size={config.batch_size} "
        f"epochs={config.epochs}"
    )


def _compute_crop_box(
    sample: Sample, pad_ratio: float
) -> tuple[float, float, float, float]:
    """Build an axis-aligned dial crop box from ellipse labels (xyxy in pixels)."""
    pad_x: float = sample.dial.rx * pad_ratio
    pad_y: float = sample.dial.ry * pad_ratio

    x_min: float = sample.dial.cx - sample.dial.rx - pad_x
    y_min: float = sample.dial.cy - sample.dial.ry - pad_y
    x_max: float = sample.dial.cx + sample.dial.rx + pad_x
    y_max: float = sample.dial.cy + sample.dial.ry + pad_y
    return (x_min, y_min, x_max, y_max)


def _map_point_to_resized_crop_xy(
    *,
    point_xy: tuple[float, float],
    crop_box_xyxy: tuple[float, float, float, float],
    image_height: int,
    image_width: int,
) -> tuple[float, float]:
    """Map a point from the original image into the resized crop frame.

    We mirror the crop + resize_with_pad geometry used by the input pipeline so
    keypoint heatmaps line up with the network input coordinates.
    """
    x_min, y_min, x_max, y_max = crop_box_xyxy
    crop_w: float = max(x_max - x_min, 1.0)
    crop_h: float = max(y_max - y_min, 1.0)
    scale: float = min(image_width / crop_w, image_height / crop_h)
    resized_w: float = crop_w * scale
    resized_h: float = crop_h * scale
    pad_x: float = 0.5 * float(image_width - resized_w)
    pad_y: float = 0.5 * float(image_height - resized_h)

    point_x, point_y = point_xy
    out_x: float = (point_x - x_min) * scale + pad_x
    out_y: float = (point_y - y_min) * scale + pad_y
    return out_x, out_y


def _make_gaussian_heatmap(
    *,
    point_xy: tuple[float, float],
    heatmap_size: int,
    sigma: float = 1.5,
) -> np.ndarray:
    """Build a single Gaussian heatmap centered on the given point."""
    if heatmap_size < 4:
        raise ValueError("heatmap_size must be >= 4.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")

    center_x, center_y = point_xy
    yy, xx = np.meshgrid(
        np.arange(heatmap_size, dtype=np.float32),
        np.arange(heatmap_size, dtype=np.float32),
        indexing="ij",
    )
    heatmap: np.ndarray = np.exp(
        -(
            (xx - np.float32(center_x)) ** 2
            + (yy - np.float32(center_y)) ** 2
        )
        / (2.0 * np.float32(sigma) ** 2)
    )
    max_value: float = float(np.max(heatmap))
    if max_value > 0.0:
        heatmap = heatmap / max_value
    return heatmap.astype(np.float32)


def _make_keypoint_heatmaps(
    *,
    center_xy: tuple[float, float],
    tip_xy: tuple[float, float],
    heatmap_size: int,
    sigma: float = 1.5,
) -> np.ndarray:
    """Build the two-channel heatmap target used by the keypoint model."""
    center_heatmap = _make_gaussian_heatmap(
        point_xy=center_xy,
        heatmap_size=heatmap_size,
        sigma=sigma,
    )
    tip_heatmap = _make_gaussian_heatmap(
        point_xy=tip_xy,
        heatmap_size=heatmap_size,
        sigma=sigma,
    )
    return np.stack([center_heatmap, tip_heatmap], axis=-1)


def _build_training_examples(
    samples: list[Sample],
    spec: GaugeSpec,
    *,
    image_height: int,
    image_width: int,
    keypoint_heatmap_size: int,
    strict_labels: bool,
    crop_pad_ratio: float,
) -> tuple[list[TrainingExample], int]:
    """Convert raw samples into trainable examples and drop out-of-sweep labels."""
    examples: list[TrainingExample] = []
    dropped_out_of_sweep: int = 0

    for sample in samples:
        try:
            # Always validate sweep first to avoid training on invalid target geometry.
            needle_fraction(sample, spec, strict=True)
        except ValueError:
            dropped_out_of_sweep += 1
            continue

        value: float = needle_value(sample, spec, strict=strict_labels)
        crop_box: tuple[float, float, float, float] = _compute_crop_box(
            sample, crop_pad_ratio
        )
        value_norm = 0.0
        if spec.max_value > spec.min_value:
            value_norm = (value - spec.min_value) / (spec.max_value - spec.min_value)
        dx: float = sample.tip.x - sample.center.x
        dy: float = sample.tip.y - sample.center.y
        length: float = math.hypot(dx, dy)
        if length <= 0.0:
            dropped_out_of_sweep += 1
            continue
        needle_unit_xy: tuple[float, float] = (dx / length, dy / length)

        center_xy: tuple[float, float] = _map_point_to_resized_crop_xy(
            point_xy=(sample.center.x, sample.center.y),
            crop_box_xyxy=crop_box,
            image_height=image_height,
            image_width=image_width,
        )
        tip_xy: tuple[float, float] = _map_point_to_resized_crop_xy(
            point_xy=(sample.tip.x, sample.tip.y),
            crop_box_xyxy=crop_box,
            image_height=image_height,
            image_width=image_width,
        )
        dial_center_xy: tuple[float, float] = _map_point_to_resized_crop_xy(
            point_xy=(sample.dial.cx, sample.dial.cy),
            crop_box_xyxy=crop_box,
            image_height=image_height,
            image_width=image_width,
        )
        crop_w = max(crop_box[2] - crop_box[0], 1.0)
        crop_h = max(crop_box[3] - crop_box[1], 1.0)
        scale = min(image_width / crop_w, image_height / crop_h)
        obb_center_x = float(dial_center_xy[0] / max(image_width, 1))
        obb_center_y = float(dial_center_xy[1] / max(image_height, 1))
        obb_width = float((2.0 * sample.dial.rx * scale) / max(image_width, 1))
        obb_height = float((2.0 * sample.dial.ry * scale) / max(image_height, 1))
        rotation_rad = math.radians(sample.dial.rotation)
        obb_angle_cos = math.cos(2.0 * rotation_rad)
        obb_angle_sin = math.sin(2.0 * rotation_rad)
        obb_params = np.array(
            [
                np.clip(obb_center_x, 0.0, 1.0),
                np.clip(obb_center_y, 0.0, 1.0),
                np.clip(obb_width, 0.0, 1.0),
                np.clip(obb_height, 0.0, 1.0),
                obb_angle_cos,
                obb_angle_sin,
            ],
            dtype=np.float32,
        )
        scale_x: float = (keypoint_heatmap_size - 1.0) / max(image_width - 1.0, 1.0)
        scale_y: float = (keypoint_heatmap_size - 1.0) / max(image_height - 1.0, 1.0)
        keypoint_heatmaps: np.ndarray = _make_keypoint_heatmaps(
            center_xy=(center_xy[0] * scale_x, center_xy[1] * scale_y),
            tip_xy=(tip_xy[0] * scale_x, tip_xy[1] * scale_y),
            heatmap_size=keypoint_heatmap_size,
        )
        keypoint_coords: np.ndarray = np.array(
            [
                [center_xy[0] * scale_x, center_xy[1] * scale_y],
                [tip_xy[0] * scale_x, tip_xy[1] * scale_y],
            ],
            dtype=np.float32,
        )
        examples.append(
            TrainingExample(
                image_path=str(sample.image_path),
                value=value,
                crop_box_xyxy=crop_box,
                needle_unit_xy=needle_unit_xy,
                value_norm=value_norm,
                center_xy=center_xy,
                tip_xy=tip_xy,
                keypoint_heatmaps=keypoint_heatmaps,
                keypoint_coords=keypoint_coords,
                obb_params=obb_params,
            )
        )

    return examples, dropped_out_of_sweep


def _load_hard_case_examples(
    manifest_path: Path,
    *,
    image_height: int,
    image_width: int,
    value_range: tuple[float, float] = (0.0, 1.0),
    target_kind: Literal[
        "value",
        "needle_unit_xy",
        "ordinal_thresholds",
        "geometry",
        "geometry_uncertainty",
        "rectifier_box",
    ] = "value",
    spec: GaugeSpec | None = None,
    example_lookup: dict[str, TrainingExample] | None = None,
) -> list[TrainingExample]:
    """Load extra board captures that should be upweighted during fine-tuning."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Hard-case manifest not found: {manifest_path}")

    if target_kind == "needle_unit_xy" and spec is None:
        raise ValueError("spec is required when target_kind='needle_unit_xy'.")
    examples: list[TrainingExample] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Hard-case manifest has no header: {manifest_path}")
        required_columns: set[str] = {"image_path", "value"}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "Hard-case manifest must include image_path and value columns."
            )

        for row in reader:
            raw_path = Path(row["image_path"])
            image_path = raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path)
            value = float(row["value"])
            min_value, max_value = value_range
            span = max_value - min_value
            value_norm = 0.0
            if span > 0:
                value_norm = min(max((value - min_value) / span, 0.0), 1.0)
            if target_kind == "needle_unit_xy":
                assert spec is not None
                needle_unit_xy = needle_unit_xy_from_value(value, spec)
            else:
                needle_unit_xy = (1.0, 0.0)
            if target_kind == "rectifier_box":
                # Rectifier export only needs the image tensors themselves.
                # Use the full frame as a dummy target box so captures outside
                # the labelled dataset can still contribute calibration images.
                crop_box_xyxy = (
                    0.0,
                    0.0,
                    float(image_width),
                    float(image_height),
                )
            else:
                crop_box_xyxy = (0.0, 0.0, float(image_width), float(image_height))
            examples.append(
                TrainingExample(
                    image_path=str(image_path),
                    value=value,
                    value_norm=value_norm,
                    crop_box_xyxy=crop_box_xyxy,
                    needle_unit_xy=needle_unit_xy,
                )
            )

    return examples


def _load_manifest_examples_for_split(
    examples: list[TrainingExample],
    manifest_path: Path,
    *,
    image_height: int,
    image_width: int,
) -> tuple[list[TrainingExample], set[str]]:
    """Match a manifest against in-memory examples, or load standalone rows.

    The split helper uses this for validation/test manifests so those sets can
    be pinned explicitly while still falling back to standalone manifest rows
    when the examples are not already present in the main dataset pool.
    """
    import csv as _csv

    resolved_manifest_path = manifest_path
    if not resolved_manifest_path.is_absolute():
        resolved_manifest_path = ML_ROOT / resolved_manifest_path

    manifest_abs_paths: set[str] = set()
    with open(resolved_manifest_path, newline="") as f:
        for row in _csv.DictReader(f):
            rel = row["image_path"]
            raw = Path(rel)
            img_path = raw if raw.is_absolute() else (REPO_ROOT / raw)
            abs_str = str(img_path.resolve())
            manifest_abs_paths.add(abs_str)
            manifest_abs_paths.add(img_path.name)

    manifest_examples: list[TrainingExample] = [
        example
        for example in examples
        if str(Path(example.image_path).resolve()) in manifest_abs_paths
        or Path(example.image_path).name in manifest_abs_paths
    ]

    if not manifest_examples:
        manifest_examples = _load_hard_case_examples(
            resolved_manifest_path,
            image_height=image_height,
            image_width=image_width,
        )

    manifest_names: set[str] = {Path(example.image_path).name for example in manifest_examples}
    return manifest_examples, manifest_names


def _interval_bin_count(
    value_min: float,
    value_max: float,
    bin_width: float,
) -> int:
    """Return the number of fixed-width bins that cover the sweep range."""
    span: float = value_max - value_min
    num_bins: int = int(math.ceil(span / bin_width))
    return max(num_bins, 2)


def _value_to_interval_index(
    value: float,
    *,
    value_min: float,
    value_max: float,
    bin_width: float,
) -> int:
    """Map a scalar label into a fixed-width interval class index."""
    if bin_width <= 0.0:
        raise ValueError("bin_width must be > 0.")
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    num_bins: int = _interval_bin_count(value_min, value_max, bin_width)
    raw_index: int = int(math.floor((value - value_min) / bin_width))
    return min(max(raw_index, 0), num_bins - 1)


def _ordinal_threshold_count(
    value_min: float,
    value_max: float,
    threshold_step: float,
) -> int:
    """Return the number of ordinal thresholds that cover the sweep range."""
    span: float = value_max - value_min
    num_thresholds: int = int(math.ceil(span / threshold_step))
    return max(num_thresholds, 2)


def _value_to_ordinal_threshold_vector(
    value: float,
    *,
    value_min: float,
    value_max: float,
    threshold_step: float,
) -> np.ndarray:
    """Map a scalar label into a cumulative ordinal threshold vector."""
    if threshold_step <= 0.0:
        raise ValueError("threshold_step must be > 0.")
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    num_thresholds: int = _ordinal_threshold_count(
        value_min, value_max, threshold_step
    )
    thresholds: np.ndarray = value_min + (
        np.arange(num_thresholds, dtype=np.float32) + 0.5
    ) * np.float32(threshold_step)
    return (np.float32(value) > thresholds).astype(np.float32)


def _build_interval_targets(
    examples: list[TrainingExample],
    *,
    value_min: float,
    value_max: float,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert training examples into scalar and interval-class targets."""
    scalar_targets = np.array([e.value for e in examples], dtype=np.float32)
    interval_targets = np.array(
        [
            _value_to_interval_index(
                example.value,
                value_min=value_min,
                value_max=value_max,
                bin_width=bin_width,
            )
            for example in examples
        ],
        dtype=np.int32,
    )
    return scalar_targets, interval_targets


def _load_init_model(init_model_path: Path) -> keras.Model:
    """Load a previously trained Keras model for warm-start fine-tuning."""
    if not init_model_path.exists():
        raise FileNotFoundError(f"Initial model not found: {init_model_path}")

    print(f"[TRAIN] Loading warm-start model from {init_model_path}.")
    model: keras.Model = keras.models.load_model(
        init_model_path,
        compile=False,
        custom_objects={
            "preprocess_input": keras.applications.mobilenet_v2.preprocess_input,
            "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
            "GaugeValueFromKeypoints": GaugeValueFromKeypoints,
        },
    )
    model.trainable = True
    return model


def _load_rectifier_model(rectifier_model_path: Path) -> keras.Model:
    """Load a saved rectifier model for crop generation during scalar training."""
    if not rectifier_model_path.exists():
        raise FileNotFoundError(f"Rectifier model not found: {rectifier_model_path}")

    print(f"[TRAIN] Loading rectifier model from {rectifier_model_path}.")
    model: keras.Model = keras.models.load_model(
        rectifier_model_path,
        compile=False,
        safe_mode=False,
        custom_objects={
            "preprocess_input": keras.applications.mobilenet_v2.preprocess_input,
        },
    )
    model.trainable = False
    return model


def _predict_rectifier_crop_box(
    rectifier_model: keras.Model,
    image_path: Path,
    *,
    image_size: int,
    rectifier_crop_scale: float,
) -> tuple[float, float, float, float]:
    """Predict a rectifier crop box in original-image coordinates for one image."""
    source_image: np.ndarray = load_rgb_image(image_path)
    orig_height: int = int(source_image.shape[0])
    orig_width: int = int(source_image.shape[1])
    full_frame: np.ndarray = resize_with_pad_rgb(
        source_image,
        (
            0.0,
            0.0,
            float(orig_width),
            float(orig_height),
        ),
        image_size=image_size,
    )
    rectifier_batch: np.ndarray = np.expand_dims(
        full_frame.astype(np.float32) / 255.0,
        axis=0,
    )
    rectifier_pred: np.ndarray | dict[str, np.ndarray] = rectifier_model.predict(
        rectifier_batch,
        verbose=0,
    )
    if isinstance(rectifier_pred, dict):
        rectifier_box = np.asarray(rectifier_pred["rectifier_box"]).reshape(-1)
    else:
        rectifier_box = np.asarray(rectifier_pred).reshape(-1)

    center_x: float = float(np.clip(rectifier_box[0], 0.0, 1.0))
    center_y: float = float(np.clip(rectifier_box[1], 0.0, 1.0))
    box_w: float = min(1.0, float(np.clip(rectifier_box[2], 0.05, 1.0)) * rectifier_crop_scale)
    box_h: float = min(1.0, float(np.clip(rectifier_box[3], 0.05, 1.0)) * rectifier_crop_scale)
    canvas_w: float = float(image_size)
    canvas_h: float = float(image_size)

    use_fixed_training_crop: bool = (
        (box_w < 0.25)
        or (box_h < 0.25)
        or (box_w > 0.95)
        or (box_h > 0.95)
    )
    if use_fixed_training_crop:
        # Keep the scalar head on the same stable crop used by the board's
        # baseline training pipeline when the rectifier box looks implausible.
        x_min = 0.1027 * canvas_w
        y_min = 0.2573 * canvas_h
        x_max = 0.7987 * canvas_w
        y_max = 0.8071 * canvas_h
    else:
        x_min = max(0.0, (center_x - 0.5 * box_w) * canvas_w)
        y_min = max(0.0, (center_y - 0.5 * box_h) * canvas_h)
        x_max = min(canvas_w, (center_x + 0.5 * box_w) * canvas_w)
        y_max = min(canvas_h, (center_y + 0.5 * box_h) * canvas_h)
        if x_max <= x_min + 1.0:
            x_max = min(canvas_w, x_min + 1.0)
        if y_max <= y_min + 1.0:
            y_max = min(canvas_h, y_min + 1.0)
    scale: float = min(canvas_w / float(orig_width), canvas_h / float(orig_height))
    scaled_width: float = float(orig_width) * scale
    scaled_height: float = float(orig_height) * scale
    pad_x: float = (canvas_w - scaled_width) * 0.5
    pad_y: float = (canvas_h - scaled_height) * 0.5
    x_min_orig: float = max(0.0, (x_min - pad_x) / scale)
    y_min_orig: float = max(0.0, (y_min - pad_y) / scale)
    x_max_orig: float = min(float(orig_width), (x_max - pad_x) / scale)
    y_max_orig: float = min(float(orig_height), (y_max - pad_y) / scale)
    if x_max_orig <= x_min_orig + 1.0:
        x_max_orig = min(float(orig_width), x_min_orig + 1.0)
    if y_max_orig <= y_min_orig + 1.0:
        y_max_orig = min(float(orig_height), y_min_orig + 1.0)
    return (x_min_orig, y_min_orig, x_max_orig, y_max_orig)


def _is_board_capture(image_path: str) -> bool:
    """Return True if this image came from the STM32 board camera.

    Board captures are PNG files with 4-digit names (capture_NNNN.png),
    ISO-timestamp names (capture_2026-*.png), or from today_converted/.
    Phone photos are .jpg files or PXL_* raw files and should NOT have
    rectifier boxes applied — the rectifier was trained on board framing.
    """
    p = Path(image_path)
    if p.suffix.lower() != ".png":
        return False
    name = p.stem  # e.g. "capture_0007" or "capture_2026-04-03_08-20-49"
    if "today_converted" in str(image_path):
        return True
    # 4-digit numbered captures: capture_NNNN
    if name.startswith("capture_") and name[8:].isdigit():
        return True
    # Timestamp captures: capture_YYYY-MM-DD_HH-MM-SS or capture_YYYY-MM-DD_*
    import re as _re
    if _re.match(r"capture_\d{4}-\d{2}-\d{2}", name):
        return True
    return False


def _apply_precomputed_crop_boxes(
    examples: list[TrainingExample],
    *,
    csv_path: Path,
) -> list[TrainingExample]:
    """Replace crop_box_xyxy on board-camera examples using a precomputed CSV lookup.

    Phone photos (.jpg, PXL_*) are left with their original fixed crop so the
    scalar only learns rectifier-crop framing from the same board-camera domain
    it will see at inference time.
    """
    import csv as _csv

    # Build lookup by both the raw CSV path AND its resolved absolute path so
    # we match regardless of whether training examples use absolute or relative paths.
    boxes: dict[str, tuple[float, float, float, float]] = {}
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            box = (float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"]))
            rel = row["image_path"]
            boxes[rel] = box
            abs_path = (ML_ROOT / rel).resolve()
            boxes[str(abs_path)] = box
            boxes[Path(rel).name] = box  # filename-only fallback

    # RECTIFY_ALL=1 → apply rectifier boxes to every example with a CSV match
    # (true 2-stage training: scalar always sees rectifier-crop framing).
    # Default (RECTIFY_ALL unset) → board captures only (preserves phone-photo fixed crop).
    rectify_all: bool = os.environ.get("RECTIFY_ALL", "0") == "1"

    out: list[TrainingExample] = []
    updated = 0
    skipped_not_board = 0
    skipped_no_box = 0
    for ex in examples:
        if (not rectify_all) and (not _is_board_capture(ex.image_path)):
            out.append(ex)
            skipped_not_board += 1
            continue
        key = str(Path(ex.image_path))
        stem = Path(ex.image_path).name
        if key in boxes:
            out.append(replace(ex, crop_box_xyxy=boxes[key]))
            updated += 1
        elif stem in boxes:
            out.append(replace(ex, crop_box_xyxy=boxes[stem]))
            updated += 1
        else:
            out.append(ex)
            skipped_no_box += 1

    print(
        f"[TRAIN] Precomputed crop boxes applied (rectify_all={rectify_all}): "
        f"{len(out)} examples, {updated} updated with rectifier box, "
        f"{skipped_not_board} skipped (not board capture), "
        f"{skipped_no_box} missing box (kept original crop).",
        flush=True,
    )
    return out


def _rectify_examples_for_scalar(
    examples: list[TrainingExample],
    *,
    rectifier_model_path: Path,
    image_size: int,
    rectifier_crop_scale: float,
) -> list[TrainingExample]:
    """Replace the training crop boxes with rectifier-predicted boxes."""
    print(
        "[TRAIN] step: rectify-crop-boxes",
        flush=True,
    )
    rectified_examples: list[TrainingExample] = []
    total_examples: int = len(examples)
    with tf.device("/CPU:0"):
        rectifier_model: keras.Model = _load_rectifier_model(rectifier_model_path)
        for index, example in enumerate(examples, start=1):
            image_path = Path(example.image_path)
            crop_box_xyxy = _predict_rectifier_crop_box(
                rectifier_model,
                image_path,
                image_size=image_size,
                rectifier_crop_scale=rectifier_crop_scale,
            )
            rectified_examples.append(
                replace(example, crop_box_xyxy=crop_box_xyxy)
            )
            if index % 50 == 0 or index == total_examples:
                print(
                    "[TRAIN] Rectified crops: "
                    f"{index}/{total_examples}",
                    flush=True,
                )
    return rectified_examples


def _iter_layers_recursive(model: keras.Model) -> list[keras.layers.Layer]:
    """Return all layers, including nested model layers, in a stable order."""
    layers: list[keras.layers.Layer] = []
    for layer in model.layers:
        layers.append(layer)
        if isinstance(layer, keras.Model):
            layers.extend(_iter_layers_recursive(layer))
    return layers


def _transfer_matching_weights(source_model: keras.Model, target_model: keras.Model) -> int:
    """Copy matching weights by name from one model into another model."""
    source_layers: dict[str, keras.layers.Layer] = {
        layer.name: layer for layer in _iter_layers_recursive(source_model)
    }
    transferred_layers: int = 0

    for layer in _iter_layers_recursive(target_model):
        source_layer = source_layers.get(layer.name)
        if source_layer is None:
            continue

        source_weights = source_layer.get_weights()
        target_weights = layer.get_weights()
        if not source_weights or not target_weights:
            continue
        if len(source_weights) != len(target_weights):
            continue
        if not all(
            source_weight.shape == target_weight.shape
            for source_weight, target_weight in zip(source_weights, target_weights)
        ):
            continue

        layer.set_weights(source_weights)
        transferred_layers += 1

    print(
        "[TRAIN] Warm-start weight transfer complete: "
        f"matched_layers={transferred_layers}",
    )
    return transferred_layers


def _split_examples(
    examples: list[TrainingExample],
    config: TrainConfig,
) -> DatasetSplit:
    """Split examples into train/val/test with a fixed random seed.

    If config.val_manifest and/or config.test_manifest are set, those sets are
    loaded directly from the requested CSV files and removed from the main pool
    before the remaining examples are split. This keeps the held-out domains
    deterministic while still allowing the training pool to be shuffled.
    """
    fixed_val_examples: list[TrainingExample] | None = None
    fixed_test_examples: list[TrainingExample] | None = None
    reserved_names: set[str] = set()

    if config.val_manifest is not None:
        val_manifest_path = Path(config.val_manifest)
        fixed_val_examples, val_names = _load_manifest_examples_for_split(
            examples,
            val_manifest_path,
            image_height=config.image_height,
            image_width=config.image_width,
        )
        if len(fixed_val_examples) == 0:
            raise ValueError(
                f"val_manifest loaded 0 examples from {val_manifest_path}. "
                "Check that image files exist and manifest has image_path+value columns."
            )
        reserved_names |= val_names
        print(
            f"[TRAIN] val_manifest: {len(fixed_val_examples)} fixed val examples",
            flush=True,
        )

    if config.test_manifest is not None:
        test_manifest_path = Path(config.test_manifest)
        fixed_test_examples, test_names = _load_manifest_examples_for_split(
            examples,
            test_manifest_path,
            image_height=config.image_height,
            image_width=config.image_width,
        )
        if len(fixed_test_examples) == 0:
            raise ValueError(
                f"test_manifest loaded 0 examples from {test_manifest_path}. "
                "Check that image files exist and manifest has image_path+value columns."
            )
        reserved_names |= test_names
        print(
            f"[TRAIN] test_manifest: {len(fixed_test_examples)} fixed test examples",
            flush=True,
        )

    if fixed_val_examples is not None and fixed_test_examples is not None:
        val_names = {Path(e.image_path).name for e in fixed_val_examples}
        test_names = {Path(e.image_path).name for e in fixed_test_examples}
        overlap = val_names & test_names
        if overlap:
            raise ValueError(
                "val_manifest and test_manifest overlap on image names: "
                f"{sorted(overlap)[:5]}"
            )

    remaining = [e for e in examples if Path(e.image_path).name not in reserved_names]

    if fixed_val_examples is not None and fixed_test_examples is not None:
        if len(remaining) == 0:
            raise ValueError(
                "val_manifest and test_manifest consumed all examples; "
                "check manifest paths."
            )
        print(
            f"[TRAIN] fixed manifest split: train={len(remaining)} val={len(fixed_val_examples)} test={len(fixed_test_examples)}",
            flush=True,
        )
        return DatasetSplit(
            train_examples=remaining,
            val_examples=fixed_val_examples,
            test_examples=fixed_test_examples,
        )

    if fixed_val_examples is not None:
        print(
            f"[TRAIN] val_manifest: {len(fixed_val_examples)} val examples, {len(remaining)} remaining for train/test",
            flush=True,
        )

        if len(remaining) < 2:
            raise ValueError("val_manifest consumed all examples; check manifest paths.")

        remaining_idx = np.arange(len(remaining))
        test_size = max(1, int(len(remaining) * config.test_fraction))
        train_idx, test_idx = train_test_split(
            remaining_idx,
            test_size=test_size,
            random_state=config.seed,
            shuffle=True,
        )
        print(
            f"[TRAIN] val_manifest split: train={len(train_idx)} val={len(fixed_val_examples)} test={len(test_idx)}",
            flush=True,
        )
        return DatasetSplit(
            train_examples=[remaining[i] for i in train_idx],
            val_examples=fixed_val_examples,
            test_examples=[remaining[i] for i in test_idx],
        )

    if fixed_test_examples is not None:
        print(
            f"[TRAIN] test_manifest: {len(fixed_test_examples)} test examples, {len(remaining)} remaining for train/val",
            flush=True,
        )

        if len(remaining) < 2:
            raise ValueError("test_manifest consumed all examples; check manifest paths.")

        remaining_idx = np.arange(len(remaining))
        val_size = max(1, int(len(remaining) * config.val_fraction))
        train_idx, val_idx = train_test_split(
            remaining_idx,
            test_size=val_size,
            random_state=config.seed,
            shuffle=True,
        )
        print(
            f"[TRAIN] test_manifest split: train={len(train_idx)} val={len(val_idx)} test={len(fixed_test_examples)}",
            flush=True,
        )
        return DatasetSplit(
            train_examples=[remaining[i] for i in train_idx],
            val_examples=[remaining[i] for i in val_idx],
            test_examples=fixed_test_examples,
        )

    if len(examples) < 3:
        raise ValueError("Need at least 3 examples to create train/val/test splits.")

    indices: np.ndarray = np.arange(len(examples))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_fraction,
        random_state=config.seed,
        shuffle=True,
    )

    val_relative: float = config.val_fraction / (1.0 - config.test_fraction)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        random_state=config.seed,
        shuffle=True,
    )

    return DatasetSplit(
        train_examples=[examples[i] for i in train_idx],
        val_examples=[examples[i] for i in val_idx],
        test_examples=[examples[i] for i in test_idx],
    )


def _crop_image_with_xyxy(image: tf.Tensor, crop_box_xyxy: tf.Tensor) -> tf.Tensor:
    """Safely crop image using float xyxy box and clip to valid image bounds."""
    shape: tf.Tensor = tf.shape(image)
    img_h: tf.Tensor = shape[0]
    img_w: tf.Tensor = shape[1]

    x_min_f: tf.Tensor = tf.clip_by_value(
        crop_box_xyxy[0], 0.0, tf.cast(img_w - 1, tf.float32)
    )
    y_min_f: tf.Tensor = tf.clip_by_value(
        crop_box_xyxy[1], 0.0, tf.cast(img_h - 1, tf.float32)
    )
    x_max_f: tf.Tensor = tf.clip_by_value(
        crop_box_xyxy[2], x_min_f + 1.0, tf.cast(img_w, tf.float32)
    )
    y_max_f: tf.Tensor = tf.clip_by_value(
        crop_box_xyxy[3], y_min_f + 1.0, tf.cast(img_h, tf.float32)
    )

    x_min: tf.Tensor = tf.cast(tf.math.floor(x_min_f), tf.int32)
    y_min: tf.Tensor = tf.cast(tf.math.floor(y_min_f), tf.int32)
    x_max: tf.Tensor = tf.cast(tf.math.ceil(x_max_f), tf.int32)
    y_max: tf.Tensor = tf.cast(tf.math.ceil(y_max_f), tf.int32)

    crop_w: tf.Tensor = tf.maximum(1, x_max - x_min)
    crop_h: tf.Tensor = tf.maximum(1, y_max - y_min)

    crop_w = tf.minimum(crop_w, img_w - x_min)
    crop_h = tf.minimum(crop_h, img_h - y_min)

    return tf.image.crop_to_bounding_box(image, y_min, x_min, crop_h, crop_w)


def _load_crop_and_preprocess_image(
    image_path: tf.Tensor,
    value: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Read image, crop dial ROI, resize, and normalize to [0, 1]."""
    image_bytes: tf.Tensor = tf.io.read_file(image_path)
    image: tf.Tensor = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = tf.ensure_shape(image, [None, None, 3])

    image = _crop_image_with_xyxy(image, crop_box_xyxy)
    # Preserve dial geometry (needle angle) by avoiding anisotropic warping.
    image = tf.image.resize_with_pad(image, image_height, image_width)

    image = tf.cast(image, tf.float32) / 255.0
    target: tf.Tensor = tf.cast(value, tf.float32)
    return image, target


def _load_rectifier_and_preprocess_image(
    image_path: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Read a full image and emit a padded-canvas crop-box center/size target."""
    image_bytes: tf.Tensor = tf.io.read_file(image_path)
    image: tf.Tensor = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = tf.ensure_shape(image, [None, None, 3])

    orig_h: tf.Tensor = tf.cast(tf.shape(image)[0], tf.float32)
    orig_w: tf.Tensor = tf.cast(tf.shape(image)[1], tf.float32)

    # Keep the full frame intact so the rectifier learns the crop on the same
    # padded 224x224 canvas used at inference time.
    image = tf.image.resize_with_pad(image, image_height, image_width)
    image = tf.cast(image, tf.float32) / 255.0

    x_min: tf.Tensor = tf.cast(crop_box_xyxy[0], tf.float32)
    y_min: tf.Tensor = tf.cast(crop_box_xyxy[1], tf.float32)
    x_max: tf.Tensor = tf.cast(crop_box_xyxy[2], tf.float32)
    y_max: tf.Tensor = tf.cast(crop_box_xyxy[3], tf.float32)

    scale: tf.Tensor = tf.minimum(
        tf.cast(image_height, tf.float32) / tf.maximum(orig_h, 1.0),
        tf.cast(image_width, tf.float32) / tf.maximum(orig_w, 1.0),
    )
    scaled_w: tf.Tensor = orig_w * scale
    scaled_h: tf.Tensor = orig_h * scale
    pad_x: tf.Tensor = 0.5 * (tf.cast(image_width, tf.float32) - scaled_w)
    pad_y: tf.Tensor = 0.5 * (tf.cast(image_height, tf.float32) - scaled_h)

    x_min_canvas: tf.Tensor = x_min * scale + pad_x
    y_min_canvas: tf.Tensor = y_min * scale + pad_y
    x_max_canvas: tf.Tensor = x_max * scale + pad_x
    y_max_canvas: tf.Tensor = y_max * scale + pad_y

    center_x: tf.Tensor = tf.clip_by_value(
        ((x_min_canvas + x_max_canvas) * 0.5) / tf.maximum(
            tf.cast(image_width, tf.float32), 1.0
        ),
        0.0,
        1.0,
    )
    center_y: tf.Tensor = tf.clip_by_value(
        ((y_min_canvas + y_max_canvas) * 0.5) / tf.maximum(
            tf.cast(image_height, tf.float32), 1.0
        ),
        0.0,
        1.0,
    )
    box_w: tf.Tensor = tf.clip_by_value(
        (x_max_canvas - x_min_canvas) / tf.maximum(
            tf.cast(image_width, tf.float32), 1.0
        ),
        0.0,
        1.0,
    )
    box_h: tf.Tensor = tf.clip_by_value(
        (y_max_canvas - y_min_canvas) / tf.maximum(
            tf.cast(image_height, tf.float32), 1.0
        ),
        0.0,
        1.0,
    )
    target: tf.Tensor = tf.stack([center_x, center_y, box_w, box_h], axis=-1)
    return image, target


def _load_crop_with_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Load/crop one image and attach the requested sample weight."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    return image, target, weight


def _load_rectifier_with_weight(
    image_path: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Load a full-frame rectifier example and attach its sample weight."""
    image, target = _load_rectifier_and_preprocess_image(
        image_path,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    return image, target, weight


def _load_crop_with_obb_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach oriented-box parameters."""
    image, _scalar_target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    obb_target = tf.cast(obb_params, tf.float32)
    return image, {"obb_params": obb_target}


def _load_crop_with_obb_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach OBB parameters plus a sample weight."""
    image, target = _load_crop_with_obb_target(
        image_path,
        value,
        obb_params,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    return image, target, {"obb_params": tf.cast(weight, tf.float32)}


def _load_crop_with_interval_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    interval_index: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach scalar, interval, and weight targets."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    interval_target = tf.cast(interval_index, tf.int32)
    targets = {
        "gauge_value": target,
        "interval_logits": interval_target,
    }
    sample_weights = {
        "gauge_value": weight,
        "interval_logits": weight,
    }
    return image, targets, sample_weights


def _load_crop_with_interval_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    interval_index: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach scalar plus interval-class targets."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    interval_target = tf.cast(interval_index, tf.int32)
    targets = {
        "gauge_value": target,
        "interval_logits": interval_target,
    }
    return image, targets


def _load_crop_with_ordinal_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    ordinal_thresholds: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach scalar plus ordinal-threshold targets."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    ordinal_target = tf.cast(ordinal_thresholds, tf.float32)
    targets = {
        "gauge_value": target,
        "ordinal_logits": ordinal_target,
    }
    return image, targets


def _load_crop_with_ordinal_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    ordinal_thresholds: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach ordinal targets plus sample weights."""
    image, targets = _load_crop_with_ordinal_target(
        image_path,
        value,
        ordinal_thresholds,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "ordinal_logits": weight,
    }
    return image, targets, sample_weights


def _load_crop_with_fraction_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    fraction: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach scalar plus sweep-fraction targets."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    fraction_target = tf.cast(fraction, tf.float32)
    targets = {
        "gauge_value": target,
        "sweep_fraction": fraction_target,
    }
    return image, targets


def _load_crop_with_fraction_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    fraction: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach sweep-fraction targets plus sample weights."""
    image, targets = _load_crop_with_fraction_target(
        image_path,
        value,
        fraction,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "sweep_fraction": weight,
    }
    return image, targets, sample_weights


def _load_crop_with_keypoint_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach scalar plus keypoint-heatmap targets."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    keypoint_target = tf.cast(heatmaps, tf.float32)
    targets = {
        "gauge_value": target,
        "keypoint_heatmaps": keypoint_target,
    }
    return image, targets


def _load_crop_with_keypoint_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    heatmap_weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach keypoint targets plus sample weights."""
    image, targets = _load_crop_with_keypoint_target(
        image_path,
        value,
        heatmaps,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "keypoint_heatmaps": heatmap_weight,
    }
    return image, targets, sample_weights


def _load_crop_with_geometry_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach geometry targets for detector training."""
    image, value_target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    keypoint_target = tf.cast(heatmaps, tf.float32)
    coord_target = tf.cast(coords, tf.float32)
    targets = {
        "gauge_value": value_target,
        "keypoint_heatmaps": keypoint_target,
        "keypoint_coords": coord_target,
    }
    return image, targets


def _load_crop_with_geometry_uncertainty_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach geometry plus uncertainty targets."""
    image, targets = _load_crop_with_geometry_target(
        image_path,
        value,
        heatmaps,
        coords,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    targets["gauge_value_lower"] = targets["gauge_value"]
    targets["gauge_value_upper"] = targets["gauge_value"]
    return image, targets


def _load_crop_with_geometry_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    heatmap_weight: tf.Tensor,
    coord_weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach geometry targets plus sample weights."""
    image, targets = _load_crop_with_geometry_target(
        image_path,
        value,
        heatmaps,
        coords,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "keypoint_heatmaps": heatmap_weight,
        "keypoint_coords": coord_weight,
    }
    return image, targets, sample_weights


def _load_crop_with_geometry_uncertainty_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    heatmap_weight: tf.Tensor,
    coord_weight: tf.Tensor,
    uncertainty_weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach geometry uncertainty targets plus weights."""
    image, targets = _load_crop_with_geometry_uncertainty_target(
        image_path,
        value,
        heatmaps,
        coords,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "keypoint_heatmaps": heatmap_weight,
        "keypoint_coords": coord_weight,
        "gauge_value_lower": uncertainty_weight,
        "gauge_value_upper": uncertainty_weight,
    }
    return image, targets, sample_weights


def _augment_glare_blobs(image: tf.Tensor) -> tf.Tensor:
    """Stamp 1–3 bright glare blobs via resized gaussian noise. Simulates specular reflections on gauge glass."""
    # Generate blobs at a fixed small resolution then resize up — avoids dynamic meshgrid entirely.
    BLOB_RES = 32
    mask = tf.zeros([BLOB_RES, BLOB_RES, 1], dtype=tf.float32)
    for _ in range(3):
        active = tf.cast(tf.random.uniform([]) < 0.5, tf.float32)
        cx = tf.random.uniform([], 2, BLOB_RES - 2, dtype=tf.float32)
        cy = tf.random.uniform([], 2, BLOB_RES - 2, dtype=tf.float32)
        # Draw a single bright pixel at (cy, cx) with strength, then blur via resize
        brightness = tf.random.uniform([], 0.5, 1.0) * active
        # Use a small dense gaussian kernel: encode as a 1-pixel impulse scaled by brightness
        # We approximate the blob by creating a [BLOB_RES, BLOB_RES] tensor with one hot pixel
        # and relying on bilinear resize to spread it.
        cy_i = tf.cast(tf.round(cy), tf.int32)
        cx_i = tf.cast(tf.round(cx), tf.int32)
        # Scatter a bright spot
        idx = tf.stack([cy_i, cx_i, 0])
        spot = tf.tensor_scatter_nd_update(
            tf.zeros([BLOB_RES, BLOB_RES, 1], dtype=tf.float32),
            [idx],
            [brightness],
        )
        mask = mask + spot

    # Resize to image size with bicubic to spread the spots into soft blobs
    image_shape = tf.shape(image)
    mask = tf.image.resize(mask, [image_shape[0], image_shape[1]], method="bicubic")
    mask = tf.clip_by_value(mask, 0.0, 1.0)  # [H, W, 1] broadcast across channels
    image = image + mask * (1.0 - image)
    return tf.clip_by_value(image, 0.0, 1.0)


def _augment_image(image: tf.Tensor) -> tf.Tensor:
    """Apply photometric augmentation that preserves gauge geometry."""
    image_shape: tf.Tensor = tf.shape(image)
    image_h: tf.Tensor = image_shape[0]
    image_w: tf.Tensor = image_shape[1]

    # Randomly crop-and-resize to inject slight scale/translation variation.
    scale: tf.Tensor = tf.random.uniform([], minval=0.92, maxval=1.0, dtype=tf.float32)
    crop_h: tf.Tensor = tf.maximum(
        2, tf.cast(tf.cast(image_h, tf.float32) * scale, dtype=tf.int32)
    )
    crop_w: tf.Tensor = tf.maximum(
        2, tf.cast(tf.cast(image_w, tf.float32) * scale, dtype=tf.int32)
    )
    image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
    image = tf.image.resize(image, [image_h, image_w])

    # AUG_HEAVY=1 → stronger photometric augmentation for board-domain robustness.
    aug_heavy: bool = os.environ.get("AUG_HEAVY", "0") == "1"
    if aug_heavy:
        image = tf.image.random_brightness(image, max_delta=0.25)
        image = tf.image.random_contrast(image, lower=0.65, upper=1.35)
        image = tf.image.random_saturation(image, lower=0.80, upper=1.20)
        image = tf.image.random_hue(image, max_delta=0.04)
    else:
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    # Clip to valid range before any pow/exp operations to prevent NaN from negative bases.
    image = tf.clip_by_value(image, 0.0, 1.0)

    if aug_heavy:
        # Wider gamma sweep at higher rate: covers under- AND over-exposed captures.
        gamma_dark: tf.Tensor = tf.random.uniform([], minval=1.0, maxval=2.5, dtype=tf.float32)
        apply_dark: tf.Tensor = tf.random.uniform([]) < 0.30
        image = tf.where(apply_dark, tf.pow(image, gamma_dark), image)
        gamma_bright: tf.Tensor = tf.random.uniform([], minval=0.5, maxval=1.0, dtype=tf.float32)
        apply_bright: tf.Tensor = tf.random.uniform([]) < 0.15
        image = tf.where(apply_bright, tf.pow(image, gamma_bright), image)
    else:
        gamma: tf.Tensor = tf.random.uniform([], minval=1.0, maxval=2.0, dtype=tf.float32)
        apply_dark: tf.Tensor = tf.random.uniform([]) < 0.10
        image = tf.where(apply_dark, tf.pow(image, gamma), image)

    # Simulate specular glare patches on gauge glass (20% chance).
    apply_glare: tf.Tensor = tf.random.uniform([]) < 0.20
    image = tf.cond(
        apply_glare,
        lambda: _augment_glare_blobs(image),
        lambda: image,
    )

    noise: tf.Tensor = tf.random.normal(
        shape=tf.shape(image),
        mean=0.0,
        stddev=0.015,
        dtype=tf.float32,
    )
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _augment_rectifier_image_and_box(
    image: tf.Tensor,
    box: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Geometric + photometric augmentation for the rectifier training path.

    Randomly zooms into a sub-region of the 224x224 padded canvas so the model
    sees dial sizes ranging from ~30% to 100% of the frame — covering both the
    far-away phone-photo distribution and close-up board-camera framing.

    The target box (cx, cy, w, h) in [0,1] canvas coords is recomputed to stay
    consistent with the zoomed view.  If the zoom window clips the dial box we
    clamp it so the target always describes the visible portion of the dial.
    """
    image_shape: tf.Tensor = tf.shape(image)
    canvas: tf.Tensor = tf.cast(image_shape[0], tf.float32)  # square 224

    # --- random zoom window in canvas pixels ---
    # zoom_scale in [0.40, 1.0]: 0.40 → 2.5x zoom-in (close-up), 1.0 → full frame
    zoom_scale: tf.Tensor = tf.random.uniform([], minval=0.40, maxval=1.0)
    win_size: tf.Tensor = canvas * zoom_scale

    max_offset: tf.Tensor = tf.maximum(canvas - win_size, 0.0)
    off_x: tf.Tensor = tf.random.uniform([], minval=0.0, maxval=1.0) * max_offset
    off_y: tf.Tensor = tf.random.uniform([], minval=0.0, maxval=1.0) * max_offset

    # crop then resize back to canvas
    off_y_i = tf.cast(tf.math.floor(off_y), tf.int32)
    off_x_i = tf.cast(tf.math.floor(off_x), tf.int32)
    win_i = tf.cast(tf.math.ceil(win_size), tf.int32)
    win_i = tf.maximum(win_i, 2)
    win_i = tf.minimum(win_i, image_shape[0] - off_y_i)
    win_i_w = tf.minimum(win_i, image_shape[1] - off_x_i)
    image = tf.image.crop_to_bounding_box(image, off_y_i, off_x_i, win_i, win_i_w)
    image = tf.image.resize(image, [image_shape[0], image_shape[1]])

    # --- recompute box target in zoomed canvas coords ---
    # input box: [cx, cy, w, h] all in [0,1] relative to original 224x224
    cx = box[0] * canvas
    cy = box[1] * canvas
    bw = box[2] * canvas
    bh = box[3] * canvas

    x_min = cx - bw * 0.5
    y_min = cy - bh * 0.5
    x_max = cx + bw * 0.5
    y_max = cy + bh * 0.5

    # map original canvas coords → zoomed canvas coords
    x_min_z = (x_min - off_x) / win_size * canvas
    y_min_z = (y_min - off_y) / win_size * canvas
    x_max_z = (x_max - off_x) / win_size * canvas
    y_max_z = (y_max - off_y) / win_size * canvas

    # clamp to [0, canvas]
    x_min_z = tf.clip_by_value(x_min_z, 0.0, canvas)
    y_min_z = tf.clip_by_value(y_min_z, 0.0, canvas)
    x_max_z = tf.clip_by_value(x_max_z, 0.0, canvas)
    y_max_z = tf.clip_by_value(y_max_z, 0.0, canvas)

    new_cx = tf.clip_by_value(((x_min_z + x_max_z) * 0.5) / canvas, 0.0, 1.0)
    new_cy = tf.clip_by_value(((y_min_z + y_max_z) * 0.5) / canvas, 0.0, 1.0)
    new_w = tf.clip_by_value((x_max_z - x_min_z) / canvas, 0.0, 1.0)
    new_h = tf.clip_by_value((y_max_z - y_min_z) / canvas, 0.0, 1.0)
    new_box = tf.stack([new_cx, new_cy, new_w, new_h])

    # photometric augmentation (same as scalar path)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    noise: tf.Tensor = tf.random.normal(
        shape=tf.shape(image), mean=0.0, stddev=0.01, dtype=tf.float32
    )
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, new_box


def _build_tf_dataset(
    examples: list[TrainingExample],
    config: TrainConfig,
    *,
    training: bool,
    target_kind: Literal[
        "value",
        "needle_unit_xy",
        "sweep_fraction",
        "interval_value",
        "ordinal_thresholds",
        "keypoint_heatmaps",
        "geometry",
        "geometry_uncertainty",
        "rectifier_box",
        "obb",
    ] = "value",
    value_range: tuple[float, float] | None = None,
) -> tf.data.Dataset:
    """Create a tf.data pipeline for efficient training/eval input."""
    paths: np.ndarray = np.array([e.image_path for e in examples], dtype=str)
    if target_kind == "value":
        targets: np.ndarray = np.array([e.value for e in examples], dtype=np.float32)
    elif target_kind == "needle_unit_xy":
        targets = np.array([e.needle_unit_xy for e in examples], dtype=np.float32)
    elif target_kind == "sweep_fraction":
        targets = np.array([e.value for e in examples], dtype=np.float32)
    elif target_kind == "interval_value":
        targets = np.array([e.value for e in examples], dtype=np.float32)
    elif target_kind == "ordinal_thresholds":
        targets = np.array([e.value for e in examples], dtype=np.float32)
    elif target_kind == "keypoint_heatmaps":
        targets = np.array([e.value for e in examples], dtype=np.float32)
    elif target_kind == "geometry":
        targets = np.array([e.value for e in examples], dtype=np.float32)
    elif target_kind == "geometry_uncertainty":
        targets = np.array([e.value for e in examples], dtype=np.float32)
    elif target_kind == "rectifier_box":
        targets = np.array([e.crop_box_xyxy for e in examples], dtype=np.float32)
    elif target_kind == "obb":
        values = np.array([e.value for e in examples], dtype=np.float32)
        targets = np.array(
            [
                e.obb_params
                if e.obb_params is not None
                else np.zeros((6,), dtype=np.float32)
                for e in examples
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unsupported target_kind '{target_kind}'.")
    boxes: np.ndarray = np.array([e.crop_box_xyxy for e in examples], dtype=np.float32)

    if training and target_kind == "value":
        weights: np.ndarray = _compute_edge_weights(
            examples, config.edge_focus_strength
        )
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
            (paths, targets, boxes, weights)
        )
        dataset = dataset.shuffle(
            buffer_size=max(len(examples), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.map(
            lambda p, y, b, w: _load_crop_with_weight(
                p,
                y,
                b,
                config.image_height,
                config.image_width,
                w,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if config.augment_training:
            dataset = dataset.map(
                lambda img, y, w: (_augment_image(img), y, w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        if config.mixup_alpha > 0.0:
            dataset = dataset.batch(config.batch_size, drop_remainder=False)
            dataset = dataset.map(
                lambda img, y, w: _mixup_value_batch(img, y, w, config.mixup_alpha),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    elif training and target_kind == "interval_value":
        if config.mixup_alpha > 0.0:
            raise ValueError("MixUp is not supported for interval-value targets yet.")
        if value_range is None:
            min_value = min(example.value for example in examples)
            max_value = max(example.value for example in examples)
        else:
            min_value, max_value = value_range
        weights = _compute_edge_weights(examples, config.edge_focus_strength)
        interval_targets = np.array(
            [
                _value_to_interval_index(
                    example.value,
                    value_min=min_value,
                    value_max=max_value,
                    bin_width=config.interval_bin_width,
                )
                for example in examples
            ],
            dtype=np.int32,
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, targets, interval_targets, boxes, weights)
        )
        dataset = dataset.shuffle(
            buffer_size=max(len(examples), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.map(
            lambda p, y, c, b, w: _load_crop_with_interval_weight(
                p,
                y,
                c,
                b,
                config.image_height,
                config.image_width,
                w,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if config.augment_training:
            dataset = dataset.map(
                lambda img, y, w: (_augment_image(img), y, w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    elif training and target_kind == "sweep_fraction":
        if config.mixup_alpha > 0.0:
            raise ValueError(
                "MixUp is not supported for sweep-fraction targets yet."
            )
        weights = _compute_edge_weights(examples, config.edge_focus_strength)
        fraction_targets = np.array([e.value_norm for e in examples], dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, targets, fraction_targets, boxes, weights)
        )
        dataset = dataset.shuffle(
            buffer_size=max(len(examples), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.map(
            lambda p, y, f, b, w: _load_crop_with_fraction_weight(
                p,
                y,
                f,
                b,
                config.image_height,
                config.image_width,
                w,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if config.augment_training:
            dataset = dataset.map(
                lambda img, y, w: (_augment_image(img), y, w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    elif training and target_kind == "keypoint_heatmaps":
        if config.mixup_alpha > 0.0:
            raise ValueError(
                "MixUp is not supported for keypoint-heatmap targets yet."
            )
        weights = _compute_edge_weights(examples, config.edge_focus_strength)
        heatmap_weights = np.array(
            [
                np.full(
                    (config.keypoint_heatmap_size, config.keypoint_heatmap_size),
                    fill_value=(
                        weight if example.keypoint_heatmaps is not None else 0.0
                    ),
                    dtype=np.float32,
                )
                for example, weight in zip(examples, weights, strict=True)
            ],
            dtype=np.float32,
        )
        heatmaps = np.array(
            [
                example.keypoint_heatmaps
                if example.keypoint_heatmaps is not None
                else np.zeros(
                    (
                        config.keypoint_heatmap_size,
                        config.keypoint_heatmap_size,
                        2,
                    ),
                    dtype=np.float32,
                )
                for example in examples
            ],
            dtype=np.float32,
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, targets, heatmaps, boxes, weights, heatmap_weights)
        )
        dataset = dataset.shuffle(
            buffer_size=max(len(examples), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.map(
            lambda p, y, h, b, w, hw: _load_crop_with_keypoint_weight(
                p,
                y,
                h,
                b,
                config.image_height,
                config.image_width,
                w,
                hw,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if config.augment_training:
            dataset = dataset.map(
                lambda img, y, w: (_augment_image(img), y, w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    elif training and target_kind == "geometry":
        if config.mixup_alpha > 0.0:
            raise ValueError("MixUp is not supported for geometry targets yet.")
        weights = _compute_edge_weights(examples, config.edge_focus_strength)
        heatmap_weights = np.array(
            [
                np.full(
                    (config.keypoint_heatmap_size, config.keypoint_heatmap_size),
                    fill_value=(
                        weight if example.keypoint_heatmaps is not None else 0.0
                    ),
                    dtype=np.float32,
                )
                for example, weight in zip(examples, weights, strict=True)
            ],
            dtype=np.float32,
        )
        heatmaps = np.array(
            [
                example.keypoint_heatmaps
                if example.keypoint_heatmaps is not None
                else np.zeros(
                    (
                        config.keypoint_heatmap_size,
                        config.keypoint_heatmap_size,
                        2,
                    ),
                    dtype=np.float32,
                )
                for example in examples
            ],
            dtype=np.float32,
        )
        coords = np.array(
            [
                example.keypoint_coords
                if example.keypoint_coords is not None
                else np.zeros((2, 2), dtype=np.float32)
                for example in examples
            ],
            dtype=np.float32,
        )
        coord_weights = np.array(
            [
                np.full(
                    (2,),
                    fill_value=weight if example.keypoint_coords is not None else 0.0,
                    dtype=np.float32,
                )
                for example, weight in zip(examples, weights, strict=True)
            ],
            dtype=np.float32,
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, targets, heatmaps, coords, boxes, weights, heatmap_weights, coord_weights)
        )
        dataset = dataset.shuffle(
            buffer_size=max(len(examples), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.map(
            lambda p, y, h, c, b, w, hw, cw: _load_crop_with_geometry_weight(
                p,
                y,
                h,
                c,
                b,
                config.image_height,
                config.image_width,
                w,
                hw,
                cw,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if config.augment_training:
            dataset = dataset.map(
                lambda img, y, w: (_augment_image(img), y, w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    elif training and target_kind == "geometry_uncertainty":
        if config.mixup_alpha > 0.0:
            raise ValueError(
                "MixUp is not supported for geometry-uncertainty targets yet."
            )
        weights = _compute_edge_weights(examples, config.edge_focus_strength)
        heatmap_weights = np.array(
            [
                np.full(
                    (config.keypoint_heatmap_size, config.keypoint_heatmap_size),
                    fill_value=(
                        weight if example.keypoint_heatmaps is not None else 0.0
                    ),
                    dtype=np.float32,
                )
                for example, weight in zip(examples, weights, strict=True)
            ],
            dtype=np.float32,
        )
        heatmaps = np.array(
            [
                example.keypoint_heatmaps
                if example.keypoint_heatmaps is not None
                else np.zeros(
                    (
                        config.keypoint_heatmap_size,
                        config.keypoint_heatmap_size,
                        2,
                    ),
                    dtype=np.float32,
                )
                for example in examples
            ],
            dtype=np.float32,
        )
        coords = np.array(
            [
                example.keypoint_coords
                if example.keypoint_coords is not None
                else np.zeros((2, 2), dtype=np.float32)
                for example in examples
            ],
            dtype=np.float32,
        )
        coord_weights = np.array(
            [
                np.full(
                    (2,),
                    fill_value=weight if example.keypoint_coords is not None else 0.0,
                    dtype=np.float32,
                )
                for example, weight in zip(examples, weights, strict=True)
            ],
            dtype=np.float32,
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, targets, heatmaps, coords, boxes, weights, heatmap_weights, coord_weights)
        )
        dataset = dataset.shuffle(
            buffer_size=max(len(examples), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.map(
            lambda p, y, h, c, b, w, hw, cw: _load_crop_with_geometry_uncertainty_weight(
                p,
                y,
                h,
                c,
                b,
                config.image_height,
                config.image_width,
                w,
                hw,
                cw,
                w,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if config.augment_training:
            dataset = dataset.map(
                lambda img, y, w: (_augment_image(img), y, w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    elif training and target_kind == "rectifier_box":
        if config.mixup_alpha > 0.0:
            raise ValueError(
                "MixUp is not supported for rectifier-box targets yet."
            )
        weights = _compute_edge_weights(examples, config.edge_focus_strength)
        dataset = tf.data.Dataset.from_tensor_slices((paths, boxes, weights))
        dataset = dataset.shuffle(
            buffer_size=max(len(examples), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.map(
            lambda p, b, w: _load_rectifier_with_weight(
                p,
                b,
                config.image_height,
                config.image_width,
                w,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if config.augment_training:
            dataset = dataset.map(
                lambda img, y, w: (*_augment_rectifier_image_and_box(img, y), w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    elif training and target_kind == "obb":
        if config.mixup_alpha > 0.0:
            raise ValueError("MixUp is not supported for OBB targets yet.")
        weights = _compute_edge_weights(examples, config.edge_focus_strength)
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, values, targets, boxes, weights)
        )
        dataset = dataset.shuffle(
            buffer_size=max(len(examples), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.map(
            lambda p, v, y, b, w: _load_crop_with_obb_weight(
                p,
                v,
                y,
                b,
                config.image_height,
                config.image_width,
                w,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if config.augment_training:
            dataset = dataset.map(
                lambda img, y, w: (_augment_image(img), y, w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    elif training and target_kind == "ordinal_thresholds":
        if config.mixup_alpha > 0.0:
            raise ValueError(
                "MixUp is not supported for ordinal-threshold targets yet."
            )
        if value_range is None:
            min_value = min(example.value for example in examples)
            max_value = max(example.value for example in examples)
        else:
            min_value, max_value = value_range
        weights = _compute_edge_weights(examples, config.edge_focus_strength)
        ordinal_targets = np.array(
            [
                _value_to_ordinal_threshold_vector(
                    example.value,
                    value_min=min_value,
                    value_max=max_value,
                    threshold_step=config.ordinal_threshold_step,
                )
                for example in examples
            ],
            dtype=np.float32,
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, targets, ordinal_targets, boxes, weights)
        )
        dataset = dataset.shuffle(
            buffer_size=max(len(examples), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.map(
            lambda p, y, o, b, w: _load_crop_with_ordinal_weight(
                p,
                y,
                o,
                b,
                config.image_height,
                config.image_width,
                w,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if config.augment_training:
            dataset = dataset.map(
                lambda img, y, w: (_augment_image(img), y, w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    else:
        if target_kind == "interval_value":
            if value_range is None:
                min_value = min(example.value for example in examples)
                max_value = max(example.value for example in examples)
            else:
                min_value, max_value = value_range
            interval_targets = np.array(
                [
                    _value_to_interval_index(
                        example.value,
                        value_min=min_value,
                        value_max=max_value,
                        bin_width=config.interval_bin_width,
                    )
                    for example in examples
                ],
                dtype=np.int32,
            )
            dataset = tf.data.Dataset.from_tensor_slices(
                (paths, targets, interval_targets, boxes)
            )
            dataset = dataset.map(
                lambda p, y, c, b: _load_crop_with_interval_target(
                    p,
                    y,
                    c,
                    b,
                    config.image_height,
                    config.image_width,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif target_kind == "sweep_fraction":
            fraction_targets = np.array(
                [example.value_norm for example in examples],
                dtype=np.float32,
            )
            dataset = tf.data.Dataset.from_tensor_slices(
                (paths, targets, fraction_targets, boxes)
            )
            dataset = dataset.map(
                lambda p, y, f, b: _load_crop_with_fraction_target(
                    p,
                    y,
                    f,
                    b,
                    config.image_height,
                    config.image_width,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif target_kind == "keypoint_heatmaps":
            heatmaps = np.array(
                [
                    example.keypoint_heatmaps
                    if example.keypoint_heatmaps is not None
                    else np.zeros(
                        (
                            config.keypoint_heatmap_size,
                            config.keypoint_heatmap_size,
                            2,
                        ),
                        dtype=np.float32,
                    )
                    for example in examples
                ],
                dtype=np.float32,
            )
            dataset = tf.data.Dataset.from_tensor_slices(
                (paths, targets, heatmaps, boxes)
            )
            dataset = dataset.map(
                lambda p, y, h, b: _load_crop_with_keypoint_target(
                    p,
                    y,
                    h,
                    b,
                    config.image_height,
                    config.image_width,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif target_kind == "geometry":
            heatmaps = np.array(
                [
                    example.keypoint_heatmaps
                    if example.keypoint_heatmaps is not None
                    else np.zeros(
                        (
                            config.keypoint_heatmap_size,
                            config.keypoint_heatmap_size,
                            2,
                        ),
                        dtype=np.float32,
                    )
                    for example in examples
                ],
                dtype=np.float32,
            )
            coords = np.array(
                [
                    example.keypoint_coords
                    if example.keypoint_coords is not None
                    else np.zeros((2, 2), dtype=np.float32)
                    for example in examples
                ],
                dtype=np.float32,
            )
            dataset = tf.data.Dataset.from_tensor_slices(
                (paths, targets, heatmaps, coords, boxes)
            )
            dataset = dataset.map(
                lambda p, y, h, c, b: _load_crop_with_geometry_target(
                    p,
                    y,
                    h,
                    c,
                    b,
                    config.image_height,
                    config.image_width,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif target_kind == "geometry_uncertainty":
            heatmaps = np.array(
                [
                    example.keypoint_heatmaps
                    if example.keypoint_heatmaps is not None
                    else np.zeros(
                        (
                            config.keypoint_heatmap_size,
                            config.keypoint_heatmap_size,
                            2,
                        ),
                        dtype=np.float32,
                    )
                    for example in examples
                ],
                dtype=np.float32,
            )
            coords = np.array(
                [
                    example.keypoint_coords
                    if example.keypoint_coords is not None
                    else np.zeros((2, 2), dtype=np.float32)
                    for example in examples
                ],
                dtype=np.float32,
            )
            dataset = tf.data.Dataset.from_tensor_slices(
                (paths, targets, heatmaps, coords, boxes)
            )
            dataset = dataset.map(
                lambda p, y, h, c, b: _load_crop_with_geometry_uncertainty_target(
                    p,
                    y,
                    h,
                    c,
                    b,
                    config.image_height,
                    config.image_width,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif target_kind == "rectifier_box":
            box_targets = np.array(
                [example.crop_box_xyxy for example in examples],
                dtype=np.float32,
            )
            dataset = tf.data.Dataset.from_tensor_slices((paths, box_targets))
            dataset = dataset.map(
                lambda p, b: _load_rectifier_and_preprocess_image(
                    p,
                    b,
                    config.image_height,
                    config.image_width,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif target_kind == "obb":
            dataset = tf.data.Dataset.from_tensor_slices((paths, values, targets, boxes))
            dataset = dataset.map(
                lambda p, v, y, b: _load_crop_with_obb_target(
                    p,
                    v,
                    y,
                    b,
                    config.image_height,
                    config.image_width,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif target_kind == "ordinal_thresholds":
            if value_range is None:
                min_value = min(example.value for example in examples)
                max_value = max(example.value for example in examples)
            else:
                min_value, max_value = value_range
            ordinal_targets = np.array(
                [
                    _value_to_ordinal_threshold_vector(
                        example.value,
                        value_min=min_value,
                        value_max=max_value,
                        threshold_step=config.ordinal_threshold_step,
                    )
                    for example in examples
                ],
                dtype=np.float32,
            )
            dataset = tf.data.Dataset.from_tensor_slices(
                (paths, targets, ordinal_targets, boxes)
            )
            dataset = dataset.map(
                lambda p, y, o, b: _load_crop_with_ordinal_target(
                    p,
                    y,
                    o,
                    b,
                    config.image_height,
                    config.image_width,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices((paths, targets, boxes))
            dataset = dataset.map(
                lambda p, y, b: _load_crop_and_preprocess_image(
                    p,
                    y,
                    b,
                    config.image_height,
                    config.image_width,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    if not (training and target_kind == "value" and config.mixup_alpha > 0.0):
        dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _edge_weight(value_norm: float, strength: float) -> float:
    """Return a weight that concentrates loss on the dial edges."""
    if strength <= 0.0:
        return 1.0
    distance = abs(value_norm - 0.5) * 2.0
    return 1.0 + strength * distance


def _compute_edge_weights(
    examples: list[TrainingExample],
    strength: float,
) -> np.ndarray:
    """Map normalized scalar labels into sample weights that emphasize extremes."""
    if strength <= 0.0:
        return np.ones(len(examples), dtype=np.float32)
    weights = np.array(
        [_edge_weight(example.value_norm, strength) for example in examples],
        dtype=np.float32,
    )
    return weights


def _sample_mixup_lambda(alpha: float) -> tf.Tensor:
    """Sample a MixUp interpolation coefficient from a symmetric Beta law."""
    if alpha <= 0.0:
        return tf.constant(1.0, dtype=tf.float32)
    left: tf.Tensor = tf.random.gamma([], alpha=alpha, beta=1.0, dtype=tf.float32)
    right: tf.Tensor = tf.random.gamma([], alpha=alpha, beta=1.0, dtype=tf.float32)
    return tf.cast(left / (left + right + 1e-7), tf.float32)


def _mixup_value_batch(
    images: tf.Tensor,
    targets: tf.Tensor,
    weights: tf.Tensor,
    alpha: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Apply MixUp to a batch of scalar-regression examples."""
    batch_size: tf.Tensor = tf.shape(images)[0]
    perm: tf.Tensor = tf.random.shuffle(tf.range(batch_size))
    mixed_images: tf.Tensor = tf.gather(images, perm)
    mixed_targets: tf.Tensor = tf.gather(targets, perm)
    mixed_weights: tf.Tensor = tf.gather(weights, perm)

    lam: tf.Tensor = _sample_mixup_lambda(alpha)
    lam_img: tf.Tensor = tf.reshape(lam, [1, 1, 1, 1])
    lam_vec: tf.Tensor = tf.reshape(lam, [1])

    images = lam_img * images + (1.0 - lam_img) * mixed_images
    targets = lam_vec * targets + (1.0 - lam_vec) * mixed_targets
    weights = lam_vec * weights + (1.0 - lam_vec) * mixed_weights
    return images, targets, weights


def _compute_mean_baseline_mae(
    train_examples: list[TrainingExample],
    test_examples: list[TrainingExample],
) -> float:
    """Compute MAE of a trivial baseline that predicts train-mean value."""
    train_values: np.ndarray = np.array([e.value for e in train_examples], dtype=np.float32)
    test_values: np.ndarray = np.array([e.value for e in test_examples], dtype=np.float32)

    mean_pred: float = float(np.mean(train_values))
    baseline_mae: float = float(np.mean(np.abs(test_values - mean_pred)))
    return baseline_mae


def _make_scalar_regression_loss(
    *,
    monotonic_pair_strength: float = 0.0,
    monotonic_pair_margin: float = 0.0,
    interpolation_pair_strength: float = 0.0,
    interpolation_pair_scale: float = DEFAULT_INTERPOLATION_PAIR_SCALE,
):
    """Create the scalar regression loss used by the training heads."""
    def monotonic_pair_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Penalize local ordering violations inside a batch."""
        y_true_flat: tf.Tensor = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_flat: tf.Tensor = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        order: tf.Tensor = tf.argsort(y_true_flat, stable=True)
        sorted_pred: tf.Tensor = tf.gather(y_pred_flat, order)
        count: tf.Tensor = tf.shape(sorted_pred)[0]

        def _compute() -> tf.Tensor:
            diffs: tf.Tensor = sorted_pred[1:] - sorted_pred[:-1]
            violations: tf.Tensor = tf.nn.relu(monotonic_pair_margin - diffs)
            return tf.reduce_mean(violations)

        return tf.cond(count < 2, lambda: tf.constant(0.0, dtype=tf.float32), _compute)

    def interpolation_pair_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Penalize local slope mismatches so nearby temperatures interpolate smoothly."""
        y_true_flat: tf.Tensor = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_flat: tf.Tensor = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        count: tf.Tensor = tf.shape(y_true_flat)[0]

        def _compute() -> tf.Tensor:
            true_i: tf.Tensor = tf.expand_dims(y_true_flat, axis=1)
            true_j: tf.Tensor = tf.expand_dims(y_true_flat, axis=0)
            pred_i: tf.Tensor = tf.expand_dims(y_pred_flat, axis=1)
            pred_j: tf.Tensor = tf.expand_dims(y_pred_flat, axis=0)

            true_diff: tf.Tensor = true_i - true_j
            pred_diff: tf.Tensor = pred_i - pred_j
            abs_true_diff: tf.Tensor = tf.abs(true_diff)
            pair_weights: tf.Tensor = tf.exp(
                -abs_true_diff / tf.constant(interpolation_pair_scale, dtype=tf.float32)
            )
            pair_mask: tf.Tensor = 1.0 - tf.eye(count, dtype=tf.float32)
            pair_error: tf.Tensor = tf.abs(pred_diff - true_diff)
            weighted_error: tf.Tensor = pair_weights * pair_mask * pair_error
            normalizer: tf.Tensor = tf.reduce_sum(pair_weights * pair_mask)
            return tf.math.divide_no_nan(tf.reduce_sum(weighted_error), normalizer)

        return tf.cond(count < 2, lambda: tf.constant(0.0, dtype=tf.float32), _compute)

    def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Blend pointwise regression loss with interpolation and ordering penalties."""
        mse: tf.Tensor = tf.reduce_mean(
            tf.square(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32))
        )
        total_loss: tf.Tensor = mse
        if monotonic_pair_strength > 0.0:
            total_loss = total_loss + monotonic_pair_strength * monotonic_pair_loss(
                y_true, y_pred
            )
        if interpolation_pair_strength > 0.0:
            total_loss = total_loss + interpolation_pair_strength * interpolation_pair_loss(
                y_true, y_pred
            )
        return total_loss

    return combined_loss


def _make_pinball_loss(quantile: float):
    """Create a pinball loss for a scalar quantile head."""
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1).")

    quantile_const: tf.Tensor = tf.constant(quantile, dtype=tf.float32)

    def pinball_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Penalize under- and over-estimates according to the target quantile."""
        error: tf.Tensor = tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(
            tf.maximum(
                quantile_const * error,
                (quantile_const - 1.0) * error,
            )
        )

    pinball_loss.__name__ = f"pinball_q{int(round(quantile * 100.0)):02d}"
    return pinball_loss


def _compile_regression_model(
    model: keras.Model,
    *,
    learning_rate: float,
    monotonic_pair_strength: float = 0.0,
    monotonic_pair_margin: float = 0.0,
    interpolation_pair_strength: float = 0.0,
    interpolation_pair_scale: float = DEFAULT_INTERPOLATION_PAIR_SCALE,
) -> None:
    """Compile a scalar regression model with standard losses and metrics."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=_make_scalar_regression_loss(
            monotonic_pair_strength=monotonic_pair_strength,
            monotonic_pair_margin=monotonic_pair_margin,
            interpolation_pair_strength=interpolation_pair_strength,
            interpolation_pair_scale=interpolation_pair_scale,
        ),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )


def _compile_fraction_model(
    model: keras.Model,
    *,
    learning_rate: float,
    fraction_loss_weight: float = DEFAULT_SWEEP_FRACTION_LOSS_WEIGHT,
) -> None:
    """Compile a sweep-fraction model with scalar and fraction losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "sweep_fraction": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "gauge_value": 1.0,
            "sweep_fraction": fraction_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "sweep_fraction": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        },
    )


def _compile_keypoint_model(
    model: keras.Model,
    *,
    learning_rate: float,
    heatmap_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
) -> None:
    """Compile a keypoint-auxiliary model with scalar and heatmap losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "keypoint_heatmaps": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "gauge_value": 1.0,
            "keypoint_heatmaps": heatmap_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "keypoint_heatmaps": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        },
    )


def _compile_geometry_model(
    model: keras.Model,
    *,
    learning_rate: float,
    heatmap_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    coord_loss_weight: float = DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
    value_loss_weight: float = DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT,
) -> None:
    """Compile a geometry detector with heatmap, coordinate, and value losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "keypoint_heatmaps": keras.losses.MeanSquaredError(),
            "keypoint_coords": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "gauge_value": value_loss_weight,
            "keypoint_heatmaps": heatmap_loss_weight,
            "keypoint_coords": coord_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "keypoint_heatmaps": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
            "keypoint_coords": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                _make_keypoint_angle_mae_metric(),
            ],
        },
    )


def _compile_geometry_uncertainty_model(
    model: keras.Model,
    *,
    learning_rate: float,
    heatmap_loss_weight: float = DEFAULT_KEYPOINT_HEATMAP_LOSS_WEIGHT,
    coord_loss_weight: float = DEFAULT_KEYPOINT_COORD_LOSS_WEIGHT,
    value_loss_weight: float = DEFAULT_GEOMETRY_VALUE_LOSS_WEIGHT,
    uncertainty_loss_weight: float = DEFAULT_GEOMETRY_UNCERTAINTY_LOSS_WEIGHT,
    low_quantile: float = DEFAULT_GEOMETRY_UNCERTAINTY_LOW_QUANTILE,
    high_quantile: float = DEFAULT_GEOMETRY_UNCERTAINTY_HIGH_QUANTILE,
) -> None:
    """Compile a geometry model with symmetric uncertainty bounds."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "keypoint_heatmaps": keras.losses.MeanSquaredError(),
            "keypoint_coords": keras.losses.MeanSquaredError(),
            "gauge_value_lower": _make_pinball_loss(low_quantile),
            "gauge_value_upper": _make_pinball_loss(high_quantile),
        },
        loss_weights={
            "gauge_value": value_loss_weight,
            "keypoint_heatmaps": heatmap_loss_weight,
            "keypoint_coords": coord_loss_weight,
            "gauge_value_lower": uncertainty_loss_weight,
            "gauge_value_upper": uncertainty_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "keypoint_heatmaps": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
            "keypoint_coords": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                _make_keypoint_angle_mae_metric(),
            ],
            "gauge_value_lower": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
            "gauge_value_upper": [
                keras.metrics.MeanAbsoluteError(name="mae"),
            ],
        },
    )


def _compile_rectifier_model(
    model: keras.Model,
    *,
    learning_rate: float,
) -> None:
    """Compile a rectifier model that regresses the normalized crop box."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "rectifier_box": keras.losses.Huber(delta=0.05),
        },
        metrics={
            "rectifier_box": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        },
    )


def _compile_obb_model(
    model: keras.Model,
    *,
    learning_rate: float,
) -> None:
    """Compile an oriented-box localizer with a compact regression loss."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "obb_params": keras.losses.Huber(delta=0.05),
        },
        metrics={
            "obb_params": [
                keras.metrics.MeanAbsoluteError(name="obb_params_mae"),
                keras.metrics.RootMeanSquaredError(name="obb_params_rmse"),
            ],
        },
    )


def _compile_interval_model(
    model: keras.Model,
    *,
    learning_rate: float,
    monotonic_pair_strength: float = 0.0,
    monotonic_pair_margin: float = 0.0,
    interval_loss_weight: float = 0.25,
) -> None:
    """Compile the hybrid interval model with scalar and coarse-bin losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )
    scalar_loss = _make_scalar_regression_loss(
        monotonic_pair_strength=monotonic_pair_strength,
        monotonic_pair_margin=monotonic_pair_margin,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": scalar_loss,
            "interval_logits": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        },
        loss_weights={
            "gauge_value": 1.0,
            "interval_logits": interval_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "interval_logits": [
                keras.metrics.SparseCategoricalAccuracy(name="acc"),
            ],
        },
    )


def _compile_ordinal_model(
    model: keras.Model,
    *,
    learning_rate: float,
    ordinal_loss_weight: float = DEFAULT_ORDINAL_LOSS_WEIGHT,
) -> None:
    """Compile an ordinal-threshold model with scalar and ordinal losses."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
            "ordinal_logits": keras.losses.BinaryCrossentropy(from_logits=True),
        },
        loss_weights={
            "gauge_value": 1.0,
            "ordinal_logits": ordinal_loss_weight,
        },
        metrics={
            "gauge_value": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        },
    )


def _direction_cosine_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Optimize angular agreement between unit needle vectors."""
    dot: tf.Tensor = tf.reduce_sum(y_true * y_pred, axis=-1)
    clipped: tf.Tensor = tf.clip_by_value(dot, -1.0, 1.0)
    return tf.reduce_mean(1.0 - clipped)


def _compile_direction_model(
    model: keras.Model,
    *,
    learning_rate: float,
    spec: GaugeSpec,
) -> None:
    """Compile a direction-regression model with value-space metrics."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=_direction_cosine_loss,
        metrics=[
            _make_angle_mae_metric(),
            _make_value_mae_metric(spec),
            _make_value_rmse_metric(spec),
        ],
    )


def _make_training_callbacks(*, monitor: str = "val_mae") -> list[keras.callbacks.Callback]:
    """Build standard callbacks used for the main training/fine-tuning phase."""
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
    ]


def _merge_histories(
    first: keras.callbacks.History,
    second: keras.callbacks.History,
) -> keras.callbacks.History:
    """Concatenate metric histories from staged training runs into one History."""
    merged_keys: set[str] = set(first.history) | set(second.history)
    merged: dict[str, list[float]] = {}
    for key in merged_keys:
        vals_a: list[float] = [float(v) for v in first.history.get(key, [])]
        vals_b: list[float] = [float(v) for v in second.history.get(key, [])]
        merged[key] = vals_a + vals_b
    second.history = merged
    return second


def _vectors_to_values_tf(y: tf.Tensor, spec: GaugeSpec) -> tf.Tensor:
    """Convert unit needle direction vectors (dx, dy) to calibrated gauge values."""
    min_angle: tf.Tensor = tf.constant(spec.min_angle_rad, dtype=tf.float32)
    sweep: tf.Tensor = tf.constant(spec.sweep_rad, dtype=tf.float32)
    min_value: tf.Tensor = tf.constant(spec.min_value, dtype=tf.float32)
    value_span: tf.Tensor = tf.constant(spec.max_value - spec.min_value, dtype=tf.float32)
    two_pi: tf.Tensor = tf.constant(2.0 * math.pi, dtype=tf.float32)

    angles: tf.Tensor = tf.atan2(y[..., 1], y[..., 0])
    shifted: tf.Tensor = tf.math.floormod(angles - min_angle, two_pi)
    fractions: tf.Tensor = tf.clip_by_value(shifted / sweep, 0.0, 1.0)
    return min_value + fractions * value_span


def _make_value_mae_metric(spec: GaugeSpec):
    """Create a Keras metric function that reports Celsius MAE from direction vectors."""

    def value_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_values: tf.Tensor = _vectors_to_values_tf(y_true, spec)
        pred_values: tf.Tensor = _vectors_to_values_tf(y_pred, spec)
        return tf.reduce_mean(tf.abs(true_values - pred_values))

    value_mae.__name__ = "mae"
    return value_mae


def _make_value_rmse_metric(spec: GaugeSpec):
    """Create a Keras metric function that reports Celsius RMSE from direction vectors."""

    def value_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_values: tf.Tensor = _vectors_to_values_tf(y_true, spec)
        pred_values: tf.Tensor = _vectors_to_values_tf(y_pred, spec)
        sq_err: tf.Tensor = tf.square(true_values - pred_values)
        return tf.sqrt(tf.reduce_mean(sq_err))

    value_rmse.__name__ = "rmse"
    return value_rmse


def _make_angle_mae_metric():
    """Create a Keras metric function that reports absolute angle error in degrees."""

    def angle_mae_deg(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_angle: tf.Tensor = tf.atan2(y_true[..., 1], y_true[..., 0])
        pred_angle: tf.Tensor = tf.atan2(y_pred[..., 1], y_pred[..., 0])
        delta: tf.Tensor = pred_angle - true_angle
        wrapped: tf.Tensor = tf.atan2(tf.sin(delta), tf.cos(delta))
        deg_abs: tf.Tensor = tf.abs(wrapped) * (180.0 / math.pi)
        return tf.reduce_mean(deg_abs)

    angle_mae_deg.__name__ = "angle_mae_deg"
    return angle_mae_deg


def _make_keypoint_angle_mae_metric():
    """Report angular error in degrees from predicted center/tip coordinates."""

    def angle_mae_deg(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_center: tf.Tensor = y_true[..., 0, :]
        true_tip: tf.Tensor = y_true[..., 1, :]
        pred_center: tf.Tensor = y_pred[..., 0, :]
        pred_tip: tf.Tensor = y_pred[..., 1, :]

        true_angle: tf.Tensor = tf.atan2(
            true_tip[..., 1] - true_center[..., 1],
            true_tip[..., 0] - true_center[..., 0],
        )
        pred_angle: tf.Tensor = tf.atan2(
            pred_tip[..., 1] - pred_center[..., 1],
            pred_tip[..., 0] - pred_center[..., 0],
        )
        delta: tf.Tensor = pred_angle - true_angle
        wrapped: tf.Tensor = tf.atan2(tf.sin(delta), tf.cos(delta))
        deg_abs: tf.Tensor = tf.abs(wrapped) * (180.0 / math.pi)
        return tf.reduce_mean(deg_abs)

    angle_mae_deg.__name__ = "angle_mae_deg"
    return angle_mae_deg


def train(config: TrainConfig) -> TrainingResult:
    """Run one full training cycle and return model + metrics."""
    print("[TRAIN] Entering train().")
    print("[TRAIN] step: validate-split", flush=True)
    _validate_split_config(config)
    print("[TRAIN] Split config validated.")
    print("[TRAIN] step: configure-runtime", flush=True)
    _configure_training_runtime(config)
    print("[TRAIN] Runtime configured.")
    print("[TRAIN] step: log-runtime-state", flush=True)
    _log_runtime_state(config)

    print("[TRAIN] step: set-seeds", flush=True)
    keras.utils.set_random_seed(config.seed)
    np.random.seed(config.seed)

    print("[TRAIN] step: load-gauge-specs", flush=True)
    specs: dict[str, GaugeSpec] = load_gauge_specs()
    if config.gauge_id not in specs:
        raise ValueError(
            f"Unknown gauge_id '{config.gauge_id}'. Available: {list(specs)}"
        )
    spec: GaugeSpec = specs[config.gauge_id]
    print(f"[TRAIN] Loaded gauge spec for '{config.gauge_id}'.")

    print("[TRAIN] step: load-labelled-dataset", flush=True)
    print("[TRAIN] Loading labelled dataset...")
    samples: list[Sample] = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
    if not samples:
        raise ValueError(
            "No samples found. Check labelled/raw paths and annotation zips."
        )
    print(f"[TRAIN] Loaded {len(samples)} labelled samples.")

    # This summary is still useful for visibility into raw annotation quality.
    print("[TRAIN] step: summarize-label-sweep", flush=True)
    print("[TRAIN] Summarizing label sweep...")
    label_summary: LabelSummary = summarize_label_sweep(samples, spec)
    print("[TRAIN] Label sweep summary complete.")

    print("[TRAIN] step: build-training-examples", flush=True)
    print("[TRAIN] Building training examples...")
    examples, dropped_out_of_sweep = _build_training_examples(
        samples,
        spec,
        image_height=config.image_height,
        image_width=config.image_width,
        keypoint_heatmap_size=config.keypoint_heatmap_size,
        strict_labels=config.strict_labels,
        crop_pad_ratio=config.crop_pad_ratio,
    )
    if len(examples) < 3:
        raise ValueError(
            "Not enough valid examples after filtering invalid sweep labels."
    )
    print(
        "[TRAIN] Training examples ready: "
        f"examples={len(examples)} dropped_out_of_sweep={dropped_out_of_sweep}"
    )

    print("[TRAIN] step: split-examples", flush=True)
    print("[TRAIN] Splitting dataset...")
    split: DatasetSplit = _split_examples(examples, config)
    example_lookup: dict[str, TrainingExample] = {
        example.image_path: example for example in examples
    }
    reserved_names: set[str] = {
        Path(example.image_path).name
        for example in split.val_examples + split.test_examples
    }

    if config.model_family == "mobilenet_v2_obb" and config.hard_case_manifest:
        raise ValueError(
            "mobilenet_v2_obb does not support hard_case_manifest. "
            "Use the labeled dataset split with val/test fractions instead."
        )

    if config.hard_case_manifest:
        print("[TRAIN] step: load-hard-cases", flush=True)
        hard_case_manifest_path: Path = Path(config.hard_case_manifest)
        if not hard_case_manifest_path.is_absolute():
            hard_case_manifest_path = ML_ROOT / hard_case_manifest_path
        hard_case_target_kind: Literal[
            "value",
            "needle_unit_xy",
            "sweep_fraction",
            "keypoint_heatmaps",
            "geometry",
            "geometry_uncertainty",
            "ordinal_thresholds",
            "rectifier_box",
        ] = (
            "value"
            if config.model_family == "mobilenet_v2_rectifier"
            else "needle_unit_xy"
            if config.model_family in {"mobilenet_v2_direction", "compact_direction"}
            else "sweep_fraction"
            if config.model_family == "mobilenet_v2_fraction"
            else "keypoint_heatmaps"
            if config.model_family in {"mobilenet_v2_keypoint", "mobilenet_v2_detector"}
            else "geometry_uncertainty"
            if config.model_family == "mobilenet_v2_geometry_uncertainty"
            else "geometry"
            if config.model_family in {"mobilenet_v2_geometry", "compact_geometry"}
            else "ordinal_thresholds"
            if config.model_family == "mobilenet_v2_ordinal"
            else "value"
        )
        hard_case_examples: list[TrainingExample] = _load_hard_case_examples(
            hard_case_manifest_path,
            image_height=config.image_height,
            image_width=config.image_width,
            value_range=(spec.min_value, spec.max_value),
            target_kind=hard_case_target_kind,
            spec=spec if hard_case_target_kind == "needle_unit_xy" else None,
        )
        filtered_hard_case_examples: list[TrainingExample] = [
            example
            for example in hard_case_examples
            if Path(example.image_path).name not in reserved_names
        ]
        if len(filtered_hard_case_examples) != len(hard_case_examples):
            print(
                "[TRAIN] Hard-case manifest overlap filtered: "
                f"kept={len(filtered_hard_case_examples)} "
                f"dropped={len(hard_case_examples) - len(filtered_hard_case_examples)}",
                flush=True,
            )
        hard_case_examples = filtered_hard_case_examples
        hard_case_repeat: int = max(config.hard_case_repeat, 0)
        if config.model_family == "mobilenet_v2_rectifier" and hard_case_examples:
            # Reuse labeled rectifier boxes from nearby-temperature examples so the
            # hard-case manifest can upweight the failing temperature regions.
            selected_hard_cases: list[TrainingExample] = []
            for hard_case_example in hard_case_examples:
                nearest_example = min(
                    examples,
                    key=lambda example: abs(example.value - hard_case_example.value),
                )
                selected_hard_cases.append(nearest_example)
            repeated_hard_cases = selected_hard_cases * hard_case_repeat
            split = DatasetSplit(
                train_examples=split.train_examples + repeated_hard_cases,
                val_examples=split.val_examples,
                test_examples=split.test_examples,
            )
            print(
                "[TRAIN] Hard-case rectifier examples mapped by value: "
                f"base={len(hard_case_examples)} selected={len(selected_hard_cases)} "
                f"repeat={hard_case_repeat} added={len(repeated_hard_cases)}"
            )
        elif hard_case_repeat > 0 and hard_case_examples:
            repeated_hard_cases: list[TrainingExample] = hard_case_examples * hard_case_repeat
            split = DatasetSplit(
                train_examples=split.train_examples + repeated_hard_cases,
                val_examples=split.val_examples,
                test_examples=split.test_examples,
            )
            print(
                "[TRAIN] Hard-case fine-tuning examples loaded: "
                f"base={len(hard_case_examples)} repeat={hard_case_repeat} "
                f"added={len(repeated_hard_cases)}"
            )

    if config.precomputed_crop_boxes_path:
        print("[TRAIN] step: apply-precomputed-crop-boxes", flush=True)
        boxes_path: Path = Path(config.precomputed_crop_boxes_path)
        if not boxes_path.is_absolute():
            boxes_path = ML_ROOT / boxes_path
        split = DatasetSplit(
            train_examples=_apply_precomputed_crop_boxes(split.train_examples, csv_path=boxes_path),
            val_examples=_apply_precomputed_crop_boxes(split.val_examples, csv_path=boxes_path),
            test_examples=_apply_precomputed_crop_boxes(split.test_examples, csv_path=boxes_path),
        )
    elif config.rectifier_model_path:
        print("[TRAIN] step: rectified-scalar-preprocess", flush=True)
        rectifier_model_path: Path = Path(config.rectifier_model_path)
        if not rectifier_model_path.is_absolute():
            rectifier_model_path = ML_ROOT / rectifier_model_path
        split = DatasetSplit(
            train_examples=_rectify_examples_for_scalar(
                split.train_examples,
                rectifier_model_path=rectifier_model_path,
                image_size=config.image_height,
                rectifier_crop_scale=config.rectifier_crop_scale,
            ),
            val_examples=_rectify_examples_for_scalar(
                split.val_examples,
                rectifier_model_path=rectifier_model_path,
                image_size=config.image_height,
                rectifier_crop_scale=config.rectifier_crop_scale,
            ),
            test_examples=_rectify_examples_for_scalar(
                split.test_examples,
                rectifier_model_path=rectifier_model_path,
                image_size=config.image_height,
                rectifier_crop_scale=config.rectifier_crop_scale,
            ),
        )
        print(
            "[TRAIN] Rectified scalar crops generated: "
            f"scale={config.rectifier_crop_scale:.2f}",
            flush=True,
        )

    print("[TRAIN] step: log-dataset-state", flush=True)
    _log_dataset_state(
        config,
        label_summary=label_summary,
        split=split,
        dropped_out_of_sweep=dropped_out_of_sweep,
    )

    print("[TRAIN] step: model-selection", flush=True)
    _log_model_choice(config)
    model: keras.Model | None = None
    init_model: keras.Model | None = None
    if config.init_model_path:
        print("[TRAIN] step: load-init-model", flush=True)
        init_model_path: Path = Path(config.init_model_path)
        if not init_model_path.is_absolute():
            init_model_path = ML_ROOT / init_model_path
        if config.model_family in {
            "mobilenet_v2_interval",
            "compact_interval",
            "compact_geometry",
            "mobilenet_v2_geometry_uncertainty",
            "mobilenet_v2_rectifier",
            "mobilenet_v2_ordinal",
            "mobilenet_v2_fraction",
            "mobilenet_v2_keypoint",
            "mobilenet_v2_detector",
            "mobilenet_v2_geometry",
            "mobilenet_v2_obb",
        }:
            init_model = _load_init_model(init_model_path)
        else:
            model = _load_init_model(init_model_path)
    if model is None:
        print("[TRAIN] step: build-model", flush=True)
        if config.model_family == "compact":
            print("[TRAIN] Building compact CNN model...")
            model = build_regression_model(config.image_height, config.image_width)
        elif config.model_family == "compact_direction":
            print("[TRAIN] Building compact CNN direction model...")
            model = build_needle_direction_model(config.image_height, config.image_width)
        elif config.model_family == "compact_interval":
            print("[TRAIN] Building compact CNN interval model...")
            model = build_compact_interval_model(
                config.image_height,
                config.image_width,
                value_min=spec.min_value,
                value_max=spec.max_value,
                bin_width=config.interval_bin_width,
            )
        elif config.model_family == "compact_geometry":
            print("[TRAIN] Building compact CNN geometry model...")
            model = build_compact_geometry_model(
                config.image_height,
                config.image_width,
                value_min=spec.min_value,
                value_max=spec.max_value,
                min_angle_rad=spec.min_angle_rad,
                sweep_rad=spec.sweep_rad,
                heatmap_size=config.keypoint_heatmap_size,
            )
        elif config.model_family == "mobilenet_v2":
            print("[TRAIN] Building MobileNetV2 model...")
            model = build_mobilenetv2_regression_model(
                config.image_height,
                config.image_width,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_tiny":
            print("[TRAIN] Building tiny MobileNetV2 model...")
            model = build_mobilenetv2_regression_model(
                config.image_height,
                config.image_width,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_direction":
            print("[TRAIN] Building MobileNetV2 direction model...")
            model = build_mobilenetv2_direction_model(
                config.image_height,
                config.image_width,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_fraction":
            print("[TRAIN] Building MobileNetV2 sweep-fraction model...")
            model = build_mobilenetv2_fraction_model(
                config.image_height,
                config.image_width,
                value_min=spec.min_value,
                value_max=spec.max_value,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_detector":
            print("[TRAIN] Building MobileNetV2 detector-first model...")
            model = build_mobilenetv2_detector_model(
                config.image_height,
                config.image_width,
                value_min=spec.min_value,
                value_max=spec.max_value,
                min_angle_rad=spec.min_angle_rad,
                sweep_rad=spec.sweep_rad,
                heatmap_size=config.keypoint_heatmap_size,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_geometry":
            print("[TRAIN] Building MobileNetV2 geometry-detector model...")
            model = build_mobilenetv2_geometry_model(
                config.image_height,
                config.image_width,
                value_min=spec.min_value,
                value_max=spec.max_value,
                min_angle_rad=spec.min_angle_rad,
                sweep_rad=spec.sweep_rad,
                heatmap_size=config.keypoint_heatmap_size,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_geometry_uncertainty":
            print("[TRAIN] Building MobileNetV2 geometry-uncertainty model...")
            model = build_mobilenetv2_geometry_uncertainty_model(
                config.image_height,
                config.image_width,
                value_min=spec.min_value,
                value_max=spec.max_value,
                min_angle_rad=spec.min_angle_rad,
                sweep_rad=spec.sweep_rad,
                heatmap_size=config.keypoint_heatmap_size,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_obb":
            print("[TRAIN] Building MobileNetV2 OBB localizer model...")
            model = build_mobilenetv2_obb_model(
                config.image_height,
                config.image_width,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_rectifier":
            print("[TRAIN] Building MobileNetV2 rectifier model...")
            model = build_mobilenetv2_rectifier_model(
                config.image_height,
                config.image_width,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_keypoint":
            print("[TRAIN] Building MobileNetV2 keypoint-heatmap model...")
            model = build_mobilenetv2_keypoint_model(
                config.image_height,
                config.image_width,
                heatmap_size=config.keypoint_heatmap_size,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_interval":
            print("[TRAIN] Building MobileNetV2 interval model...")
            model = build_mobilenetv2_interval_model(
                config.image_height,
                config.image_width,
                value_min=spec.min_value,
                value_max=spec.max_value,
                bin_width=config.interval_bin_width,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        elif config.model_family == "mobilenet_v2_ordinal":
            print("[TRAIN] Building MobileNetV2 ordinal model...")
            model = build_mobilenetv2_ordinal_model(
                config.image_height,
                config.image_width,
                value_min=spec.min_value,
                value_max=spec.max_value,
                threshold_step=config.ordinal_threshold_step,
                pretrained=config.mobilenet_pretrained,
                backbone_trainable=config.mobilenet_backbone_trainable,
                alpha=config.mobilenet_alpha,
                head_units=config.mobilenet_head_units,
                head_dropout=config.mobilenet_head_dropout,
            )
        else:
            raise ValueError(f"Unsupported model_family '{config.model_family}'.")
    if init_model is not None:
        print("[TRAIN] step: transfer-init-weights", flush=True)
        _transfer_matching_weights(init_model, model)
        model.trainable = True
    target_kind: Literal[
        "value",
        "needle_unit_xy",
        "sweep_fraction",
        "interval_value",
        "ordinal_thresholds",
        "keypoint_heatmaps",
        "geometry",
        "geometry_uncertainty",
        "rectifier_box",
        "obb",
    ] = (
        "needle_unit_xy"
        if config.model_family
        in {"mobilenet_v2_direction", "compact_direction"}
        else "sweep_fraction"
        if config.model_family == "mobilenet_v2_fraction"
        else "keypoint_heatmaps"
        if config.model_family in {"mobilenet_v2_keypoint", "mobilenet_v2_detector"}
        else "geometry_uncertainty"
        if config.model_family == "mobilenet_v2_geometry_uncertainty"
        else "rectifier_box"
        if config.model_family == "mobilenet_v2_rectifier"
        else "obb"
        if config.model_family == "mobilenet_v2_obb"
        else "geometry"
        if config.model_family in {"mobilenet_v2_geometry", "compact_geometry"}
        else "ordinal_thresholds"
        if config.model_family == "mobilenet_v2_ordinal"
        else "interval_value"
        if config.model_family in {"mobilenet_v2_interval", "compact_interval"}
        else "value"
    )
    print("[TRAIN] step: build-datasets", flush=True)
    train_ds = _build_tf_dataset(
        split.train_examples,
        config,
        training=True,
        target_kind=target_kind,
        value_range=(spec.min_value, spec.max_value)
        if target_kind
        in {
            "interval_value",
            "ordinal_thresholds",
            "sweep_fraction",
            "keypoint_heatmaps",
            "geometry",
            "geometry_uncertainty",
            "rectifier_box",
            "obb",
        }
        else None,
    )
    val_ds = _build_tf_dataset(
        split.val_examples,
        config,
        training=False,
        target_kind=target_kind,
        value_range=(spec.min_value, spec.max_value)
        if target_kind
        in {
            "interval_value",
            "ordinal_thresholds",
            "sweep_fraction",
            "keypoint_heatmaps",
            "geometry",
            "geometry_uncertainty",
            "rectifier_box",
            "obb",
        }
        else None,
    )
    test_ds = _build_tf_dataset(
        split.test_examples,
        config,
        training=False,
        target_kind=target_kind,
        value_range=(spec.min_value, spec.max_value)
        if target_kind
        in {
            "interval_value",
            "ordinal_thresholds",
            "sweep_fraction",
            "keypoint_heatmaps",
            "geometry",
            "geometry_uncertainty",
            "rectifier_box",
            "obb",
        }
        else None,
    )
    print("[TRAIN] TensorFlow datasets built.")
    print("[TRAIN] step: callbacks", flush=True)
    model_name: str = getattr(model, "name", model.__class__.__name__)
    if hasattr(model, "count_params"):
        model_params: str = f"{int(model.count_params()):,}"
    else:
        model_params = "unknown"
    print(f"[TRAIN] Built model '{model_name}' with {model_params} parameters.")
    monitor_metric: str = (
        "val_mae"
        if config.model_family == "mobilenet_v2_rectifier"
        else "val_gauge_value_mae"
        if config.model_family
        in {
            "mobilenet_v2_interval",
            "compact_interval",
            "compact_geometry",
            "mobilenet_v2_geometry_uncertainty",
            "mobilenet_v2_ordinal",
            "mobilenet_v2_fraction",
            "mobilenet_v2_keypoint",
            "mobilenet_v2_detector",
            "mobilenet_v2_geometry",
        }
        else "val_obb_params_mae"
        if config.model_family == "mobilenet_v2_obb"
        else "val_mae"
    )
    callbacks: list[keras.callbacks.Callback] = _make_training_callbacks(
        monitor=monitor_metric
    )
    print("[TRAIN] step: maybe-two-stage", flush=True)
    should_use_two_stage_mobilenet: bool = (
        config.model_family
        in {
            "mobilenet_v2",
            "mobilenet_v2_tiny",
            "mobilenet_v2_direction",
            "mobilenet_v2_fraction",
            "mobilenet_v2_detector",
            "mobilenet_v2_geometry",
            "mobilenet_v2_geometry_uncertainty",
            "mobilenet_v2_obb",
            "mobilenet_v2_rectifier",
            "mobilenet_v2_keypoint",
            "mobilenet_v2_interval",
            "mobilenet_v2_ordinal",
        }
        and config.mobilenet_pretrained
        and config.mobilenet_backbone_trainable
        and config.mobilenet_warmup_epochs > 0
        and config.epochs > 1
        and config.init_model_path is None
    )

    if should_use_two_stage_mobilenet:
        print("[TRAIN] step: two-stage-start", flush=True)
        warmup_epochs: int = min(config.mobilenet_warmup_epochs, config.epochs - 1)
        backbone = getattr(model, "_mobilenet_backbone", None)
        if backbone is None:
            raise RuntimeError(
                "MobileNetV2 staged training requested, but backbone handle was not found."
            )

        # Stage 1: train regression head with frozen pretrained backbone.
        print(
            "[TRAIN] MobileNetV2 stage 1: "
            f"warmup_epochs={warmup_epochs} "
            "backbone_trainable=False"
        )
        backbone.trainable = False
        if config.model_family in {"mobilenet_v2_direction", "compact_direction"}:
            _compile_direction_model(
                model,
                learning_rate=config.learning_rate,
                spec=spec,
            )
        elif config.model_family == "mobilenet_v2_fraction":
            _compile_fraction_model(
                model,
                learning_rate=config.learning_rate,
                fraction_loss_weight=config.sweep_fraction_loss_weight,
            )
        elif config.model_family in {
            "mobilenet_v2_keypoint",
            "mobilenet_v2_detector",
        }:
            _compile_keypoint_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
            )
        elif config.model_family == "compact_geometry":
            _compile_geometry_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
                coord_loss_weight=config.keypoint_coord_loss_weight,
                value_loss_weight=config.geometry_value_loss_weight,
            )
        elif config.model_family == "mobilenet_v2_geometry":
            _compile_geometry_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
                coord_loss_weight=config.keypoint_coord_loss_weight,
                value_loss_weight=config.geometry_value_loss_weight,
            )
        elif config.model_family == "mobilenet_v2_geometry_uncertainty":
            _compile_geometry_uncertainty_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
                coord_loss_weight=config.keypoint_coord_loss_weight,
                value_loss_weight=config.geometry_value_loss_weight,
                uncertainty_loss_weight=config.geometry_uncertainty_loss_weight,
                low_quantile=config.geometry_uncertainty_low_quantile,
                high_quantile=config.geometry_uncertainty_high_quantile,
            )
        elif config.model_family == "mobilenet_v2_obb":
            _compile_obb_model(
                model,
                learning_rate=config.learning_rate,
            )
        elif config.model_family == "mobilenet_v2_rectifier":
            _compile_rectifier_model(
                model,
                learning_rate=config.learning_rate,
            )
        elif config.model_family == "mobilenet_v2_interval":
            _compile_interval_model(
                model,
                learning_rate=config.learning_rate,
                monotonic_pair_strength=config.monotonic_pair_strength,
                monotonic_pair_margin=config.monotonic_pair_margin,
            )
        elif config.model_family == "compact_interval":
            _compile_interval_model(
                model,
                learning_rate=config.learning_rate,
                monotonic_pair_strength=config.monotonic_pair_strength,
                monotonic_pair_margin=config.monotonic_pair_margin,
            )
        elif config.model_family == "mobilenet_v2_ordinal":
            _compile_ordinal_model(
                model,
                learning_rate=config.learning_rate,
                ordinal_loss_weight=config.ordinal_loss_weight,
            )
        else:
            _compile_regression_model(
                model,
                learning_rate=config.learning_rate,
                monotonic_pair_strength=config.monotonic_pair_strength,
                monotonic_pair_margin=config.monotonic_pair_margin,
                interpolation_pair_strength=config.interpolation_pair_strength,
                interpolation_pair_scale=config.interpolation_pair_scale,
            )
        warmup_history: keras.callbacks.History = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            callbacks=[],
            verbose=2,
        )

        # Stage 2: unfreeze backbone and fine-tune end-to-end with callbacks.
        print(
            "[TRAIN] MobileNetV2 stage 2: "
            f"fine_tune_epochs={config.epochs - warmup_epochs} "
            "backbone_trainable=True"
        )
        backbone.trainable = True
        if config.model_family in {"mobilenet_v2_direction", "compact_direction"}:
            _compile_direction_model(
                model,
                learning_rate=config.learning_rate,
                spec=spec,
            )
        elif config.model_family == "mobilenet_v2_fraction":
            _compile_fraction_model(
                model,
                learning_rate=config.learning_rate,
                fraction_loss_weight=config.sweep_fraction_loss_weight,
            )
        elif config.model_family in {
            "mobilenet_v2_keypoint",
            "mobilenet_v2_detector",
        }:
            _compile_keypoint_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
            )
        elif config.model_family == "compact_geometry":
            _compile_geometry_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
                coord_loss_weight=config.keypoint_coord_loss_weight,
                value_loss_weight=config.geometry_value_loss_weight,
            )
        elif config.model_family == "mobilenet_v2_geometry":
            _compile_geometry_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
                coord_loss_weight=config.keypoint_coord_loss_weight,
                value_loss_weight=config.geometry_value_loss_weight,
            )
        elif config.model_family == "mobilenet_v2_geometry_uncertainty":
            _compile_geometry_uncertainty_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
                coord_loss_weight=config.keypoint_coord_loss_weight,
                value_loss_weight=config.geometry_value_loss_weight,
                uncertainty_loss_weight=config.geometry_uncertainty_loss_weight,
                low_quantile=config.geometry_uncertainty_low_quantile,
                high_quantile=config.geometry_uncertainty_high_quantile,
            )
        elif config.model_family == "mobilenet_v2_obb":
            _compile_obb_model(
                model,
                learning_rate=config.learning_rate,
            )
        elif config.model_family == "mobilenet_v2_rectifier":
            _compile_rectifier_model(
                model,
                learning_rate=config.learning_rate,
            )
        elif config.model_family == "mobilenet_v2_interval":
            _compile_interval_model(
                model,
                learning_rate=config.learning_rate,
                monotonic_pair_strength=config.monotonic_pair_strength,
                monotonic_pair_margin=config.monotonic_pair_margin,
            )
        elif config.model_family == "compact_interval":
            _compile_interval_model(
                model,
                learning_rate=config.learning_rate,
                monotonic_pair_strength=config.monotonic_pair_strength,
                monotonic_pair_margin=config.monotonic_pair_margin,
            )
        elif config.model_family == "mobilenet_v2_ordinal":
            _compile_ordinal_model(
                model,
                learning_rate=config.learning_rate,
                ordinal_loss_weight=config.ordinal_loss_weight,
            )
        else:
            _compile_regression_model(
                model,
                learning_rate=config.learning_rate,
                monotonic_pair_strength=config.monotonic_pair_strength,
                monotonic_pair_margin=config.monotonic_pair_margin,
                interpolation_pair_strength=config.interpolation_pair_strength,
                interpolation_pair_scale=config.interpolation_pair_scale,
            )
        fine_tune_history: keras.callbacks.History = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            initial_epoch=warmup_epochs,
            callbacks=callbacks,
            verbose=2,
        )
        history = _merge_histories(warmup_history, fine_tune_history)
    else:
        print("[TRAIN] step: single-stage-fit", flush=True)
        print("[TRAIN] Compact CNN stage: training end-to-end with callbacks.")
        if config.model_family in {"mobilenet_v2_direction", "compact_direction"}:
            _compile_direction_model(
                model,
                learning_rate=config.learning_rate,
                spec=spec,
            )
        elif config.model_family == "mobilenet_v2_fraction":
            _compile_fraction_model(
                model,
                learning_rate=config.learning_rate,
                fraction_loss_weight=config.sweep_fraction_loss_weight,
            )
        elif config.model_family in {
            "mobilenet_v2_keypoint",
            "mobilenet_v2_detector",
        }:
            _compile_keypoint_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
            )
        elif config.model_family == "compact_geometry":
            _compile_geometry_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
                coord_loss_weight=config.keypoint_coord_loss_weight,
                value_loss_weight=config.geometry_value_loss_weight,
            )
        elif config.model_family == "mobilenet_v2_geometry":
            _compile_geometry_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
                coord_loss_weight=config.keypoint_coord_loss_weight,
                value_loss_weight=config.geometry_value_loss_weight,
            )
        elif config.model_family == "mobilenet_v2_geometry_uncertainty":
            _compile_geometry_uncertainty_model(
                model,
                learning_rate=config.learning_rate,
                heatmap_loss_weight=config.keypoint_heatmap_loss_weight,
                coord_loss_weight=config.keypoint_coord_loss_weight,
                value_loss_weight=config.geometry_value_loss_weight,
                uncertainty_loss_weight=config.geometry_uncertainty_loss_weight,
                low_quantile=config.geometry_uncertainty_low_quantile,
                high_quantile=config.geometry_uncertainty_high_quantile,
            )
        elif config.model_family == "mobilenet_v2_obb":
            _compile_obb_model(
                model,
                learning_rate=config.learning_rate,
            )
        elif config.model_family == "mobilenet_v2_rectifier":
            _compile_rectifier_model(
                model,
                learning_rate=config.learning_rate,
            )
        elif config.model_family == "mobilenet_v2_interval":
            _compile_interval_model(
                model,
                learning_rate=config.learning_rate,
                monotonic_pair_strength=config.monotonic_pair_strength,
                monotonic_pair_margin=config.monotonic_pair_margin,
            )
        elif config.model_family == "compact_interval":
            _compile_interval_model(
                model,
                learning_rate=config.learning_rate,
                monotonic_pair_strength=config.monotonic_pair_strength,
                monotonic_pair_margin=config.monotonic_pair_margin,
            )
        elif config.model_family == "mobilenet_v2_ordinal":
            _compile_ordinal_model(
                model,
                learning_rate=config.learning_rate,
                ordinal_loss_weight=config.ordinal_loss_weight,
            )
        else:
            _compile_regression_model(
                model,
                learning_rate=config.learning_rate,
                monotonic_pair_strength=config.monotonic_pair_strength,
                monotonic_pair_margin=config.monotonic_pair_margin,
                interpolation_pair_strength=config.interpolation_pair_strength,
                interpolation_pair_scale=config.interpolation_pair_scale,
            )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
        )

    raw_metrics: dict[str, float] = model.evaluate(test_ds, return_dict=True, verbose=0)
    test_metrics: dict[str, float] = {k: float(v) for k, v in raw_metrics.items()}

    baseline_test_mae: float = _compute_mean_baseline_mae(
        split.train_examples,
        split.test_examples,
    )
    test_metrics["baseline_mae_mean_predictor"] = baseline_test_mae

    return TrainingResult(
        model=model,
        history=history,
        label_summary=label_summary,
        test_metrics=test_metrics,
        baseline_test_mae=baseline_test_mae,
        dropped_out_of_sweep=dropped_out_of_sweep,
    )
