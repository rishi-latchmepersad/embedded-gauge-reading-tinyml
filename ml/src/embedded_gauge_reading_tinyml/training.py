"""Training pipeline for gauge-value regression with dial ROI cropping."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from embedded_gauge_reading_tinyml.dataset import Sample, load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import (
    GaugeSpec,
    load_gauge_specs,
    needle_fraction,
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
    DEFAULT_SEED,
    DEFAULT_STRICT_LABELS,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
)
from embedded_gauge_reading_tinyml.models import (
    build_mobilenetv2_regression_model,
    build_mobilenetv2_direction_model,
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
    model_family: Literal[
        "compact",
        "mobilenet_v2",
        "mobilenet_v2_tiny",
        "mobilenet_v2_direction",
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
    init_model_path: str | None = None


@dataclass(frozen=True)
class TrainingExample:
    """One training row with image path, targets, and dial crop box."""

    image_path: str
    value: float
    crop_box_xyxy: tuple[float, float, float, float]
    needle_unit_xy: tuple[float, float]


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


def _configure_training_runtime(config: TrainConfig) -> None:
    """Configure TensorFlow runtime for CPU/GPU selection and memory behavior."""
    gpus: list[tf.config.PhysicalDevice] = tf.config.list_physical_devices("GPU")

    if config.device == "cpu":
        if gpus:
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError as exc:
                raise RuntimeError(
                    "Unable to force CPU mode because TensorFlow runtime is already initialized."
                ) from exc
        keras.mixed_precision.set_global_policy("float32")
        return

    if config.device == "gpu" and not gpus:
        raise ValueError(
            "device='gpu' was requested, but TensorFlow did not detect a GPU."
        )

    if gpus and config.gpu_memory_growth:
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
    visible_gpus: list[tf.config.PhysicalDevice] = tf.config.list_physical_devices(
        "GPU"
    )
    visible_cpus: list[tf.config.PhysicalDevice] = tf.config.list_physical_devices(
        "CPU"
    )
    print(
        "[TRAIN] Runtime: "
        f"device={config.device} "
        f"gpu_memory_growth={config.gpu_memory_growth} "
        f"mixed_precision={keras.mixed_precision.global_policy().name}"
    )
    print(f"[TRAIN] TensorFlow CPUs: {visible_cpus}")
    print(f"[TRAIN] TensorFlow GPUs: {visible_gpus}")


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


def _build_training_examples(
    samples: list[Sample],
    spec: GaugeSpec,
    *,
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
        dx: float = sample.tip.x - sample.center.x
        dy: float = sample.tip.y - sample.center.y
        length: float = math.hypot(dx, dy)
        if length <= 0.0:
            dropped_out_of_sweep += 1
            continue
        needle_unit_xy: tuple[float, float] = (dx / length, dy / length)
        examples.append(
            TrainingExample(
                image_path=str(sample.image_path),
                value=value,
                crop_box_xyxy=crop_box,
                needle_unit_xy=needle_unit_xy,
            )
        )

    return examples, dropped_out_of_sweep


def _load_hard_case_examples(
    manifest_path: Path,
    *,
    image_height: int,
    image_width: int,
) -> list[TrainingExample]:
    """Load extra board captures that should be upweighted during fine-tuning."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Hard-case manifest not found: {manifest_path}")

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
            examples.append(
                TrainingExample(
                    image_path=str(image_path),
                    value=value,
                    crop_box_xyxy=(0.0, 0.0, float(image_width), float(image_height)),
                    needle_unit_xy=(1.0, 0.0),
                )
            )

    return examples


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
        },
    )
    model.trainable = True
    return model


def _split_examples(
    examples: list[TrainingExample],
    config: TrainConfig,
) -> DatasetSplit:
    """Split examples into train/val/test with a fixed random seed."""
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


def _augment_image(image: tf.Tensor) -> tf.Tensor:
    """Apply light photometric augmentation that preserves gauge geometry."""
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

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)

    noise: tf.Tensor = tf.random.normal(
        shape=tf.shape(image),
        mean=0.0,
        stddev=0.01,
        dtype=tf.float32,
    )
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _build_tf_dataset(
    examples: list[TrainingExample],
    config: TrainConfig,
    *,
    training: bool,
    target_kind: Literal["value", "needle_unit_xy"] = "value",
) -> tf.data.Dataset:
    """Create a tf.data pipeline for efficient training/eval input."""
    paths: np.ndarray = np.array([e.image_path for e in examples], dtype=str)
    if target_kind == "value":
        targets: np.ndarray = np.array([e.value for e in examples], dtype=np.float32)
    elif target_kind == "needle_unit_xy":
        targets = np.array([e.needle_unit_xy for e in examples], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported target_kind '{target_kind}'.")
    boxes: np.ndarray = np.array([e.crop_box_xyxy for e in examples], dtype=np.float32)

    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((paths, targets, boxes))

    if training:
        dataset = dataset.shuffle(
            buffer_size=max(len(examples), 1),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )

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

    if training and config.augment_training:
        dataset = dataset.map(
            lambda img, y: (_augment_image(img), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


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


def _compile_regression_model(
    model: keras.Model,
    *,
    learning_rate: float,
) -> None:
    """Compile a scalar regression model with standard losses and metrics."""
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
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


def _make_training_callbacks() -> list[keras.callbacks.Callback]:
    """Build standard callbacks used for the main training/fine-tuning phase."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_mae",
            mode="min",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
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


def train(config: TrainConfig) -> TrainingResult:
    """Run one full training cycle and return model + metrics."""
    print("[TRAIN] Entering train().")
    _validate_split_config(config)
    print("[TRAIN] Split config validated.")
    _configure_training_runtime(config)
    print("[TRAIN] Runtime configured.")
    _log_runtime_state(config)

    keras.utils.set_random_seed(config.seed)
    np.random.seed(config.seed)

    specs: dict[str, GaugeSpec] = load_gauge_specs()
    if config.gauge_id not in specs:
        raise ValueError(
            f"Unknown gauge_id '{config.gauge_id}'. Available: {list(specs)}"
        )
    spec: GaugeSpec = specs[config.gauge_id]
    print(f"[TRAIN] Loaded gauge spec for '{config.gauge_id}'.")

    print("[TRAIN] Loading labelled dataset...")
    samples: list[Sample] = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
    if not samples:
        raise ValueError(
            "No samples found. Check labelled/raw paths and annotation zips."
        )
    print(f"[TRAIN] Loaded {len(samples)} labelled samples.")

    # This summary is still useful for visibility into raw annotation quality.
    print("[TRAIN] Summarizing label sweep...")
    label_summary: LabelSummary = summarize_label_sweep(samples, spec)
    print("[TRAIN] Label sweep summary complete.")

    print("[TRAIN] Building training examples...")
    examples, dropped_out_of_sweep = _build_training_examples(
        samples,
        spec,
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

    print("[TRAIN] Splitting dataset...")
    split: DatasetSplit = _split_examples(examples, config)

    if config.hard_case_manifest and config.model_family != "mobilenet_v2_direction":
        hard_case_manifest_path: Path = Path(config.hard_case_manifest)
        if not hard_case_manifest_path.is_absolute():
            hard_case_manifest_path = ML_ROOT / hard_case_manifest_path
        hard_case_examples: list[TrainingExample] = _load_hard_case_examples(
            hard_case_manifest_path,
            image_height=config.image_height,
            image_width=config.image_width,
        )
        hard_case_repeat: int = max(config.hard_case_repeat, 0)
        if hard_case_repeat > 0 and hard_case_examples:
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

    _log_dataset_state(
        config,
        label_summary=label_summary,
        split=split,
        dropped_out_of_sweep=dropped_out_of_sweep,
    )

    _log_model_choice(config)
    if config.init_model_path:
        init_model_path: Path = Path(config.init_model_path)
        if not init_model_path.is_absolute():
            init_model_path = ML_ROOT / init_model_path
        model = _load_init_model(init_model_path)
    elif config.model_family == "compact":
        print("[TRAIN] Building compact CNN model...")
        model = build_regression_model(config.image_height, config.image_width)
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
    else:
        raise ValueError(f"Unsupported model_family '{config.model_family}'.")
    target_kind: Literal["value", "needle_unit_xy"] = (
        "needle_unit_xy"
        if config.model_family == "mobilenet_v2_direction"
        else "value"
    )
    train_ds = _build_tf_dataset(
        split.train_examples,
        config,
        training=True,
        target_kind=target_kind,
    )
    val_ds = _build_tf_dataset(
        split.val_examples,
        config,
        training=False,
        target_kind=target_kind,
    )
    test_ds = _build_tf_dataset(
        split.test_examples,
        config,
        training=False,
        target_kind=target_kind,
    )
    print("[TRAIN] TensorFlow datasets built.")
    model_name: str = getattr(model, "name", model.__class__.__name__)
    if hasattr(model, "count_params"):
        model_params: str = f"{int(model.count_params()):,}"
    else:
        model_params = "unknown"
    print(f"[TRAIN] Built model '{model_name}' with {model_params} parameters.")
    callbacks: list[keras.callbacks.Callback] = _make_training_callbacks()
    should_use_two_stage_mobilenet: bool = (
        config.model_family
        in {"mobilenet_v2", "mobilenet_v2_tiny", "mobilenet_v2_direction"}
        and config.mobilenet_pretrained
        and config.mobilenet_backbone_trainable
        and config.mobilenet_warmup_epochs > 0
        and config.epochs > 1
        and config.init_model_path is None
    )

    if should_use_two_stage_mobilenet:
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
        if config.model_family == "mobilenet_v2_direction":
            _compile_direction_model(
                model,
                learning_rate=config.learning_rate,
                spec=spec,
            )
        else:
            _compile_regression_model(model, learning_rate=config.learning_rate)
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
        if config.model_family == "mobilenet_v2_direction":
            _compile_direction_model(
                model,
                learning_rate=config.learning_rate,
                spec=spec,
            )
        else:
            _compile_regression_model(model, learning_rate=config.learning_rate)
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
        print("[TRAIN] Compact CNN stage: training end-to-end with callbacks.")
        if config.model_family == "mobilenet_v2_direction":
            _compile_direction_model(
                model,
                learning_rate=config.learning_rate,
                spec=spec,
            )
        else:
            _compile_regression_model(model, learning_rate=config.learning_rate)
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
