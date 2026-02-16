"""Training pipeline for gauge-value regression with dial ROI cropping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


# Resolve ML/data paths from this file so training works from any cwd.
ML_ROOT: Path = Path(__file__).resolve().parents[2]
LABELLED_DIR: Path = ML_ROOT / "data" / "labelled"
RAW_DIR: Path = ML_ROOT / "data" / "raw"


@dataclass(frozen=True)
class TrainConfig:
    """Config values for reproducible training runs."""

    gauge_id: str = "littlegood_home_temp_gauge_c"
    image_height: int = 224
    image_width: int = 224
    batch_size: int = 8
    epochs: int = 40
    learning_rate: float = 1e-3
    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    strict_labels: bool = True
    crop_pad_ratio: float = 0.15
    augment_training: bool = True


@dataclass(frozen=True)
class TrainingExample:
    """One training row with image path, scalar target, and dial crop box."""

    image_path: str
    value: float
    crop_box_xyxy: tuple[float, float, float, float]


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

        # strict_labels is still respected for value computation behavior.
        value: float = needle_value(sample, spec, strict=strict_labels)
        crop_box: tuple[float, float, float, float] = _compute_crop_box(
            sample, crop_pad_ratio
        )
        examples.append(
            TrainingExample(
                image_path=str(sample.image_path),
                value=value,
                crop_box_xyxy=crop_box,
            )
        )

    return examples, dropped_out_of_sweep


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
    image: tf.Tensor = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.ensure_shape(image, [None, None, 3])

    image = _crop_image_with_xyxy(image, crop_box_xyxy)
    image = tf.image.resize(image, [image_height, image_width])

    image = tf.cast(image, tf.float32) / 255.0
    target: tf.Tensor = tf.cast(value, tf.float32)
    return image, target


def _augment_image(image: tf.Tensor) -> tf.Tensor:
    """Apply light photometric augmentation that preserves gauge geometry."""
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _build_tf_dataset(
    examples: list[TrainingExample],
    config: TrainConfig,
    *,
    training: bool,
) -> tf.data.Dataset:
    """Create a tf.data pipeline for efficient training/eval input."""
    paths: np.ndarray = np.array([e.image_path for e in examples], dtype=str)
    values: np.ndarray = np.array([e.value for e in examples], dtype=np.float32)
    boxes: np.ndarray = np.array([e.crop_box_xyxy for e in examples], dtype=np.float32)

    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
        (paths, values, boxes)
    )

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


def build_regression_model(image_height: int, image_width: int) -> keras.Model:
    """Build a compact CNN regressor for dial images."""
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(1, name="gauge_value")(x)

    return keras.Model(inputs=inputs, outputs=output, name="gauge_value_regressor")


def _compute_mean_baseline_mae(
    train_examples: list[TrainingExample],
    test_examples: list[TrainingExample],
) -> float:
    """Compute MAE of a trivial baseline that predicts train-mean value."""
    train_values: np.ndarray = np.array(
        [e.value for e in train_examples], dtype=np.float32
    )
    test_values: np.ndarray = np.array(
        [e.value for e in test_examples], dtype=np.float32
    )

    mean_pred: float = float(np.mean(train_values))
    baseline_mae: float = float(np.mean(np.abs(test_values - mean_pred)))
    return baseline_mae


def train(config: TrainConfig) -> TrainingResult:
    """Run one full training cycle and return model + metrics."""
    _validate_split_config(config)

    keras.utils.set_random_seed(config.seed)
    np.random.seed(config.seed)

    specs: dict[str, GaugeSpec] = load_gauge_specs()
    if config.gauge_id not in specs:
        raise ValueError(
            f"Unknown gauge_id '{config.gauge_id}'. Available: {list(specs)}"
        )
    spec: GaugeSpec = specs[config.gauge_id]

    samples: list[Sample] = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
    if not samples:
        raise ValueError(
            "No samples found. Check labelled/raw paths and annotation zips."
        )

    # This summary is still useful for visibility into raw annotation quality.
    label_summary: LabelSummary = summarize_label_sweep(samples, spec)

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

    split: DatasetSplit = _split_examples(examples, config)

    train_ds = _build_tf_dataset(split.train_examples, config, training=True)
    val_ds = _build_tf_dataset(split.val_examples, config, training=False)
    test_ds = _build_tf_dataset(split.test_examples, config, training=False)

    model: keras.Model = build_regression_model(config.image_height, config.image_width)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.Huber(delta=2.0),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    history: keras.callbacks.History = model.fit(
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
