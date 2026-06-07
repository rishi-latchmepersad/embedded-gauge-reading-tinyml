#!/usr/bin/env python3
"""Train a heatmap-based centre detector for the gauge-reading pipeline.

The model uses a **U-Net-style decoder** on top of a frozen MobileNetV2
(alpha=0.35) backbone to predict a single-channel sigmoid heatmap, then
``SpatialSoftArgmax2D`` extracts the (cx, cy) coordinate.

Config from environment variables (defaults in parentheses):
  RUN_TAG             — short label, e.g. "heatmap_v1"   (required)
  WARMUP_EPOCHS       — frozen-backbone epochs            (40)
  FINETUNE_EPOCHS     — unfreeze epochs (0 when SKIP_FINETUNE=1) (12)
  HEAD_UNITS          — unused in heatmap model           (128)
  HEAD_DROPOUT        — unused in heatmap model           (0.30)
  LR_WARMUP           — learning rate for warmup          (3e-4)
  LR_FINETUNE         — learning rate for fine-tune       (3e-5)
  CAPTURE_WEIGHT      — sample weight for "capture" src   (3.0)
  STRONG_AUG          — "1" for stronger photometric aug  (0)
  TRANSLATE_AUG       — "1" for random crop translation   (1)
  MAX_TRANSLATE_PX    — max pixel shift for translation   (10)
  SKIP_FINETUNE       — "1" to skip fine-tune phase       (1)
  HEATMAP_SIZE        — output heatmap spatial size       (56)
  HEATMAP_SIGMA       — Gaussian sigma for target hm (px) (1.5)
  HEATMAP_LOSS_WEIGHT — weight of auxiliary hm MSE loss   (0.1)
  TEMPERATURE         — softargmax temperature            (10.0)
  VERBOSE             — "0" silent, "2" per-epoch         (2)
  DATA_DIR            — path to dataset directory
  BACKBONE_ALPHA      — MobileNetV2 width multiplier      (0.35)

Example:
  RUN_TAG=heatmap_v1 WARMUP_EPOCHS=40 SKIP_FINETUNE=1 DATA_DIR=... \\
    poetry run python scripts/train_center_heatmap.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

# Force the dataset pipeline to stay on CPU so image decoding / resizing
# does not compete with the model for the limited GPU memory (2.2 GB).
_CPU = tf.device("/CPU:0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedded_gauge_reading_tinyml.training import (
    _load_crop_with_weight_maybe_board_style,
)
from embedded_gauge_reading_tinyml.models import (
    SpatialSoftArgmax2D,
    build_mobilenetv2_heatmap_center_model,
)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CD_CROP_W = 155
CD_CROP_H = 123
MAX_TRANSLATE_PX = 10
BATCH_SIZE = 8
REVIEW_WEIGHT = 0.8

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "training"
DEPLOY_DIR = PROJECT_ROOT / "artifacts" / "deployment" / "center_model_board_mimic_int8"


def _log(message: str) -> None:
    print(message, flush=True)


def _env(key: str, default):
    val = os.environ.get(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


RUN_TAG = _env("RUN_TAG", None)

DATA_OVERRIDE = _env("DATA_DIR", None)
if DATA_OVERRIDE:
    DATA_DIR = Path(DATA_OVERRIDE)
else:
    DATA_DIR = PROJECT_ROOT / "data" / "center_training_board_mimic"
METADATA_PATH = DATA_DIR / "metadata.json"
if RUN_TAG is None:
    raise ValueError("RUN_TAG is required")

WARMUP_EPOCHS = _env("WARMUP_EPOCHS", 40)
FINETUNE_EPOCHS = _env("FINETUNE_EPOCHS", 12)
HEAD_UNITS = _env("HEAD_UNITS", 128)
HEAD_DROPOUT = _env("HEAD_DROPOUT", 0.30)
LR_WARMUP = _env("LR_WARMUP", 3e-4)
LR_FINETUNE = _env("LR_FINETUNE", 3e-5)
CAPTURE_WEIGHT = _env("CAPTURE_WEIGHT", 3.0)
PSEUDO_WEIGHT = _env("PSEUDO_WEIGHT", 0.5)
STRONG_AUG = _env("STRONG_AUG", False)
TRANSLATE_AUG = _env("TRANSLATE_AUG", True)
MAX_TRANSLATE_PX = _env("MAX_TRANSLATE_PX", 10)
BOARD_STYLE_PROB = _env("BOARD_STYLE_PROB", 0.5)
UNFREEZE_LAST_N = _env("UNFREEZE_LAST_N", 0)
SKIP_FINETUNE = _env("SKIP_FINETUNE", True)
HEATMAP_SIZE = _env("HEATMAP_SIZE", 56)
HEATMAP_SIGMA = _env("HEATMAP_SIGMA", 1.5)
HEATMAP_LOSS_WEIGHT = _env("HEATMAP_LOSS_WEIGHT", 0.1)
TEMPERATURE = _env("TEMPERATURE", 10.0)
VERBOSE = _env("VERBOSE", 2)
BACKBONE_ALPHA = _env("BACKBONE_ALPHA", 0.35)
EPOCHS_TOTAL = WARMUP_EPOCHS + (0 if SKIP_FINETUNE else FINETUNE_EPOCHS)

RUN_NAME = f"center_heatmap_{RUN_TAG}_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR = ARTIFACTS_DIR / RUN_NAME
TFLITE_PATH = DEPLOY_DIR / f"model_int8_{RUN_TAG}.tflite"


def load_metadata(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_label(
    entry: dict,
) -> tuple[str, np.ndarray, float, np.ndarray, np.ndarray]:
    source_path = str(REPO_ROOT / entry["source_path"])
    center_xy = np.array(
        [float(entry["center_x_norm"]), float(entry["center_y_norm"])],
        dtype=np.float32,
    )
    crop_x_min = float(entry["crop_x_min"])
    crop_y_min = float(entry["crop_y_min"])
    crop_width = float(entry["crop_width"])
    crop_height = float(entry["crop_height"])
    crop_box_xyxy = np.array(
        [crop_x_min, crop_y_min, crop_x_min + crop_width, crop_y_min + crop_height],
        dtype=np.float32,
    )
    full_center_xy = np.array(
        [
            float(entry.get("full_frame_center_x", 112.0)),
            float(entry.get("full_frame_center_y", 112.0)),
        ],
        dtype=np.float32,
    )
    source_kind = str(entry["source_kind"])
    quality_flag = str(entry["quality_flag"])
    if source_kind == "capture":
        weight = CAPTURE_WEIGHT
    elif source_kind == "pseudo":
        weight = PSEUDO_WEIGHT
    else:
        weight = 1.0
    if quality_flag == "review":
        weight *= REVIEW_WEIGHT
    return source_path, center_xy, float(weight), crop_box_xyxy, full_center_xy


def _make_gaussian_heatmap_tf(
    cx_norm: tf.Tensor,
    cy_norm: tf.Tensor,
    heatmap_size: int = 56,
    sigma: float = 1.5,
) -> tf.Tensor:
    """Create a 2-D Gaussian heatmap centred at normalised (cx, cy)."""
    cx = cx_norm * tf.cast(heatmap_size - 1, tf.float32)
    cy = cy_norm * tf.cast(heatmap_size - 1, tf.float32)

    x = tf.range(heatmap_size, dtype=tf.float32)
    y = tf.range(heatmap_size, dtype=tf.float32)
    xx, yy = tf.meshgrid(x, y, indexing="ij")

    heatmap = tf.exp(
        -((tf.cast(xx, tf.float32) - cx) ** 2 + (tf.cast(yy, tf.float32) - cy) ** 2)
        / (2.0 * sigma ** 2)
    )
    return heatmap[..., tf.newaxis]


def _translate_crop(
    crop_box_xyxy: tf.Tensor,
    label: tf.Tensor,
    full_center_xy: tf.Tensor,
    max_px: int = 10,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Randomly shift the CD crop box and recompute the normalised label."""
    dx = tf.random.uniform([], -max_px, max_px + 1, dtype=tf.int32)
    dy = tf.random.uniform([], -max_px, max_px + 1, dtype=tf.int32)

    crop_x = tf.cast(crop_box_xyxy[0], tf.int32)
    crop_y = tf.cast(crop_box_xyxy[1], tf.int32)
    new_x = tf.clip_by_value(crop_x + dx, 0, IMAGE_WIDTH - CD_CROP_W)
    new_y = tf.clip_by_value(crop_y + dy, 0, IMAGE_HEIGHT - CD_CROP_H)

    new_box = tf.stack([
        tf.cast(new_x, tf.float32),
        tf.cast(new_y, tf.float32),
        tf.cast(new_x + CD_CROP_W, tf.float32),
        tf.cast(new_y + CD_CROP_H, tf.float32),
    ])

    scale = tf.minimum(
        tf.cast(IMAGE_WIDTH, tf.float32) / tf.cast(CD_CROP_W, tf.float32),
        tf.cast(IMAGE_HEIGHT, tf.float32) / tf.cast(CD_CROP_H, tf.float32),
    )
    rw = tf.maximum(
        tf.cast(tf.round(tf.cast(CD_CROP_W, tf.float32) * scale), tf.int32), 1
    )
    rh = tf.maximum(
        tf.cast(tf.round(tf.cast(CD_CROP_H, tf.float32) * scale), tf.int32), 1
    )
    off_x = (IMAGE_WIDTH - rw) // 2
    off_y = (IMAGE_HEIGHT - rh) // 2

    full_cx = full_center_xy[0]
    full_cy = full_center_xy[1]
    crop_scale_x = tf.cast(rw, tf.float32) / tf.cast(CD_CROP_W, tf.float32)
    crop_scale_y = tf.cast(rh, tf.float32) / tf.cast(CD_CROP_H, tf.float32)
    padded_x = (
        full_cx - tf.cast(new_x, tf.float32)
    ) * crop_scale_x + tf.cast(off_x, tf.float32)
    padded_y = (
        full_cy - tf.cast(new_y, tf.float32)
    ) * crop_scale_y + tf.cast(off_y, tf.float32)
    new_label = tf.stack([
        padded_x / tf.cast(IMAGE_WIDTH, tf.float32),
        padded_y / tf.cast(IMAGE_HEIGHT, tf.float32),
    ])
    new_label = tf.clip_by_value(new_label, 0.0, 1.0)
    return new_box, new_label


def _augment(
    image: tf.Tensor, label: tf.Tensor, strong: bool
) -> tuple[tf.Tensor, tf.Tensor]:
    if strong:
        image = tf.image.random_brightness(image, max_delta=0.30)
        image = tf.image.random_contrast(image, lower=0.50, upper=1.50)
        image = tf.image.random_saturation(image, lower=0.70, upper=1.30)
        image = tf.image.random_hue(image, max_delta=0.08)
        angle_rad = tf.random.uniform([], -0.070, 0.070)
    else:
        image = tf.image.random_brightness(image, max_delta=0.18)
        image = tf.image.random_contrast(image, lower=0.70, upper=1.30)
        image = tf.image.random_saturation(image, lower=0.80, upper=1.20)
        image = tf.image.random_hue(image, max_delta=0.05)
        angle_rad = tf.random.uniform([], -0.035, 0.035)

    cos_theta = tf.cos(angle_rad)
    sin_theta = tf.sin(angle_rad)
    center_x = tf.constant(IMAGE_WIDTH / 2.0, tf.float32)
    center_y = tf.constant(IMAGE_HEIGHT / 2.0, tf.float32)
    translate_x = (1.0 - cos_theta) * center_x + sin_theta * center_y
    translate_y = -sin_theta * center_x + (1.0 - cos_theta) * center_y
    transform = [
        cos_theta,
        -sin_theta,
        translate_x,
        sin_theta,
        cos_theta,
        translate_y,
        0.0,
        0.0,
    ]
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=[IMAGE_HEIGHT, IMAGE_WIDTH],
        fill_value=0.0,
        interpolation="BILINEAR",
    )[0]

    label_x = label[0] * tf.constant(IMAGE_WIDTH, tf.float32) - center_x
    label_y = label[1] * tf.constant(IMAGE_HEIGHT, tf.float32) - center_y
    rot_x = cos_theta * label_x - sin_theta * label_y
    rot_y = sin_theta * label_x + cos_theta * label_y
    label = tf.stack([
        (rot_x + center_x) / tf.constant(IMAGE_WIDTH, tf.float32),
        (rot_y + center_y) / tf.constant(IMAGE_HEIGHT, tf.float32),
    ])
    return tf.clip_by_value(image, 0.0, 255.0), label


def _augment_mild(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return _augment(image, label, strong=False)


def _augment_strong(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return _augment(image, label, strong=True)


def build_dataset(
    entries: list[dict],
    *,
    augment: bool = False,
    shuffle: bool = False,
    use_sample_weights: bool = False,
    strong_aug: bool = False,
) -> tf.data.Dataset:
    paths: list[str] = []
    labels: list[np.ndarray] = []
    weights: list[float] = []
    crop_boxes: list[np.ndarray] = []
    full_centers: list[np.ndarray] = []
    for entry in entries:
        path, label, weight, crop_box_xyxy, full_center_xy = parse_label(entry)
        paths.append(path)
        labels.append(label)
        weights.append(weight)
        crop_boxes.append(crop_box_xyxy)
        full_centers.append(full_center_xy)

    with _CPU:
        ds = tf.data.Dataset.from_tensor_slices((
            np.asarray(paths, dtype=np.str_),
            np.asarray(labels, dtype=np.float32),
            np.asarray(weights, dtype=np.float32),
            np.asarray(crop_boxes, dtype=np.float32),
            np.asarray(full_centers, dtype=np.float32),
        ))

    def _load(
        path: tf.Tensor,
        label: tf.Tensor,
        weight: tf.Tensor,
        crop_box_xyxy: tf.Tensor,
        full_center_xy: tf.Tensor,
    ):
        if augment and TRANSLATE_AUG:
            crop_box_xyxy, label = _translate_crop(
                crop_box_xyxy,
                label,
                full_center_xy,
                max_px=MAX_TRANSLATE_PX,
            )
        image, label, weight = _load_crop_with_weight_maybe_board_style(
            path,
            label,
            crop_box_xyxy,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            weight,
            BOARD_STYLE_PROB,
        )
        image = tf.cast(image * 255.0, tf.float32)
        # Generate heatmap target from the (cx, cy) label
        hm = _make_gaussian_heatmap_tf(
            label[0], label[1],
            heatmap_size=HEATMAP_SIZE,
            sigma=HEATMAP_SIGMA,
        )
        targets = {
            "center_xy": label,
            "heatmap": hm,
        }
        if use_sample_weights:
            return image, targets, weight
        return image, targets

    with _CPU:
        # Use sequential loading to avoid GPU memory buildup from parallel
        # decode threads competing with the training loop.
        ds = ds.map(_load, num_parallel_calls=4)
        if augment:
            aug_fn = _augment_strong if strong_aug else _augment_mild

            # Extract center_xy, augment, then rebuild dict targets.
            # The extra lambda layers keep the graph readable for TF Autograph.
            def _augment_with_rebuild(image, targets):
                img_aug, label_aug = aug_fn(image, targets["center_xy"])
                hm_aug = _make_gaussian_heatmap_tf(
                    label_aug[0],
                    label_aug[1],
                    heatmap_size=HEATMAP_SIZE,
                    sigma=HEATMAP_SIGMA,
                )
                return img_aug, {"center_xy": label_aug, "heatmap": hm_aug}

            ds = ds.map(_augment_with_rebuild, num_parallel_calls=4)
        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(max(len(entries), 1), 512),
                reshuffle_each_iteration=True,
            )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def _evaluate_by_source_kind(
    predictions: np.ndarray,
    labels: np.ndarray,
    entries: list[dict],
) -> dict[str, float]:
    errors_px = np.abs(predictions - labels) * float(IMAGE_WIDTH)
    summary: dict[str, float] = {}
    for source_kind in ("capture", "pxl"):
        mask = np.array(
            [str(entry["source_kind"]) == source_kind for entry in entries],
            dtype=bool,
        )
        if not np.any(mask):
            continue
        summary[f"{source_kind}_count"] = float(mask.sum())
        summary[f"{source_kind}_center_mae_px"] = float(
            np.mean(errors_px[mask])
        )
    summary["overall_center_mae_px"] = float(np.mean(errors_px))
    summary["overall_center_cx_mae_px"] = float(np.mean(errors_px[:, 0]))
    summary["overall_center_cy_mae_px"] = float(np.mean(errors_px[:, 1]))
    return summary


def export_tflite(model: keras.Model, path: Path) -> dict:
    # Export only the center_xy branch for firmware deployment
    export_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer("center_xy").output,
        name="heatmap_cd_export",
    )

    def _representative_dataset():
        for entry in train_entries[: min(len(train_entries), 200)]:
            image_path = DATA_DIR / entry["image_path"]
            image = tf.io.read_file(str(image_path))
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
            image = tf.cast(image, tf.float32)
            yield [np.expand_dims(image.numpy(), axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    return {
        "input_scale": float(in_det["quantization"][0]),
        "input_zero_point": int(in_det["quantization"][1]),
        "output_scale": float(out_det["quantization"][0]),
        "output_zero_point": int(out_det["quantization"][1]),
        "size_bytes": len(tflite_model),
    }


def main() -> None:
    # Disable XLA — the tf2xla compiler fails to resolve the element-wise
    # shape broadcast between the (B, H, W, 1) heatmap tensor and the
    # auxiliary MSE target when the model uses dict outputs.
    tf.config.optimizer.set_jit(False)

    # Enable GPU memory growth to guard against OOM across epochs.
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(DEPLOY_DIR, exist_ok=True)

    global train_entries, test_entries
    _log(f"=== HEATMAP CD: {RUN_TAG} ===")
    _log(
        f"  warmup={WARMUP_EPOCHS} finetune={0 if SKIP_FINETUNE else FINETUNE_EPOCHS}"
    )
    _log(f"  heatmap_size={HEATMAP_SIZE} sigma={HEATMAP_SIGMA}")
    _log(f"  hm_loss_weight={HEATMAP_LOSS_WEIGHT} temperature={TEMPERATURE}")
    _log(f"  lr_warmup={LR_WARMUP} capture_weight={CAPTURE_WEIGHT}")
    _log(f"  strong_aug={STRONG_AUG} skip_finetune={SKIP_FINETUNE}")
    _log(f"  backbone_alpha={BACKBONE_ALPHA}")

    all_entries = load_metadata(METADATA_PATH)
    train_entries = [e for e in all_entries if str(e["split"]) == "train"]
    val_entries = [e for e in all_entries if str(e["split"]) == "val"]
    test_entries = [e for e in all_entries if str(e["split"]) == "test"]

    _log(
        f"  train={len(train_entries)} val={len(val_entries)} "
        f"test={len(test_entries)}"
    )

    # NOTE: use_sample_weights=False because Keras's per-sample weight
    # broadcasting fails for the (B, H, W, 1) heatmap output when
    # combined with the (B,) sample-weight vector.  The capture-weight
    # oversampling effect is approximated by higher CAPTURE_WEIGHT anyway.
    train_ds = build_dataset(
        train_entries,
        augment=True,
        shuffle=True,
        use_sample_weights=False,
        strong_aug=STRONG_AUG,
    )
    val_ds = build_dataset(val_entries, augment=False, use_sample_weights=False)
    test_ds = build_dataset(test_entries, augment=False, use_sample_weights=False)

    _log("Building heatmap model...")
    model = build_mobilenetv2_heatmap_center_model(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        pretrained=True,
        backbone_trainable=False,
        alpha=BACKBONE_ALPHA,
        heatmap_size=HEATMAP_SIZE,
        temperature=TEMPERATURE,
    )
    backbone = getattr(model, "_mobilenet_backbone")
    model.summary(print_fn=lambda line: _log(line))

    # --- Warmup (frozen backbone) ---
    _log("--- Phase 1: warmup (backbone frozen) ---")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_WARMUP),
        loss={
            "center_xy": keras.losses.MeanSquaredError(),
            "heatmap": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "center_xy": 1.0,
            "heatmap": HEATMAP_LOSS_WEIGHT,
        },
        metrics={
            "center_xy": [keras.metrics.MeanAbsoluteError(name="mae")],
            "heatmap": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
        jit_compile=False,
    )

    warmup_callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(RUN_DIR / "best_model.keras"),
            monitor="val_center_xy_mae",
            mode="min",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_center_xy_mae",
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
        keras.callbacks.CSVLogger(str(RUN_DIR / "training_log.csv")),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=warmup_callbacks,
        verbose=VERBOSE,
    )

    if not SKIP_FINETUNE:
        _log("--- Phase 2: fine-tune ---")
        backbone.trainable = True
        if UNFREEZE_LAST_N > 0:
            from embedded_gauge_reading_tinyml.training import (
                _configure_mobilenet_backbone_trainability,
            )

            def _unfreeze_last_n(bb: keras.Model, n: int) -> None:
                layers_w = [l for l in bb.layers if len(l.weights) > 0]
                n = min(n, len(layers_w))
                for layer in layers_w[:-n]:
                    layer.trainable = False
                for layer in layers_w[-n:]:
                    layer.trainable = True

            _unfreeze_last_n(backbone, UNFREEZE_LAST_N)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LR_FINETUNE),
            loss={
                "center_xy": keras.losses.MeanSquaredError(),
                "heatmap": keras.losses.MeanSquaredError(),
            },
            loss_weights={
                "center_xy": 1.0,
                "heatmap": HEATMAP_LOSS_WEIGHT,
            },
            metrics={
                "center_xy": [keras.metrics.MeanAbsoluteError(name="mae")],
                "heatmap": [keras.metrics.MeanAbsoluteError(name="mae")],
            },
            jit_compile=False,
        )
        ft_callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(RUN_DIR / "best_model.keras"),
                monitor="val_center_xy_mae",
                mode="min",
                save_best_only=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_center_xy_mae",
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=0,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_center_xy_mae",
                mode="min",
                patience=8,
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.CSVLogger(str(RUN_DIR / "training_log_finetune.csv")),
        ]
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=WARMUP_EPOCHS,
            epochs=WARMUP_EPOCHS + FINETUNE_EPOCHS,
            callbacks=ft_callbacks,
            verbose=VERBOSE,
        )

    # --- Evaluate holdout ---
    _log("Evaluating holdout...")
    best_model = keras.models.load_model(
        str(RUN_DIR / "best_model.keras"),
        custom_objects={
            "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
        },
    )

    test_results = best_model.evaluate(test_ds, verbose=0)
    _log(f"  test metrics: {test_results}")

    # Get predictions (center_xy only)
    test_preds_dict = best_model.predict(test_ds, verbose=0)
    test_predictions = test_preds_dict["center_xy"]

    test_labels_np = np.stack(
        [t["center_xy"].numpy() for _, t in test_ds.unbatch()], axis=0
    )
    holdout_summary = _evaluate_by_source_kind(
        test_predictions, test_labels_np, test_entries
    )
    for key, value in holdout_summary.items():
        _log(f"  {key}: {value:.4f}")

    best_model.save(str(RUN_DIR / "final_model.keras"))

    if holdout_summary.get("capture_count", 0) == 0:
        center_xy_idx = (
            best_model.output_names.index("center_xy")
            if "center_xy" in best_model.output_names
            else 0
        )
        test_mae_px = test_results[center_xy_idx] * IMAGE_WIDTH
    else:
        test_mae_px = holdout_summary["overall_center_mae_px"]

    summary = {
        "run_tag": RUN_TAG,
        "run_name": RUN_NAME,
        "run_dir": str(RUN_DIR),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "test_samples": len(test_entries),
        "warmup_epochs": WARMUP_EPOCHS,
        "finetune_epochs": 0 if SKIP_FINETUNE else FINETUNE_EPOCHS,
        "heatmap_size": HEATMAP_SIZE,
        "heatmap_sigma": HEATMAP_SIGMA,
        "heatmap_loss_weight": HEATMAP_LOSS_WEIGHT,
        "temperature": TEMPERATURE,
        "learning_rate_warmup": LR_WARMUP,
        "learning_rate_finetune": LR_FINETUNE,
        "capture_weight": CAPTURE_WEIGHT,
        "strong_aug": bool(STRONG_AUG),
        "unfreeze_last_n": UNFREEZE_LAST_N,
        "skip_finetune": bool(SKIP_FINETUNE),
        "backbone_alpha": BACKBONE_ALPHA,
        "test_mae_px": float(test_mae_px),
        **holdout_summary,
    }
    with (RUN_DIR / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # --- Export int8 TFLite ---
    _log("Exporting int8 TFLite...")
    qinfo = export_tflite(best_model, TFLITE_PATH)
    _log(
        f"  size={qinfo['size_bytes']} input_scale={qinfo['input_scale']} "
        f"input_zp={qinfo['input_zero_point']} output_scale={qinfo['output_scale']} "
        f"output_zp={qinfo['output_zero_point']}"
    )

    if (
        qinfo["input_scale"] != 1.0
        or qinfo["input_zero_point"] != -128
        or qinfo["output_scale"] != 0.00390625
        or qinfo["output_zero_point"] != -128
    ):
        _log(
            "WARNING: Quantization params differ from firmware expected "
            "(input scale=1.0 zp=-128, output scale=0.00390625 zp=-128)"
        )

    deploy_metadata = {
        "source_model": str(RUN_DIR / "best_model.keras"),
        "tflite_path": str(TFLITE_PATH.relative_to(PROJECT_ROOT)),
        "run_tag": RUN_TAG,
        "input_shape": [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "output_shape": [1, 2],
        **qinfo,
        "test_mae_px": float(test_mae_px),
        "capture_mae_px": holdout_summary.get("capture_center_mae_px", None),
        "pxl_mae_px": holdout_summary.get("pxl_center_mae_px", None),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "test_samples": len(test_entries),
    }
    meta_path = DEPLOY_DIR / f"metadata_{RUN_TAG}.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(deploy_metadata, handle, indent=2)

    _log(f"Saved: {TFLITE_PATH}")
    _log(f"Saved: {meta_path}")
    _log("Done.")


if __name__ == "__main__":
    main()
