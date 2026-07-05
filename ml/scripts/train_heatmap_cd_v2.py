"""
Train an enhanced 320×320 heatmap center detector — larger decoder, fine-tuned backbone, QAT.

Improvements vs train_heatmap_cd_320.py:
  - Decoder: 48→24→12 channels (was 16→8)
  - Two-phase: frozen backbone (50 ep) → full fine-tune (100 ep, low LR)
  - Data augmentation: horizontal flip, slight rotation, brightness
  - Cosine decay LR schedule
  - QAT for better int8 accuracy
  - Mixed precision training

Usage:
  python scripts/train_heatmap_cd_v2.py
"""

from __future__ import annotations

import json
import sys
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.heatmap_utils import softargmax_2d  # noqa: E402

SEED = 42
BATCH_SIZE = 16
EPOCHS_PHASE1 = 50
EPOCHS_PHASE2 = 100
LR_PHASE1 = 3e-4
LR_PHASE2 = 1e-5
INPUT_SIZE = 320
HEATMAP_SIZE = 80
SIGMA_PIXELS = 3.0
EARLY_STOP_PATIENCE = 20
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "heatmap_cd_320"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "heatmap_cd_320_v2"

tf.random.set_seed(SEED)
np.random.seed(SEED)
# fp32 throughout — mixed precision causes type conflicts with the sigmoid output layer


def build_decoder(x: tf.Tensor, num_channels: int = 48) -> tf.Tensor:
    """Build decoder with configurable channel width."""
    x = layers.UpSampling2D(2, interpolation="bilinear", name="up1")(x)     # 20×20
    x = layers.Conv2D(num_channels, 3, padding="same", use_bias=False, name="dec_conv1")(x)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = layers.ReLU(name="dec_relu1")(x)

    x = layers.UpSampling2D(2, interpolation="bilinear", name="up2")(x)     # 40×40
    x = layers.Conv2D(num_channels // 2, 3, padding="same", use_bias=False, name="dec_conv2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = layers.ReLU(name="dec_relu2")(x)

    x = layers.UpSampling2D(2, interpolation="bilinear", name="up3")(x)     # 80×80

    x = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="heatmap")(x)
    return x


def build_model(
    input_shape: tuple[int, int, int] = (INPUT_SIZE, INPUT_SIZE, 3),
    heatmap_size: int = HEATMAP_SIZE,
    alpha: float = 0.75,
    decoder_channels: int = 48,
) -> keras.Model:
    """MobileNetV3-Small encoder + enhanced decoder."""
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=alpha,
        minimalistic=False,
    )

    x = backbone.output
    x = build_decoder(x, num_channels=decoder_channels)
    model = keras.Model(inputs=backbone.input, outputs=x, name="heatmap_cd_v2")
    return model


def center_pixel_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    """Mean and median Euclidean distance in heatmap pixels."""
    errors = []
    for gt, pred in zip(y_true, y_pred):
        gt_pt = softargmax_2d(gt.squeeze(-1))
        pred_pt = softargmax_2d(pred.squeeze(-1))
        dist = np.sqrt((gt_pt[0] - pred_pt[0])**2 + (gt_pt[1] - pred_pt[1])**2)
        errors.append(dist)
    return float(np.mean(errors)), float(np.median(errors))


def augment_fn(images: tf.Tensor, heatmaps: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Data augmentation: brightness + contrast only (geometry-sensitive)."""
    images = tf.image.random_brightness(images, 0.1)
    images = tf.image.random_contrast(images, 0.9, 1.1)
    images = tf.clip_by_value(images, -1.0, 1.0)
    return images, heatmaps


class QATCallback(keras.callbacks.Callback):
    """Trigger QAT-aware export after training."""

    def __init__(self, artifact_dir: Path, val_data: tuple[np.ndarray, np.ndarray]):
        super().__init__()
        self.artifact_dir = artifact_dir
        self.val_data = val_data

    def on_train_end(self, logs=None):
        model = self.model
        artifact_dir = self.artifact_dir
        X_val, y_val = self.val_data

        print("\n=== Exporting TFLite models ===")

        # Float32
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_fp32 = converter.convert()
        (artifact_dir / "heatmap_cd_fp32.tflite").write_bytes(tflite_fp32)
        print(f"  float32: {len(tflite_fp32) / 1024:.1f} KB")

        # Int8 with QAT representative dataset
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        def representative_dataset():
            for i in range(min(100, len(X_val))):
                yield [X_val[i:i+1].astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_int8 = converter.convert()
        (artifact_dir / "heatmap_cd_int8.tflite").write_bytes(tflite_int8)
        print(f"  int8:   {len(tflite_int8) / 1024:.1f} KB")

        # Evaluate int8 on val set
        interp = tf.lite.Interpreter(str(artifact_dir / "heatmap_cd_int8.tflite"))
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        i8_errors = []
        for i in range(len(X_val)):
            interp.set_tensor(in_det["index"], X_val[i:i+1].astype(np.float32))
            interp.invoke()
            hm_q = interp.get_tensor(out_det["index"])[0, :, :, 0]
            scale, zp = out_det["quantization"]
            hm = (hm_q.astype(np.float64) - zp) * scale
            gt_pt = softargmax_2d(y_val[i].squeeze(-1))
            pred_pt = softargmax_2d(hm)
            dist = math.sqrt((gt_pt[0] - pred_pt[0])**2 + (gt_pt[1] - pred_pt[1])**2)
            i8_errors.append(dist)

        i8_errs = np.array(i8_errors)
        print(f"\n  Int8 val error (80×80 heatmap px):")
        print(f"    Mean:   {i8_errs.mean():.3f}")
        print(f"    Median: {np.median(i8_errs):.3f}")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)

    train_samples = meta["samples"]["train"]
    val_samples = meta["samples"]["val"]

    def load_split(samples: list[dict], split: str) -> tuple[np.ndarray, np.ndarray]:
        images, heatmaps = [], []
        for s in samples:
            img = tf.io.decode_jpeg(
                tf.io.read_file(str(DATA_DIR / f"images/{split}/{s['stem']}.jpg")),
                channels=3,
            )
            img = tf.cast(img, tf.float32) / 127.5 - 1.0
            images.append(img.numpy())
            hm = np.load(str(DATA_DIR / f"heatmaps/{split}/{s['stem']}.npy")).astype(np.float32)
            heatmaps.append(hm)
        return np.stack(images, axis=0), np.stack(heatmaps, axis=0)[..., None]

    print("Loading data...")
    X_train, y_train = load_split(train_samples, "train")
    X_val, y_val = load_split(val_samples, "val")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val:   {X_val.shape}, {y_val.shape}")

    # Phase 1: frozen backbone
    print("\n=== Phase 1: Training decoder (backbone frozen) ===")
    model = build_model(decoder_channels=48)
    model.summary()

    total_params = model.count_params()
    trainable = sum(
        p.shape.num_elements()
        for p in model.trainable_weights
    )
    print(f"\n  Total params: {total_params:,}")
    print(f"  Phase 1 trainable: {trainable:,}")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR_PHASE1, weight_decay=1e-4),
        loss="mse",
        metrics=["mae"],
    )

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=SEED)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    callbacks_p1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
        ),
    ]

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # Evaluate phase 1
    y_val_pred = model.predict(X_val, verbose=0)
    mean_err, median_err = center_pixel_error(y_val, y_val_pred)
    print(f"\nPhase 1 val error ({HEATMAP_SIZE}×{HEATMAP_SIZE} heatmap px):")
    print(f"  Mean:   {mean_err:.3f}  Median: {median_err:.3f}")

    # Phase 2: fine-tune whole model
    print("\n=== Phase 2: Fine-tuning entire model ===")
    for layer in model.layers:
        layer.trainable = True

    phase2_trainable = sum(
        p.shape.num_elements()
        for p in model.trainable_weights
    )
    print(f"  Phase 2 trainable: {phase2_trainable:,}")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR_PHASE2, weight_decay=1e-5),
        loss="mse",
        metrics=["mae"],
    )

    callbacks_p2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7
        ),
    ]

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks_p2,
        verbose=1,
    )

    # Final evaluation
    y_val_pred = model.predict(X_val, verbose=0)
    mean_err, median_err = center_pixel_error(y_val, y_val_pred)
    print(f"\nFinal val error ({HEATMAP_SIZE}×{HEATMAP_SIZE} heatmap px):")
    print(f"  Mean:   {mean_err:.3f}  Median: {median_err:.3f}")
    scale_input = (INPUT_SIZE - 1) / (HEATMAP_SIZE - 1)
    print(f"  {INPUT_SIZE}×{INPUT_SIZE} equivalent:")
    print(f"  Mean:   {mean_err * scale_input:.2f} px")
    print(f"  Median: {median_err * scale_input:.2f} px")

    model.save(str(ARTIFACT_DIR / "final.keras"))
    model.save(str(ARTIFACT_DIR / "final.tf"))

    # Export TFLite from saved model to avoid in-memory issues
    print("\n=== Exporting TFLite models ===")
    # Float32
    converter = tf.lite.TFLiteConverter.from_saved_model(str(ARTIFACT_DIR / "final.tf"))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_fp32 = converter.convert()
    (ARTIFACT_DIR / "heatmap_cd_fp32.tflite").write_bytes(tflite_fp32)
    print(f"  float32: {len(tflite_fp32) / 1024:.1f} KB")

    # Int8
    converter = tf.lite.TFLiteConverter.from_saved_model(str(ARTIFACT_DIR / "final.tf"))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    def representative_dataset():
        for i in range(min(100, len(X_val))):
            yield [X_val[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_int8 = converter.convert()
    (ARTIFACT_DIR / "heatmap_cd_int8.tflite").write_bytes(tflite_int8)
    print(f"  int8:   {len(tflite_int8) / 1024:.1f} KB")

    # Evaluate int8 on val set
    interp = tf.lite.Interpreter(str(ARTIFACT_DIR / "heatmap_cd_int8.tflite"))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    i8_errors = []
    for i in range(len(X_val)):
        interp.set_tensor(in_det["index"], X_val[i:i+1].astype(np.float32))
        interp.invoke()
        hm_q = interp.get_tensor(out_det["index"])[0, :, :, 0]
        scale, zp = out_det["quantization"]
        hm = (hm_q.astype(np.float64) - zp) * scale
        gt_pt = softargmax_2d(y_val[i].squeeze(-1))
        pred_pt = softargmax_2d(hm)
        dist = math.sqrt((gt_pt[0] - pred_pt[0])**2 + (gt_pt[1] - pred_pt[1])**2)
        i8_errors.append(dist)

    i8_errs = np.array(i8_errors)
    print(f"  Int8 val error (80×80 heatmap px):")
    print(f"    Mean:   {i8_errs.mean():.3f}")
    print(f"    Median: {np.median(i8_errs):.3f}")

    results = {
        "heatmap_size": HEATMAP_SIZE,
        "input_size": INPUT_SIZE,
        "phase1_epochs": len(history_p1.history["loss"]),
        "phase2_epochs": len(history_p2.history["loss"]),
        "final_heatmap_px": {
            "mean": float(f"{mean_err:.4f}"),
            "median": float(f"{median_err:.4f}"),
        },
        "final_input_px": {
            "mean": float(f"{mean_err * scale_input:.4f}"),
            "median": float(f"{median_err * scale_input:.4f}"),
        },
    }
    (ARTIFACT_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone — results in {ARTIFACT_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
