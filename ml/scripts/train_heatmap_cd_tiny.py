"""
Train a tiny no-BN CNN for heatmap center detection.
Architecture designed for clean TFLite export and small dataset (fewer params).

Encoder: 4× stride-2 conv (320→20), 20→10 via ave pool
Decoder: 3× upsample + conv (10→20→40→80)
No BatchNorm — avoids TFLite folding issues.
~65K params — resistant to overfitting on 282 samples.

Usage:
  python scripts/train_heatmap_cd_tiny.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embedded_gauge_reading_tinyml.heatmap_utils import softargmax_2d  # noqa: E402

SEED = 42
BATCH_SIZE = 16
EPOCHS = 200
LR = 2e-3
INPUT_SIZE = 320
HEATMAP_SIZE = 80
EARLY_STOP_PATIENCE = 30
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "heatmap_cd_320"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "heatmap_cd_tiny"

tf.random.set_seed(SEED)
np.random.seed(SEED)


def conv3x3(x, channels: int, stride: int = 1, name: str = "conv"):
    x = layers.Conv2D(channels, 3, strides=stride, padding="same", use_bias=True, name=name)(x)
    x = layers.LeakyReLU(0.1, name=f"{name}_lrelu")(x)
    return x


def build_tiny_cnn(
    input_shape: tuple[int, int, int] = (INPUT_SIZE, INPUT_SIZE, 3),
) -> keras.Model:
    """Tiny no-BN CNN: encoder 320→20, pool to 10, decoder 10→80 heatmap."""

    inp = layers.Input(shape=input_shape, name="input")

    # Encoder (stride-2 conv for downsampling)
    x = conv3x3(inp, 16, stride=2, name="enc1")    # 160×160
    x = conv3x3(x, 24, stride=2, name="enc2")       # 80×80
    x = conv3x3(x, 32, stride=2, name="enc3")       # 40×40
    x = conv3x3(x, 48, stride=2, name="enc4")       # 20×20

    # Bottleneck
    x = layers.AveragePooling2D(2, name="bottleneck_pool")(x)  # 10×10

    # Decoder (upsample + conv)
    x = layers.UpSampling2D(2, interpolation="bilinear", name="up1")(x)  # 20×20
    x = conv3x3(x, 48, name="dec1")

    x = layers.UpSampling2D(2, interpolation="bilinear", name="up2")(x)  # 40×40
    x = conv3x3(x, 32, name="dec2")

    x = layers.UpSampling2D(2, interpolation="bilinear", name="up3")(x)  # 80×80
    x = conv3x3(x, 24, name="dec3")

    # Output heatmap
    x = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="heatmap")(x)

    model = keras.Model(inputs=inp, outputs=x, name="heatmap_cd_tiny")
    return model


def center_pixel_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    errors = []
    for gt, pred in zip(y_true, y_pred):
        gt_pt = softargmax_2d(gt.squeeze(-1))
        pred_pt = softargmax_2d(pred.squeeze(-1))
        dist = np.sqrt((gt_pt[0] - pred_pt[0])**2 + (gt_pt[1] - pred_pt[1])**2)
        errors.append(dist)
    return float(np.mean(errors)), float(np.median(errors))


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

    model = build_tiny_cnn()
    model.summary()

    total_params = model.count_params()
    print(f"\n  Total params: {total_params:,}")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4),
        loss="mse",
        metrics=["mae"],
    )

    # Data augmentation
    def augment_fn(images, heatmaps):
        images = tf.image.random_brightness(images, 0.1)
        images = tf.image.random_contrast(images, 0.9, 1.1)
        images = tf.clip_by_value(images, -1.0, 1.0)
        return images, heatmaps

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=SEED)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            str(ARTIFACT_DIR / "best.weights.h5"),
            monitor="val_loss", save_best_only=True, save_weights_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Restore best weights
    model.load_weights(str(ARTIFACT_DIR / "best.weights.h5"))

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

    # Export TFLite
    print("\n=== Exporting TFLite models ===")

    # Float32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_fp32 = converter.convert()
    (ARTIFACT_DIR / "heatmap_cd_fp32.tflite").write_bytes(tflite_fp32)
    print(f"  float32: {len(tflite_fp32) / 1024:.1f} KB")

    # Int8
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
    print(f"\n  Int8 val error (80×80 heatmap px):")
    print(f"    Mean:   {i8_errs.mean():.3f}")
    print(f"    Median: {np.median(i8_errs):.3f}")
    print(f"  {INPUT_SIZE}×{INPUT_SIZE} equivalent:")
    print(f"    Mean:   {i8_errs.mean() * scale_input:.2f} px")
    print(f"    Median: {np.median(i8_errs) * scale_input:.2f} px")

    # Save results
    results = {
        "heatmap_size": HEATMAP_SIZE,
        "input_size": INPUT_SIZE,
        "total_params": total_params,
        "epochs_trained": len(history.history["loss"]),
        "keras_val_heatmap_px": {
            "mean": float(f"{mean_err:.4f}"),
            "median": float(f"{median_err:.4f}"),
        },
        "int8_val_heatmap_px": {
            "mean": float(f"{i8_errs.mean():.4f}"),
            "median": float(f"{np.median(i8_errs):.4f}"),
        },
        "tflite_fp32_kb": round(len(tflite_fp32) / 1024, 1),
        "tflite_int8_kb": round(len(tflite_int8) / 1024, 1),
    }
    (ARTIFACT_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone — results in {ARTIFACT_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
