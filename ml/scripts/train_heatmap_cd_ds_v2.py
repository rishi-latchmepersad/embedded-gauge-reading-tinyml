"""
Train a 320×320 heatmap center detector with scaled-up DS-CNN + skip connections.

Architecture:
  Encoder: Conv2D → 3× DSConv (stride-2) blocks, 32→64→128→256 channels
  Decoder: UpSampling2D + DSConv blocks with skip connections from encoder
  ~92K params — trains end-to-end on 602 samples

Usage:
  python scripts/train_heatmap_cd_ds_v2.py
"""

from __future__ import annotations

import json
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
LEARNING_RATE = 1e-3
INPUT_SIZE = 320
HEATMAP_SIZE = 80
EARLY_STOP_PATIENCE = 30
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "heatmap_cd_320"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "heatmap_cd_ds_v2"

tf.random.set_seed(SEED)
np.random.seed(SEED)


def ds_conv_block(
    x: tf.Tensor, filters: int, stride: int = 1, name: str = ""
) -> tf.Tensor:
    """Depthwise-separable conv block: DW Conv(3×3) → BN → ReLU → PW Conv(1×1) → BN → ReLU."""
    shortcut = x
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    x = layers.Conv2D(filters, 1, use_bias=False, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)
    return x


def build_model(
    input_shape: tuple[int, int, int] = (INPUT_SIZE, INPUT_SIZE, 3),
    heatmap_size: int = HEATMAP_SIZE,
) -> keras.Model:
    """Scaled-up depthwise-separable CNN with skip connections.

    Encoder:  320→160→80→40→20  (4 stages, stride-2, channels 32→64→128→256)
    Skip connections: 80×80×64 and 40×40×128 from encoder feed into decoder
    Decoder:  20↑→40(concat skip2)→80(concat skip1)→80 heatmap
    ~92K params total
    """
    inputs = layers.Input(shape=input_shape, name="input")
    skips: dict[str, tf.Tensor] = {}

    # Encoder
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="enc_conv")(inputs)
    x = layers.BatchNormalization(name="enc_conv_bn")(x)
    x = layers.ReLU(name="enc_conv_relu")(x)  # 160×160×32

    x = ds_conv_block(x, 64, stride=2, name="enc_ds1")   # 80×80×64
    skips["s1"] = x

    x = ds_conv_block(x, 128, stride=2, name="enc_ds2")  # 40×40×128
    skips["s2"] = x

    x = ds_conv_block(x, 256, stride=2, name="enc_ds3")  # 20×20×256

    # Decoder with skip connections
    x = layers.UpSampling2D(2, interpolation="bilinear", name="dec_up1")(x)  # 40×40
    x = layers.Concatenate(name="dec_cat1")([x, skips["s2"]])
    x = ds_conv_block(x, 128, stride=1, name="dec_ds1")

    x = layers.UpSampling2D(2, interpolation="bilinear", name="dec_up2")(x)  # 80×80
    x = layers.Concatenate(name="dec_cat2")([x, skips["s1"]])
    x = ds_conv_block(x, 64, stride=1, name="dec_ds2")

    # Output heatmap
    heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="heatmap")(x)

    model = keras.Model(inputs=inputs, outputs=heatmap)
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


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)

    train_samples = meta["samples"]["train"]
    val_samples = meta["samples"]["val"]

    # Preload all data into memory
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

    print("Loading training data...")
    X_train, y_train = load_split(train_samples, "train")
    print(f"  Train: {X_train.shape}, {y_train.shape}")

    print("Loading validation data...")
    X_val, y_val = load_split(val_samples, "val")
    print(f"  Val:   {X_val.shape}, {y_val.shape}")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=SEED)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build model
    model = build_model()
    model.summary()

    # Count trainable params
    trainable = int(np.sum([keras.backend.count_params(p) for p in model.trainable_weights]))
    print(f"\nTotal trainable params: {trainable:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError(name="mse")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(ARTIFACT_DIR / "best.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(ARTIFACT_DIR / "final.keras"))
    print(f"\nModel saved to {ARTIFACT_DIR}")

    # Evaluate pixel error (Keras model)
    y_val_pred = model.predict(X_val, verbose=0)

    mean_err, median_err = center_pixel_error(y_val, y_val_pred)
    print(f"\nKeras model — Validation center error ({HEATMAP_SIZE}×{HEATMAP_SIZE} heatmap px):")
    print(f"  Mean:   {mean_err:.3f} px")
    print(f"  Median: {median_err:.3f} px")

    scale_input = (INPUT_SIZE - 1) / (HEATMAP_SIZE - 1)
    print(f"  {INPUT_SIZE}×{INPUT_SIZE} equivalent:")
    print(f"  Mean:   {mean_err * scale_input:.2f} px")
    print(f"  Median: {median_err * scale_input:.2f} px")

    results = {
        "model": "depthwise_separable_cnn_v2_scaled",
        "params": trainable,
        "heatmap_size": HEATMAP_SIZE,
        "input_size": INPUT_SIZE,
        "keras_heatmap_pixels": {"mean": float(f"{mean_err:.4f}"), "median": float(f"{median_err:.4f}")},
        "keras_input_pixels": {
            "mean": float(f"{mean_err * scale_input:.4f}"),
            "median": float(f"{median_err * scale_input:.4f}"),
        },
        "epochs_trained": len(history.history["loss"]),
        "best_val_loss": float(f"{min(history.history['val_loss']):.6f}"),
    }

    # Export TFLite
    print("\nExporting TFLite models...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_fp32 = converter.convert()
    (ARTIFACT_DIR / "heatmap_cd_fp32.tflite").write_bytes(tflite_fp32)
    print(f"  float32: {len(tflite_fp32) / 1024:.1f} KB")

    # Int8 quantization
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

    # Validate TFLite models
    print("\n--- Validating TFLite models ---")

    def eval_tflite(tflite_path: str, label: str) -> dict:
        interp = tf.lite.Interpreter(str(tflite_path))
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        is_int8 = in_det["dtype"] == np.uint8
        input_scale, input_zero = in_det["quantization"] if is_int8 else (1.0, 0)
        out_scale, out_zero = out_det["quantization"] if is_int8 else (1.0, 0)

        errs = []
        for i in range(len(X_val)):
            if is_int8:
                inp = np.round(X_val[i:i+1] / input_scale + input_zero).astype(np.uint8)
            else:
                inp = X_val[i:i+1].astype(np.float32)

            interp.set_tensor(in_det["index"], inp)
            interp.invoke()
            raw = interp.get_tensor(out_det["index"])

            if is_int8:
                pred = (raw.astype(np.float32) - out_zero) * out_scale
            else:
                pred = raw

            gt_pt = softargmax_2d(y_val[i, :, :, 0])
            pred_pt = softargmax_2d(pred[0, :, :, 0])
            dist = np.sqrt((gt_pt[0] - pred_pt[0])**2 + (gt_pt[1] - pred_pt[1])**2)
            errs.append(dist)

        err_arr = np.array(errs)
        mean_px = float(err_arr.mean())
        median_px = float(np.median(err_arr))
        print(f"  {label}:")
        print(f"    Mean:   {mean_px:.3f} px ({mean_px * scale_input:.2f} px @{INPUT_SIZE})")
        print(f"    Median: {median_px:.3f} px ({median_px * scale_input:.2f} px @{INPUT_SIZE})")
        return {"mean": mean_px, "median": median_px}

    results["tflite_fp32_kb"] = round(len(tflite_fp32) / 1024, 1)
    results["tflite_int8_kb"] = round(len(tflite_int8) / 1024, 1)

    fp32_err = eval_tflite(str(ARTIFACT_DIR / "heatmap_cd_fp32.tflite"), "TFLite fp32")
    int8_err = eval_tflite(str(ARTIFACT_DIR / "heatmap_cd_int8.tflite"), "TFLite int8")

    results["tflite_fp32"] = fp32_err
    results["tflite_int8"] = int8_err

    (ARTIFACT_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone — results in {ARTIFACT_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
