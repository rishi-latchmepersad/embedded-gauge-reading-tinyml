#!/usr/bin/env python3
"""Train the compact ellipse-conditioned center/tip U-Net.

The deployment path is grayscale crop + ellipse mask -> two 56x56 sigmoid
heatmaps.  QAT is applied to the TFLite-compatible graph before export; no
float16 or post-training rescue conversion is used for this candidate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras
from PIL import Image

# why: tfmot patches tf.keras in this environment; the repository's working
# QAT scripts use the standalone tf_keras API so Conv2D objects stay tracked.
layers = keras.layers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
TEMP_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
ARTIFACTS = ROOT / "artifacts" / "gauge_center_tip_littlegood_v6"
INPUT_SIZE = 160
HEATMAP_SIZE = 80
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42


def configure_gpu() -> None:
    """Apply the repository's 15 GB WSL cap before TensorFlow allocates GPU memory."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Apply two compact Conv-BN-ReLU operations in a U-Net stage."""
    for index in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{index}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{index}")(x)
        x = layers.ReLU(6.0, name=f"{name}_relu{index}")(x)
    return x


def build_model() -> keras.Model:
    """Build a 160-input, 40-output, NPU-friendly two-channel U-Net."""
    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 2), name="ellipse_conditioned_input")
    e1 = conv_block(inputs, 16, "enc1")
    p1 = layers.MaxPooling2D(2, name="pool1")(e1)
    e2 = conv_block(p1, 24, "enc2")
    p2 = layers.MaxPooling2D(2, name="pool2")(e2)
    e3 = conv_block(p2, 40, "enc3")
    p3 = layers.MaxPooling2D(2, name="pool3")(e3)
    b = conv_block(p3, 64, "bottleneck")
    u2 = layers.UpSampling2D(2, interpolation="nearest", name="up2")(b)
    u2 = layers.Concatenate(name="cat2")([u2, e3])
    u2 = conv_block(u2, 40, "dec2")
    # The extra 2x nearest-neighbor stage costs little SRAM but halves the
    # output quantization step, which matters most for the needle tip.
    u1 = layers.UpSampling2D(2, interpolation="nearest", name="up1")(u2)
    u1 = layers.Concatenate(name="cat1")([u1, e2])
    u1 = conv_block(u1, 24, "dec1")
    out = layers.Conv2D(2, 1, activation="sigmoid", name="center_tip_heatmaps")(u1)
    return keras.Model(inputs, out, name="gauge_center_tip_v1")


def make_input(image_path: Path, row: dict[str, object]) -> np.ndarray:
    """Load grayscale crop and rasterize the upstream ellipse as channel two."""
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    ellipse = np.asarray(row["ellipse"], dtype=np.float32)
    # The crop is centered on the ellipse and scaled by max(rx, ry)*1.18.
    cx, cy, rx, ry = ellipse
    side = max(2.0 * rx, 2.0 * ry) * 1.18
    x = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / INPUT_SIZE * side + cx - side / 2.0
    y = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / INPUT_SIZE * side + cy - side / 2.0
    xx, yy = np.meshgrid(x, y)
    mask = (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0).astype(np.float32)
    # why: [-1,1] improves signed-int8 utilization while preserving the mask.
    return np.stack([image * 2.0 - 1.0, mask * 2.0 - 1.0], axis=-1)


def load_arrays(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load split images, ellipse masks, and two-channel Gaussian targets."""
    rows = json.loads((data_dir / "metadata.json").read_text())["splits"][split]
    inputs, targets = [], []
    for row in rows:
        inputs.append(make_input(data_dir / row["image"], row))
        targets.append(np.load(data_dir / row["heatmap"]).astype(np.float32))
    return np.stack(inputs), np.stack(targets)


def heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Weight Gaussian peak pixels so sparse heatmaps do not train to zero."""
    # why: the tip is the downstream decision point and has more ambiguous
    # annotations, so its channel receives twice the peak supervision.
    channel_weight = tf.constant([1.0, 2.0], dtype=y_true.dtype)[None, None, None, :]
    weights = 1.0 + 24.0 * y_true * channel_weight
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))


def export_int8(model: keras.Model, calibration: np.ndarray, path: Path) -> None:
    """Export a fully integer TFLite graph using representative crop inputs."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([sample[None].astype(np.float32)] for sample in calibration[:256])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path.write_bytes(converter.convert())


def main() -> None:
    """Train FP32, fine-tune QAT, and write the deployment candidate."""
    configure_gpu()
    tf.keras.utils.set_random_seed(SEED)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    x_train_base, y_train_base = load_arrays(DATA, "train")
    x_val_base, y_val_base = load_arrays(DATA, "val")
    x_train_temp, y_train_temp = load_arrays(TEMP_DATA, "train")
    x_val_temp, y_val_temp = load_arrays(TEMP_DATA, "val")
    # Keep the LittleGood domain at its natural frequency in the merged set;
    # this protects generalization while still exposing the model to the new
    # camera and gauge appearance.
    # why: LittleGood is the acceptance domain, but it is only a small
    # fraction of the generic gauge corpus; repeat it during training without
    # contaminating the held-out validation/test splits.
    # Merge each source sample once so this baseline measures generalization,
    # rather than gaining accuracy from duplicated LittleGood frames.
    x_train = np.concatenate((x_train_base, x_train_temp))
    y_train = np.concatenate((y_train_base, y_train_temp))
    x_val = np.concatenate((x_val_base, x_val_temp))
    y_val = np.concatenate((y_val_base, y_val_temp))
    base = build_model()
    base.compile(optimizer=keras.optimizers.Adam(1e-3), loss=heatmap_loss)
    base.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)], verbose=2)
    qat = tfmot.quantization.keras.quantize_model(base)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=heatmap_loss)
    qat.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=3, callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)], verbose=2)
    qat.save_weights(ARTIFACTS / "gauge_center_tip_v1_qat.weights.h5")
    export_int8(qat, x_train, ARTIFACTS / "gauge_center_tip_v1_int8.tflite")
    (ARTIFACTS / "training.json").write_text(json.dumps({"input_shape": [1, 160, 160, 2], "output_shape": [1, 80, 80, 2], "qat": True, "activation_budget_bytes": 1500000}, indent=2))
    print(f"Wrote {ARTIFACTS / 'gauge_center_tip_v1_int8.tflite'}")


if __name__ == "__main__":
    main()
