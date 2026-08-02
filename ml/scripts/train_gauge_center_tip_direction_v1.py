"""Train full-frame heatmaps with a canonical-radius tip target."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_fullframe_v1 import decode, tip_weighted_loss
from train_gauge_center_tip_v1 import build_model, configure_gpu, export_int8, load_arrays

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
TEMP = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
OUT = ROOT / "artifacts" / "gauge_center_tip_direction_littlegood_v4"
INPUT = 160
SIZE = 80
RADIUS = 0.29


def build_model_224() -> keras.Model:
    """Build the same compact U-Net at the board's 224px input contract."""
    layers = keras.layers; inputs = keras.Input((INPUT, INPUT, 2), name="fullframe_224_input")
    def block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        """Apply two quantization-friendly convolution blocks."""
        for index in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{index}")(x); x = layers.BatchNormalization(name=f"{name}_bn{index}")(x); x = layers.ReLU(6.0, name=f"{name}_relu{index}")(x)
        return x
    e1=block(inputs,16,"enc1"); e2=block(layers.MaxPooling2D(2)(e1),24,"enc2"); e3=block(layers.MaxPooling2D(2)(e2),40,"enc3"); b=block(layers.MaxPooling2D(2)(e3),64,"bottleneck"); u2=layers.UpSampling2D(2,interpolation="nearest")(b);u2=layers.Concatenate()([u2,e3]);u2=block(u2,40,"dec2");u1=layers.UpSampling2D(2,interpolation="nearest")(u2);u1=layers.Concatenate()([u1,e2]);u1=block(u1,24,"dec1");out=layers.Conv2D(2,1,activation="sigmoid")(u1);return keras.Model(inputs,out)


def fixed_targets(data_dir: Path, split: str) -> np.ndarray:
    """Create center and canonical-radius tip Gaussian targets."""
    rows = json.loads((data_dir / "metadata.json").read_text())["splits"][split]
    result = np.zeros((len(rows), SIZE, SIZE, 2), dtype=np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for index, row in enumerate(rows):
        center = np.asarray(row["center_xy_norm"], dtype=np.float32)
        tip = np.asarray(row["tip_xy_norm"], dtype=np.float32)
        direction = tip - center; direction /= np.linalg.norm(direction) + 1e-6
        points = (center, center + direction * RADIUS)
        for channel, point in enumerate(points):
            px, py = point * SIZE - .5
            result[index, ..., channel] = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 2.2**2))
    return result


def fixed_targets_points(points: np.ndarray) -> np.ndarray:
    """Create canonical-radius targets from already-localized crop points."""
    result = np.zeros((len(points), SIZE, SIZE, 2), dtype=np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for index, pair in enumerate(points):
        center, tip = pair
        direction = tip - center
        direction /= np.linalg.norm(direction) + 1e-6
        for channel, point in enumerate((center, center + direction * RADIUS)):
            px, py = point * SIZE - 0.5
            result[index, ..., channel] = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 2.2**2))
    return result


def fixed_targets_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Convert existing crop-local center/tip heatmaps to canonical targets."""
    points = decode(heatmaps)
    return fixed_targets_points(points)


def main() -> None:
    """Train, QAT-export, and score canonical-radius direction predictions."""
    configure_gpu(); tf.keras.utils.set_random_seed(42); OUT.mkdir(parents=True, exist_ok=True)
    xb, yb = load_arrays(DATA, "train"); xv, yv = load_arrays(DATA, "val")
    student = {split: np.load(ROOT / "data/initial_temp_gauge_v1/student_conditioned" / f"{split}.npz") for split in ("train", "val", "test")}
    x_train = np.concatenate((xb, student["train"]["inputs"]))
    y_train = np.concatenate((fixed_targets_heatmaps(yb), fixed_targets_points(student["train"]["points"])))
    x_val = np.concatenate((xv, student["val"]["inputs"]))
    y_val = np.concatenate((fixed_targets_heatmaps(yv), fixed_targets_points(student["val"]["points"])))
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tip_weighted_loss)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, epochs=12, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=tip_weighted_loss)
    qat.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, epochs=4, verbose=2)
    path = OUT / "gauge_center_tip_direction_v1_int8.tflite"; export_int8(qat, x_train, path)
    interpreter = tf.lite.Interpreter(model_path=str(path)); interpreter.allocate_tensors(); inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]; predictions=[]
    xt = student["test"]["inputs"].astype(np.float32)
    for sample in xt:
        scale, zero = inp["quantization"]; interpreter.set_tensor(inp["index"], np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]); interpreter.invoke(); raw=interpreter.get_tensor(out["index"]).astype(np.float32); scale, zero=out["quantization"]; predictions.append((raw-zero)*scale)
    predictions=np.concatenate(predictions); decoded=decode(predictions); targets=student["test"]["points"]; direction=decoded[:,1]-decoded[:,0]; direction/=np.linalg.norm(direction,axis=1,keepdims=True)+1e-6; projected=decoded[:,0]+direction*RADIUS; result=np.stack((decoded[:,0],projected),axis=1); errors=np.linalg.norm((result-targets)*160,axis=2)
    report={"samples":len(errors),"center_within_8px":float((errors[:,0]<=8).mean()),"tip_within_8px":float((errors[:,1]<=8).mean()),"center_error_px_mean":float(errors[:,0].mean()),"tip_error_px_mean":float(errors[:,1].mean()),"radius":RADIUS,"bytes":path.stat().st_size};(OUT/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
