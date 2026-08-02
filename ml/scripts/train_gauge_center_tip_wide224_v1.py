"""Train a higher-resolution QAT keypoint model for small needle details."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_direction_radius_v1 import radius_targets
from train_gauge_center_tip_v1 import configure_gpu


ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_wide224_littlegood_v1"
INPUT = 224
HEATMAP = 112


def heat_targets(points: np.ndarray) -> np.ndarray:
    """Create high-resolution center and canonical-direction heatmaps."""
    yy, xx = np.mgrid[0:HEATMAP, 0:HEATMAP]; result = np.zeros((len(points), HEATMAP, HEATMAP, 2), np.float32)
    for index, pair in enumerate(points):
        center, tip = pair; direction = tip - center; direction /= np.linalg.norm(direction) + 1e-6
        for channel, point in enumerate((center, center + direction * 0.27)):
            px, py = point * HEATMAP - 0.5; result[index, ..., channel] = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 2.2**2))
    return result


def block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Apply two quantization-friendly convolution blocks."""
    for index in range(2):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{index}")(x); x = keras.layers.BatchNormalization(name=f"{name}_bn{index}")(x); x = keras.layers.ReLU(6.0, name=f"{name}_relu{index}")(x)
    return x


def build_model() -> keras.Model:
    """Build a 224-input U-Net with a scalar radius head."""
    layers = keras.layers; inputs = keras.Input((INPUT, INPUT, 2), name="wide224_input")
    e1 = block(inputs, 24, "enc1"); e2 = block(layers.MaxPooling2D(2)(e1), 32, "enc2"); e3 = block(layers.MaxPooling2D(2)(e2), 48, "enc3"); b = block(layers.MaxPooling2D(2)(e3), 80, "bottleneck")
    u2 = layers.UpSampling2D(2, interpolation="nearest")(b); u2 = block(layers.Concatenate()([u2, e3]), 48, "dec2"); u1 = layers.UpSampling2D(2, interpolation="nearest")(u2); u1 = block(layers.Concatenate()([u1, e2]), 32, "dec1"); heat = layers.Conv2D(2, 1, activation="sigmoid", name="heatmaps")(u1)
    radius = layers.Conv2D(16, 28, padding="valid", use_bias=False, name="radius_collapse")(b); radius = layers.BatchNormalization()(radius); radius = layers.ReLU(6.0)(radius); radius = layers.Flatten()(radius); radius = layers.Dense(24, activation="relu")(radius); radius = layers.Dense(1, activation="sigmoid", name="radius")(radius)
    return keras.Model(inputs, [heat, radius])


def resize_inputs(inputs: np.ndarray) -> np.ndarray:
    """Resize the prepared crop contract without changing normalized labels."""
    return tf.image.resize(inputs, (INPUT, INPUT), method="bilinear").numpy().astype(np.float32)


def dataset(inputs: np.ndarray, heat: np.ndarray, radius: np.ndarray, training: bool) -> tf.data.Dataset:
    """Build a no-duplication quarter-turn augmentation dataset."""
    ds = tf.data.Dataset.from_tensor_slices((inputs, heat, radius))
    if training: ds = ds.shuffle(len(inputs), seed=42, reshuffle_each_iteration=True)
    def augment(image: tf.Tensor, target: tf.Tensor, length: tf.Tensor) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Rotate image and heatmaps while preserving radius."""
        k = tf.random.uniform((), 0, 4, dtype=tf.int32, seed=42); return tf.image.rot90(image, k), (tf.image.rot90(target, k), length)
    ds = ds.map(augment if training else lambda image, target, length: (image, (target, length)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(16).prefetch(tf.data.AUTOTUNE)


def export_int8(model: keras.Model, calibration: np.ndarray, path: Path) -> dict[str, object]:
    """Export a fully integer two-output TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model); converter.optimizations = [tf.lite.Optimize.DEFAULT]; indices = np.linspace(0, len(calibration) - 1, min(256, len(calibration)), dtype=int); converter.representative_dataset = lambda: ([calibration[i][None].astype(np.float32)] for i in indices); converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; converter.inference_input_type = tf.int8; converter.inference_output_type = tf.int8; blob = converter.convert(); path.write_bytes(blob); interpreter = tf.lite.Interpreter(model_content=blob); interpreter.allocate_tensors(); return {"bytes": len(blob), "input": interpreter.get_input_details()[0]["shape"].tolist(), "outputs": [x["shape"].tolist() for x in interpreter.get_output_details()]}


def decode(heatmaps: np.ndarray) -> np.ndarray:
    """Decode both high-resolution peaks with weighted centroids."""
    result=[]
    for sample in heatmaps:
        row=[]
        for channel in range(2):
            hm=sample[...,channel]; y,x=np.unravel_index(np.argmax(hm),hm.shape); y0,y1=max(0,y-6),min(HEATMAP,y+7);x0,x1=max(0,x-6),min(HEATMAP,x+7); yy,xx=np.mgrid[y0:y1,x0:x1]; w=np.maximum(hm[y0:y1,x0:x1]-.03,0)**2; total=w.sum(); row.append(np.asarray(((xx*w).sum()/total+.5,(yy*w).sum()/total+.5),np.float32)/HEATMAP if total else np.asarray((x+.5,y+.5),np.float32)/HEATMAP)
        result.append(row)
    return np.asarray(result,np.float32)


def main() -> None:
    """Train, export, and score the corrected untouched LittleGood test."""
    configure_gpu(); tf.keras.utils.set_random_seed(42); OUT.mkdir(parents=True, exist_ok=True); data={s:np.load(STUDENT/f"{s}.npz") for s in ("train","val","test")}; xtr=resize_inputs(data["train"]["inputs"]);xv=resize_inputs(data["val"]["inputs"]);xt=resize_inputs(data["test"]["inputs"]);htr=heat_targets(data["train"]["points"]);hv=heat_targets(data["val"]["points"]);rtr=radius_targets(data["train"]["points"]);rv=radius_targets(data["val"]["points"]); losses=[tf.keras.losses.MeanSquaredError(),tf.keras.losses.Huber(.03)];model=build_model();model.compile(optimizer=keras.optimizers.Adam(1e-3),loss=losses,loss_weights=[1,4]);model.fit(dataset(xtr,htr,rtr,True),validation_data=dataset(xv,hv,rv,False),epochs=14,verbose=2);qat=tfmot.quantization.keras.quantize_model(model);qat.compile(optimizer=keras.optimizers.Adam(2e-4),loss=losses,loss_weights=[1,4]);qat.fit(dataset(xtr,htr,rtr,True),validation_data=dataset(xv,hv,rv,False),epochs=5,verbose=2);path=OUT/"gauge_center_tip_wide224_v1_int8.tflite";contract=export_int8(qat,xtr,path);interpreter=tf.lite.Interpreter(model_path=str(path));interpreter.allocate_tensors();inp=interpreter.get_input_details()[0];outs=interpreter.get_output_details();heat=next(o for o in outs if len(o["shape"])==4);rad=next(o for o in outs if len(o["shape"])==2);hs=[];rs=[]
    for sample in xt:
        sc,z=inp["quantization"];interpreter.set_tensor(inp["index"],np.clip(np.round(sample/sc+z),-128,127).astype(np.int8)[None]);interpreter.invoke()
        for o,a in ((heat,hs),(rad,rs)):
            raw=interpreter.get_tensor(o["index"]).astype(np.float32);sc,z=o["quantization"];a.append((raw-z)*sc)
    d=decode(np.concatenate(hs));v=d[:,1]-d[:,0];v/=np.linalg.norm(v,axis=1,keepdims=True)+1e-6;prediction=np.stack((d[:,0],d[:,0]+v*np.concatenate(rs)*.5),1);errors=np.linalg.norm((prediction-data["test"]["points"])*160,axis=2);report={"samples":len(errors),"center_within_8px":float(np.mean(errors[:,0]<=8)),"tip_within_8px":float(np.mean(errors[:,1]<=8)),"center_error_px_mean":float(errors[:,0].mean()),"tip_error_px_mean":float(errors[:,1].mean()),"contract":contract};(OUT/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
