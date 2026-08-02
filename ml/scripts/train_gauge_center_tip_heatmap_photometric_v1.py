"""Train a photometrically augmented endpoint heatmap QAT model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from train_gauge_center_tip_fullframe_v1 import decode
from train_gauge_center_tip_v1 import build_model, configure_gpu, export_int8, heatmap_loss, load_arrays


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_heatmap_photometric_littlegood_v1"


def dataset(inputs: np.ndarray, targets: np.ndarray, training: bool) -> tf.data.Dataset:
    """Apply photometric-only augmentation while preserving point geometry."""
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if training: ds = ds.shuffle(len(inputs), seed=42, reshuffle_each_iteration=True)
    def augment(image: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Change grayscale appearance but leave the ellipse mask and labels fixed."""
        gray, mask = image[..., :1], image[..., 1:]
        gray = tf.image.random_brightness(gray, 0.15, seed=42); gray = tf.image.random_contrast(gray, 0.70, 1.30, seed=43); gray += tf.random.normal(tf.shape(gray), stddev=0.025, seed=44)
        return tf.concat((tf.clip_by_value(gray, -1.0, 1.0), mask), axis=-1), target
    if training: ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(16).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    """Train, QAT-export, and evaluate all corrected LittleGood test frames."""
    configure_gpu(); tf.keras.utils.set_random_seed(42); OUT.mkdir(parents=True, exist_ok=True); gx,gy=load_arrays(DATA,"train");gv,gyv=load_arrays(DATA,"val");s={k:np.load(STUDENT/f"{k}.npz") for k in ("train","val","test")};xtr=np.concatenate((gx,s["train"]["inputs"]));ytr=np.concatenate((gy,s["train"]["heatmaps"]));xv=np.concatenate((gv,s["val"]["inputs"]));yv=np.concatenate((gyv,s["val"]["heatmaps"]));model=build_model();model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=heatmap_loss);model.fit(dataset(xtr,ytr,True),validation_data=dataset(xv,yv,False),epochs=12,verbose=2);qat=tfmot.quantization.keras.quantize_model(model);qat.compile(optimizer=tf.keras.optimizers.Adam(2e-4),loss=heatmap_loss);qat.fit(dataset(xtr,ytr,True),validation_data=dataset(xv,yv,False),epochs=4,verbose=2);path=OUT/"gauge_center_tip_heatmap_photometric_v1_int8.tflite";contract=export_int8(qat,xtr,path);it=tf.lite.Interpreter(model_path=str(path));it.allocate_tensors();inp=it.get_input_details()[0];out=it.get_output_details()[0];pred=[]
    for sample in s["test"]["inputs"]:
        sc,z=inp["quantization"];it.set_tensor(inp["index"],np.clip(np.round(sample/sc+z),-128,127).astype(np.int8)[None]);it.invoke();raw=it.get_tensor(out["index"]).astype(np.float32);sc,z=out["quantization"];pred.append((raw-z)*sc)
    decoded=decode(np.concatenate(pred));errors=np.linalg.norm((decoded-s["test"]["points"])*160,axis=2);report={"samples":len(errors),"center_within_8px":float(np.mean(errors[:,0]<=8)),"tip_within_8px":float(np.mean(errors[:,1]<=8)),"center_error_px_mean":float(errors[:,0].mean()),"tip_error_px_mean":float(errors[:,1].mean()),"contract":contract};(OUT/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
