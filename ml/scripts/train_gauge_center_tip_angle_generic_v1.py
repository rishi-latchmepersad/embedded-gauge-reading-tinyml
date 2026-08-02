"""Train a generic-corpus angle, center, and radius QAT keypoint model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_direction_radius_v1 import radius_targets
from train_gauge_center_tip_fullframe_v1 import decode
from train_gauge_center_tip_v1 import configure_gpu, load_arrays
from train_gauge_center_tip_angle_v1 import build_model


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_angle_generic_littlegood_v1"
SCALE = 0.5


def targets(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode center, normalized radius, and unit direction."""
    vector = points[:, 1] - points[:, 0]; length = np.linalg.norm(vector, axis=1, keepdims=True); return np.concatenate((points[:, 0], length / SCALE), axis=1), vector / (length + 1e-6)


def dataset(inputs: np.ndarray, geometry: np.ndarray, direction: np.ndarray, training: bool) -> tf.data.Dataset:
    """Build a no-duplication quarter-turn geometric augmentation dataset."""
    ds = tf.data.Dataset.from_tensor_slices((inputs, geometry, direction))
    if training: ds = ds.shuffle(len(inputs), seed=42, reshuffle_each_iteration=True)
    def augment(image: tf.Tensor, geo: tf.Tensor, unit: tf.Tensor) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Rotate center and direction consistently with the crop."""
        k = tf.random.uniform((), 0, 4, dtype=tf.int32, seed=42); image = tf.image.rot90(image, k); center = geo[:2];
        center = tf.switch_case(k, branch_fns=(lambda: center, lambda: tf.stack((center[1], 1.0-center[0])), lambda: 1.0-center, lambda: tf.stack((1.0-center[1], center[0]))))
        unit = tf.switch_case(k, branch_fns=(lambda: unit, lambda: tf.stack((unit[1],-unit[0])), lambda: -unit, lambda: tf.stack((-unit[1],unit[0]))))
        return image, (tf.concat((center, geo[2:]), axis=0), unit)
    if training: ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else: ds = ds.map(lambda image, geo, unit: (image, (geo, unit)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(16).prefetch(tf.data.AUTOTUNE)


def export_int8(model: keras.Model, calibration: np.ndarray, path: Path) -> dict[str, object]:
    """Export and describe the multi-output full-int8 graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model); converter.optimizations = [tf.lite.Optimize.DEFAULT]; indices = np.linspace(0, len(calibration)-1, min(256,len(calibration)), dtype=int); converter.representative_dataset = lambda: ([calibration[i][None].astype(np.float32)] for i in indices); converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; converter.inference_input_type=tf.int8; converter.inference_output_type=tf.int8; blob=converter.convert();path.write_bytes(blob);it=tf.lite.Interpreter(model_content=blob);it.allocate_tensors();return {"bytes":len(blob),"input":it.get_input_details()[0]["shape"].tolist(),"outputs":[x["shape"].tolist() for x in it.get_output_details()]}


def main() -> None:
    """Train, export, and score the untouched corrected LittleGood test."""
    configure_gpu(); tf.keras.utils.set_random_seed(42); OUT.mkdir(parents=True, exist_ok=True); gx,gh=load_arrays(DATA,"train");gv,gvh=load_arrays(DATA,"val");s={k:np.load(STUDENT/f"{k}.npz") for k in ("train","val","test")};gp=decode(gh);gvp=decode(gvh);ggeo,gdir=targets(gp);vgeo,vdir=targets(gvp);sgeo,sdir=targets(s["train"]["points"]);svgeo,svdir=targets(s["val"]["points"]);xtr=np.concatenate((gx,s["train"]["inputs"]));xv=np.concatenate((gv,s["val"]["inputs"]));geo=np.concatenate((ggeo,sgeo));directions=np.concatenate((gdir,sdir));vgeo=np.concatenate((vgeo,svgeo));vdir=np.concatenate((vdir,svdir));model=build_model();model.compile(optimizer=keras.optimizers.Adam(1e-3),loss=[keras.losses.Huber(.03),keras.losses.MeanSquaredError()],loss_weights=[1,3]);model.fit(dataset(xtr,geo,directions,True),validation_data=dataset(xv,vgeo,vdir,False),epochs=14,verbose=2);qat=tfmot.quantization.keras.quantize_model(model);qat.compile(optimizer=keras.optimizers.Adam(2e-4),loss=[keras.losses.Huber(.03),keras.losses.MeanSquaredError()],loss_weights=[1,3]);qat.fit(dataset(xtr,geo,directions,True),validation_data=dataset(xv,vgeo,vdir,False),epochs=5,verbose=2);path=OUT/"gauge_center_tip_angle_generic_v1_int8.tflite";contract=export_int8(qat,xtr,path);it=tf.lite.Interpreter(model_path=str(path));it.allocate_tensors();inp=it.get_input_details()[0];outs=sorted(it.get_output_details(),key=lambda x:int(x["shape"][-1]),reverse=True);vals=[]
    for sample in s["test"]["inputs"]:
        sc,z=inp["quantization"];it.set_tensor(inp["index"],np.clip(np.round(sample/sc+z),-128,127).astype(np.int8)[None]);it.invoke();row=[]
        for o in outs:
            raw=it.get_tensor(o["index"]).astype(np.float32);sc,z=o["quantization"];row.append((raw-z)*sc)
        vals.append(row)
    geo=np.concatenate([v[0] for v in vals]);direction=np.concatenate([v[1] for v in vals]);direction/=np.linalg.norm(direction,axis=1,keepdims=True)+1e-6;prediction=np.stack((geo[:,:2],geo[:,:2]+direction*geo[:,2:3]*SCALE),axis=1);errors=np.linalg.norm((prediction-s["test"]["points"])*160,axis=2);report={"samples":len(errors),"center_within_8px":float(np.mean(errors[:,0]<=8)),"tip_within_8px":float(np.mean(errors[:,1]<=8)),"center_error_px_mean":float(errors[:,0].mean()),"tip_error_px_mean":float(errors[:,1].mean()),"contract":contract};(OUT/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
