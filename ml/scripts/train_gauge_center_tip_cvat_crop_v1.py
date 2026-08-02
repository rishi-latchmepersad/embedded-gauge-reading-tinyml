"""Train the crop keypoint model with angle-diverse CVAT supervision."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from PIL import Image

from train_gauge_center_tip_vector_v1 import build_model, configure_gpu, coordinate_loss, export_int8, make_dataset, predict_int8


ROOT = Path(__file__).resolve().parents[1]
CVAT = ROOT.parent / "tmp" / "cvat_first450_filtered"
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_cvat_crop_littlegood_v1"
SIZE = 160
CROP_SCALE = 1.35


def center_box(element: ET.Element) -> np.ndarray:
    """Return a CVAT box center in 640-pixel coordinates."""
    return np.asarray(((float(element.attrib["xtl"]) + float(element.attrib["xbr"])) / 2.0, (float(element.attrib["ytl"]) + float(element.attrib["ybr"])) / 2.0), np.float32)


def load_cvat() -> tuple[np.ndarray, np.ndarray]:
    """Build crop-local CVAT tensors with the same transform as LittleGood."""
    root = ET.parse(CVAT / "annotations.xml").getroot(); inputs=[]; targets=[]
    for node in root.findall("image"):
        objects={label: next((x for x in node if x.attrib.get("label")==label), None) for label in ("Center","Tip","GaugeFace")}
        if any(value is None for value in objects.values()): continue
        face=objects["GaugeFace"].attrib; ellipse=np.asarray((float(face["cx"]),float(face["cy"]),float(face["rx"]),float(face["ry"])),np.float32); side=max(2*ellipse[2],2*ellipse[3])*CROP_SCALE; left,top=ellipse[:2]-side/2
        image=Image.open(CVAT/"images"/node.attrib["name"]).convert("L").crop((float(left),float(top),float(left+side),float(top+side))).resize((SIZE,SIZE),Image.Resampling.BILINEAR); gray=np.asarray(image,np.float32)/255
        axis=(np.arange(SIZE,dtype=np.float32)+.5)/SIZE*side; xx,yy=np.meshgrid(axis+left,axis+top); mask=(((xx-ellipse[0])/max(ellipse[2],1))**2+((yy-ellipse[1])/max(ellipse[3],1))**2<=1).astype(np.float32); inputs.append(np.stack((gray*2-1,mask*2-1),-1)); targets.append(np.concatenate(((center_box(objects["Center"])-[left,top])/side,(center_box(objects["Tip"])-[left,top])/side)))
    return np.asarray(inputs,np.float32),np.asarray(targets,np.float32)


def main() -> None:
    """Train, QAT-export, and evaluate all 97 untouched LittleGood frames."""
    configure_gpu();tf.keras.utils.set_random_seed(42); cvat_x,cvat_y=load_cvat(); student={s:np.load(STUDENT/f"{s}.npz") for s in ("train","val","test")}; train_x=np.concatenate((cvat_x,student["train"]["inputs"]));train_y=np.concatenate((cvat_y,student["train"]["points"].reshape(-1,4)));val_x=student["val"]["inputs"];val_y=student["val"]["points"].reshape(-1,4); train_ds=make_dataset(train_x,train_y,True,12);val_ds=make_dataset(val_x,val_y,False,0);model=build_model();model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=coordinate_loss);model.fit(train_ds,validation_data=val_ds,epochs=15,verbose=2);qat=tfmot.quantization.keras.quantize_model(model);qat.compile(optimizer=tf.keras.optimizers.Adam(2e-4),loss=coordinate_loss);qat.fit(train_ds,validation_data=val_ds,epochs=6,verbose=2);OUT.mkdir(parents=True,exist_ok=True);path=OUT/"gauge_center_tip_cvat_crop_v1_int8.tflite";contract=export_int8(qat,train_x,path);prediction=predict_int8(path,student["test"]["inputs"]).reshape(-1,2,2);truth=student["test"]["points"];errors=np.linalg.norm((prediction-truth)*SIZE,axis=2);report={"cvat_samples":len(cvat_x),"littlegood_test_samples":len(truth),"center_within_8px":float(np.mean(errors[:,0]<=8)),"tip_within_8px":float(np.mean(errors[:,1]<=8)),"center_error_px_mean":float(errors[:,0].mean()),"tip_error_px_mean":float(errors[:,1].mean()),"contract":contract};(OUT/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
