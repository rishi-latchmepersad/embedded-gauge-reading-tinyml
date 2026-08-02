"""PTQ export + test_3 evaluation for MobileNetV2 ellipse model."""
import sys
sys.path.insert(0, "scripts")
import numpy as np, json, zipfile, io
import tensorflow as tf
import tf_keras as keras
from pathlib import Path
from PIL import Image
from xml.etree import ElementTree as ET
from train_gauge_ellipse_mobilenetv2 import build_model, IMAGE_SIZE

# Load FP32 weights
model = build_model(alpha=0.35)
model.load_weights("artifacts/gauge_ellipse_mobilenetv2_v1/fp32.weights.h5")

# Build representative dataset from diverse training images
train_paths = sorted(Path("data/gauge_face_ellipse_v1_640_gray/images/train").glob("*.png"))
rng = np.random.default_rng(42)
cal_indices = rng.permutation(len(train_paths))[:500]

def representative():
    for idx in cal_indices:
        img = tf.io.decode_png(tf.io.read_file(str(train_paths[idx])), channels=1)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        yield [tf.cast(img[None], tf.float32) / 255.0]

# PTQ export
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
blob = converter.convert()
out_path = Path("artifacts/gauge_ellipse_mobilenetv2_v1/ellipse_mnv2_ptq_int8.tflite")
out_path.write_bytes(blob)
print(f"Exported {len(blob)} bytes to {out_path}")

# Evaluate on test_3
interp = tf.lite.Interpreter(model_content=blob)
interp.allocate_tensors()
inp_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]

z = zipfile.ZipFile("data/labelled/test_3.zip")
root = ET.parse(z.open("annotations.xml")).getroot()
print("\nPTQ int8 on test_3:")
errors = []
for img_elem in root.findall("image"):
    name = img_elem.get("name")
    for e in img_elem:
        if e.get("label") == "GaugeFace" and e.tag == "ellipse":
            gt = {c: float(e.get(c)) / 640.0 for c in ("cx", "cy", "rx", "ry")}
    data = z.read(f"images/{name}")
    img = Image.open(io.BytesIO(data)).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, np.float32)[None, ..., None] / 255.0
    s, zp = inp_d["quantization"]
    t = np.clip(np.round(arr / float(s) + float(zp)), -128, 127).astype(np.int8)
    interp.set_tensor(inp_d["index"], t)
    interp.invoke()
    raw = interp.get_tensor(out_d["index"]).astype(np.float32)
    s, zp = out_d["quantization"]
    pred = (raw - float(zp)) * float(s)
    pcx, pcy, prx, pry = pred[0, :4]
    cerr = np.sqrt((pcx - gt["cx"]) ** 2 + (pcy - gt["cy"]) ** 2) * 640
    errors.append(cerr)
    print(f"  {name}: cx={pcx:.4f}({gt['cx']:.4f}) cy={pcy:.4f}({gt['cy']:.4f}) rx={prx:.4f}({gt['rx']:.4f}) ry={pry:.4f}({gt['ry']:.4f}) center_err={cerr:.1f}px")

print(f"\nCenter err mean: {np.mean(errors):.1f}px  max: {np.max(errors):.1f}px")
print(f"Center <=8px: {np.mean(np.array(errors) <= 8):.1%}")
