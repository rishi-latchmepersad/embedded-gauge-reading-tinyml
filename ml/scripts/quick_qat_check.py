"""Quick check: does the QAT model also collapse on in-domain images?"""
import sys
sys.path.insert(0, "scripts")
import zipfile, io, numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

tflite = Path("artifacts/gauge_ellipse_littlegood_v11/gauge_ellipse_v9_int8.tflite")
interp = tf.lite.Interpreter(model_path=str(tflite))
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]

# Load FP32 model
from train_gauge_ellipse_v9 import build_model
fp32 = build_model()
fp32.load_weights("artifacts/gauge_ellipse_littlegood_v11/gauge_ellipse_v9_fp32.weights.h5")

z = zipfile.ZipFile("data/labelled/test_1.zip")
from xml.etree import ElementTree as ET
root = ET.parse(z.open("annotations.xml")).getroot()
images = root.findall("image")[:5]

print("Comparison: QAT int8 vs FP32 on 5 generic test_1 images:")
for img_elem in images:
    name = img_elem.get("name")
    data = z.read(f"images/{name}")
    img = Image.open(io.BytesIO(data)).convert("L").resize((320, 320))
    img_arr = np.asarray(img, np.float32)[None, ..., None] / 255.0

    # FP32
    fp = fp32.predict(img_arr, verbose=0)[0]

    # QAT int8
    s, zp = inp_d["quantization"]
    t = np.clip(np.round(img_arr / float(s) + float(zp)), -128, 127).astype(np.int8)
    interp.set_tensor(inp_d["index"], t)
    interp.invoke()
    raw = interp.get_tensor(out_d["index"]).astype(np.float32)
    s, zp = out_d["quantization"]
    qat = (raw - float(zp)) * float(s)

    print(f"  {name[:30]}: FP32=({fp[0]:.3f},{fp[1]:.3f},{fp[2]:.3f},{fp[3]:.3f})  QAT=({qat[0,0]:.3f},{qat[0,1]:.3f},{qat[0,2]:.3f},{qat[0,3]:.3f})")
