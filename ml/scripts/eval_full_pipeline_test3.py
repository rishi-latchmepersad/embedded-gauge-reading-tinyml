"""Evaluate full pipeline: QAT encoder ellipse → v9 center/tip on test_3."""
import sys
sys.path.insert(0, "scripts")
import zipfile, io, numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
from xml.etree import ElementTree as ET

# Load ellipse model
ell_path = Path("artifacts/gauge_ellipse_qat_encoder_v1/ellipse_qat_encoder_int8.tflite")
ell_interp = tf.lite.Interpreter(model_path=str(ell_path))
ell_interp.allocate_tensors()
ell_in, ell_out = ell_interp.get_input_details()[0], ell_interp.get_output_details()[0]

# Load center/tip model
ct_path = Path("artifacts/gauge_center_tip_littlegood_v9/gauge_center_tip_v8_int8.tflite")
ct_interp = tf.lite.Interpreter(model_path=str(ct_path))
ct_interp.allocate_tensors()
ct_in, ct_out = ct_interp.get_input_details()[0], ct_interp.get_output_details()[0]

z = zipfile.ZipFile("data/labelled/test_3.zip")
root = ET.parse(z.open("annotations.xml")).getroot()

CROP_SCALE = 1.35
INPUT_SIZE = 160

print("Full pipeline (QAT encoder ellipse → v9 center/tip) on test_3:")
errors_c, errors_t = [], []
for img_elem in root.findall("image"):
    name = img_elem.get("name")
    for e in img_elem:
        if e.get("label") == "GaugeFace" and e.tag == "ellipse":
            gt_e = {c: float(e.get(c)) for c in ("cx", "cy", "rx", "ry")}
        elif e.get("label") == "Center" and e.tag == "ellipse":
            gt_c = np.array([float(e.get("cx")), float(e.get("cy"))])
        elif e.get("label") == "Tip" and e.tag == "ellipse":
            gt_t = np.array([float(e.get("cx")), float(e.get("cy"))])

    data = z.read(f"images/{name}")
    img = Image.open(io.BytesIO(data)).convert("L")
    img_224 = np.asarray(img.resize((224, 224)), np.float32)[None, ..., None] / 255.0

    # Ellipse detection
    s, zp = ell_in["quantization"]
    t = np.clip(np.round(img_224 / float(s) + float(zp)), -128, 127).astype(np.int8)
    ell_interp.set_tensor(ell_in["index"], t)
    ell_interp.invoke()
    raw = ell_interp.get_tensor(ell_out["index"]).astype(np.float32)
    s, zp = ell_out["quantization"]
    pred_ell = (raw - float(zp)) * float(s)
    pcx, pcy, prx, pry = pred_ell[0, :4]

    # Map to 640px
    ecx, ecy, erx, ery = pcx * 640, pcy * 640, prx * 640, pry * 640
    side = max(2 * erx, 2 * ery) * CROP_SCALE
    left, top = ecx - side / 2, ecy - side / 2

    # Crop and resize to 160x160
    left_i, top_i = int(max(0, left)), int(max(0, top))
    crop = Image.fromarray(np.asarray(img, np.uint8)).crop(
        (left_i, top_i, int(left + side), int(top + side))
    )
    crop160 = np.asarray(crop.resize((INPUT_SIZE, INPUT_SIZE)), np.float32) / 255.0

    # Ellipse mask
    xs = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / float(INPUT_SIZE) * side + left
    ys = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / float(INPUT_SIZE) * side + top
    xx, yy = np.meshgrid(xs, ys)
    mask = (((xx - ecx) / max(erx, 1)) ** 2 + ((yy - ecy) / max(ery, 1)) ** 2 <= 1).astype(np.float32)
    ct_input = np.stack([crop160 * 2 - 1, mask * 2 - 1], axis=-1).astype(np.float32)

    # Center/tip detection
    s, zp = ct_in["quantization"]
    t = np.clip(np.round(ct_input[None] / float(s) + float(zp)), -128, 127).astype(np.int8)
    ct_interp.set_tensor(ct_in["index"], t)
    ct_interp.invoke()
    raw = ct_interp.get_tensor(ct_out["index"]).astype(np.float32)
    s, zp = ct_out["quantization"]
    hm = (raw - float(zp)) * float(s)

    # Decode heatmaps
    decoded = []
    for ch in range(2):
        h = hm[0, ..., ch]
        y, x = np.unravel_index(np.argmax(h), h.shape)
        y0, y1 = max(0, y - 4), min(80, y + 5)
        x0, x1 = max(0, x - 4), min(80, x + 5)
        yyh, xxh = np.mgrid[y0:y1, x0:x1]
        w = np.maximum(h[y0:y1, x0:x1] - 0.03, 0) ** 2
        total = float(w.sum())
        if total > 0:
            pt = np.asarray([(xxh * w).sum() / total + 0.5, (yyh * w).sum() / total + 0.5], np.float32) / 80
        else:
            pt = np.asarray([(x + 0.5) / 80, (y + 0.5) / 80], np.float32)
        decoded.append(pt)
    decoded = np.asarray(decoded, np.float32)

    pred_c = np.array([left, top]) + decoded[0] * side
    pred_t = np.array([left, top]) + decoded[1] * side
    c_err = float(np.linalg.norm(pred_c - gt_c))
    t_err = float(np.linalg.norm(pred_t - gt_t))
    errors_c.append(c_err)
    errors_t.append(t_err)
    print(f"  {name}: center={c_err:.1f}px tip={t_err:.1f}px  (ellipse: cx={pcx:.3f} cy={pcy:.3f})")

print()
ec, et = np.array(errors_c), np.array(errors_t)
print(f"Center <=8px: {np.mean(ec <= 8):.1%} ({int(np.sum(ec <= 8))}/11)")
print(f"Tip <=8px:    {np.mean(et <= 8):.1%} ({int(np.sum(et <= 8))}/11)")
print(f"Center mean: {ec.mean():.1f}px  median: {np.median(ec):.1f}px  max: {ec.max():.1f}px")
print(f"Tip mean:    {et.mean():.1f}px  median: {np.median(et):.1f}px  max: {et.max():.1f}px")
