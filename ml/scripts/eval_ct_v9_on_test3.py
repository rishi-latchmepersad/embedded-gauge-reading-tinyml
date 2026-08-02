"""Evaluate v9 center/tip model on test_3 using ground-truth ellipses."""
import zipfile, io, json, numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
from xml.etree import ElementTree as ET

ct_path = Path('artifacts/gauge_center_tip_littlegood_v9/gauge_center_tip_v8_int8.tflite')
ct_interp = tf.lite.Interpreter(model_path=str(ct_path))
ct_interp.allocate_tensors()
ct_in = ct_interp.get_input_details()[0]
ct_out = ct_interp.get_output_details()[0]

z = zipfile.ZipFile('data/labelled/test_3.zip')
tree = ET.parse(z.open('annotations.xml'))
root = tree.getroot()

INPUT_SIZE = 160
CROP_SCALE = 1.35
errors_c, errors_t = [], []

for image_elem in root.findall('image'):
    name = image_elem.get('name')
    for elem in image_elem:
        label = elem.get('label', '')
        if label == 'GaugeFace' and elem.tag == 'ellipse':
            gt_e = np.array([float(elem.get(c)) for c in ('cx','cy','rx','ry')])
        elif label == 'Center' and elem.tag == 'ellipse':
            gt_c = np.array([float(elem.get('cx')), float(elem.get('cy'))])
        elif label == 'Tip' and elem.tag == 'ellipse':
            gt_t = np.array([float(elem.get('cx')), float(elem.get('cy'))])

    data = z.read(f'images/{name}')
    img = np.asarray(Image.open(io.BytesIO(data)).convert('L'), dtype=np.float32)

    ecx, ecy, erx, ery = float(gt_e[0]), float(gt_e[1]), float(gt_e[2]), float(gt_e[3])
    side = max(2.0*erx, 2.0*ery) * CROP_SCALE
    left, top_val = ecx - side/2.0, ecy - side/2.0

    left_i, top_i = int(left), int(top_val)
    right_i, bottom_i = int(left + side), int(top_val + side)
    left_i = max(0, min(639, left_i))
    top_i = max(0, min(639, top_i))
    right_i = max(1, min(640, right_i))
    bottom_i = max(1, min(640, bottom_i))

    crop = Image.fromarray(img.astype(np.uint8)).crop((left_i, top_i, right_i, bottom_i))
    crop160 = np.asarray(crop.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    xs = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / float(INPUT_SIZE) * side + left
    ys = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / float(INPUT_SIZE) * side + top_val
    xx, yy = np.meshgrid(xs, ys)
    mask = (((xx - ecx) / max(erx, 1.0))**2 + ((yy - ecy) / max(ery, 1.0))**2 <= 1.0).astype(np.float32)
    ct_input = np.stack([crop160 * 2.0 - 1.0, mask * 2.0 - 1.0], axis=-1).astype(np.float32)

    s_in, z_in = ct_in['quantization']
    t = np.clip(np.round(ct_input[None] / float(s_in) + float(z_in)), -128, 127).astype(np.int8)
    ct_interp.set_tensor(ct_in['index'], t)
    ct_interp.invoke()
    raw = ct_interp.get_tensor(ct_out['index']).astype(np.float32)
    s_out, z_out = ct_out['quantization']
    hm = (raw - float(z_out)) * float(s_out)

    decoded = []
    for ch in range(2):
        h = hm[0, ..., ch]
        y, x = np.unravel_index(np.argmax(h), h.shape)
        y0, y1 = max(0, y-4), min(80, y+5)
        x0, x1 = max(0, x-4), min(80, x+5)
        yyh, xxh = np.mgrid[y0:y1, x0:x1]
        w = np.maximum(h[y0:y1, x0:x1] - 0.03, 0)**2
        total = float(w.sum())
        if total > 0:
            pt = np.asarray([(xxh*w).sum()/total + 0.5, (yyh*w).sum()/total + 0.5], dtype=np.float32) / 80.0
        else:
            pt = np.asarray([(x+0.5)/80.0, (y+0.5)/80.0], dtype=np.float32)
        decoded.append(pt)
    decoded = np.asarray(decoded, dtype=np.float32)

    pred_c = np.array([left, top_val], dtype=np.float32) + decoded[0] * side
    pred_t = np.array([left, top_val], dtype=np.float32) + decoded[1] * side
    c_err = float(np.linalg.norm(pred_c - gt_c))
    t_err = float(np.linalg.norm(pred_t - gt_t))
    errors_c.append(c_err)
    errors_t.append(t_err)
    print(f'{name}: center={c_err:.1f}px  tip={t_err:.1f}px')

print()
ec = np.array(errors_c); et = np.array(errors_t)
print(f'Center ≤8px: {np.mean(ec<=8):.1%} ({int(np.sum(ec<=8))}/11)')
print(f'Tip ≤8px:    {np.mean(et<=8):.1%} ({int(np.sum(et<=8))}/11)')
print(f'Center mean: {ec.mean():.1f}px  median: {np.median(ec):.1f}px  max: {ec.max():.1f}px')
print(f'Tip mean:    {et.mean():.1f}px  median: {np.median(et):.1f}px  max: {et.max():.1f}px')
