#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

python3 -u - <<'EOF'
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "src")
import tensorflow as tf
from embedded_gauge_reading_tinyml.board_crop_compare import load_yuv422_capture_as_rgb, resize_with_pad_rgb

RECTIFIER_KERAS = Path("artifacts/training/mobilenetv2_rectifier_zoom_aug_v4/model.keras")
SCALAR_TFLITE   = Path("artifacts/deployment/prod_model_v0.2_raw_int8/model_int8.tflite")
CAPTURES_DIR    = Path("../captured_images")
CROP_SCALE      = 1.25
IMAGE_SIZE      = 224

print(f"Loading rectifier: {RECTIFIER_KERAS}")
rectifier = tf.keras.models.load_model(str(RECTIFIER_KERAS), compile=False)

print(f"Loading scalar: {SCALAR_TFLITE}")
interp = tf.lite.Interpreter(model_path=str(SCALAR_TFLITE))
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

# pick recent captures
yuv_files = sorted(CAPTURES_DIR.glob("capture_2026*.yuv422"))[-8:]
if not yuv_files:
    yuv_files = sorted(CAPTURES_DIR.glob("*.yuv422"))[-8:]

for yuv_path in yuv_files:
    rgb = load_yuv422_capture_as_rgb(yuv_path, width=IMAGE_SIZE, height=IMAGE_SIZE)
    # rectifier: full frame -> box
    rect_input = rgb.astype(np.float32)[None]  # (1,224,224,3)
    raw_box = rectifier.predict(rect_input, verbose=0)
    if isinstance(raw_box, dict):
        box = np.array(list(raw_box.values())[0]).reshape(-1)
    else:
        box = np.array(raw_box).reshape(-1)
    cx, cy, bw, bh = box[0], box[1], box[2], box[3]
    bw_s = min(1.0, float(np.clip(bw, 0.05, 1.0)) * CROP_SCALE)
    bh_s = min(1.0, float(np.clip(bh, 0.05, 1.0)) * CROP_SCALE)
    x_min = int(max(0, (cx - bw_s/2) * IMAGE_SIZE))
    y_min = int(max(0, (cy - bh_s/2) * IMAGE_SIZE))
    x_max = int(min(IMAGE_SIZE, (cx + bw_s/2) * IMAGE_SIZE))
    y_max = int(min(IMAGE_SIZE, (cy + bh_s/2) * IMAGE_SIZE))
    crop_w, crop_h = x_max - x_min, y_max - y_min

    # crop and resize for scalar
    crop = rgb[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        print(f"{yuv_path.name}: EMPTY CROP box=({cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f})")
        continue
    resized = resize_with_pad_rgb(crop, IMAGE_SIZE, IMAGE_SIZE)
    scalar_input = resized.astype(np.float32)[None]

    # quantize
    scale = inp_det['quantization_parameters']['scales'][0]
    zp    = inp_det['quantization_parameters']['zero_points'][0]
    q = np.clip(np.round(scalar_input / scale + zp), -128, 127).astype(np.int8)
    interp.set_tensor(inp_det['index'], q)
    interp.invoke()
    raw = interp.get_tensor(out_det['index'])
    out_scale = out_det['quantization_parameters']['scales'][0]
    out_zp    = out_det['quantization_parameters']['zero_points'][0]
    pred = float((raw.flat[0] - out_zp) * out_scale)

    print(f"{yuv_path.name}: box=({cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f}) "
          f"crop=x{x_min}y{y_min}w{crop_w}h{crop_h} pred={pred:.2f}C")

EOF
