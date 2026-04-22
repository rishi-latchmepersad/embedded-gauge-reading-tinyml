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

SCALAR_TFLITE = Path("artifacts/deployment/prod_model_v0.2_raw_int8/model_int8.tflite")
RECTIFIER_KERAS = Path("artifacts/training/mobilenetv2_rectifier_zoom_aug_v4/model.keras")
CAPTURES_DIR  = Path("../captured_images")
IMAGE_SIZE    = 224
CROP_SCALE    = 1.25

# Fixed training crop ratios (from app_ai.c)
# APP_AI_TRAINING_CROP_X_MIN_RATIO 0.103f, X_MAX 0.794f, Y_MIN 0.254f, Y_MAX 0.803f
TX0 = int(0.103 * IMAGE_SIZE)
TY0 = int(0.254 * IMAGE_SIZE)
TW  = int((0.794 - 0.103) * IMAGE_SIZE)
TH  = int((0.803 - 0.254) * IMAGE_SIZE)
print(f"Fixed training crop: x={TX0} y={TY0} w={TW} h={TH}")

interp = tf.lite.Interpreter(model_path=str(SCALAR_TFLITE))
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

rectifier = tf.keras.models.load_model(str(RECTIFIER_KERAS), compile=False)

# today's captures at 34C
captures = sorted(CAPTURES_DIR.glob("capture_2026-04-20_16-*.yuv422"))[:6]
if not captures:
    captures = sorted(CAPTURES_DIR.glob("*.yuv422"))[-6:]

def run_scalar(rgb_crop):
    resized = resize_with_pad_rgb(rgb_crop, IMAGE_SIZE, IMAGE_SIZE)
    s = inp_det['quantization_parameters']['scales'][0]
    zp = inp_det['quantization_parameters']['zero_points'][0]
    q = np.clip(np.round(resized.astype(np.float32) / s + zp), -128, 127).astype(np.int8)[None]
    interp.set_tensor(inp_det['index'], q)
    interp.invoke()
    raw = interp.get_tensor(out_det['index'])
    os = out_det['quantization_parameters']['scales'][0]
    ozp = out_det['quantization_parameters']['zero_points'][0]
    return float((raw.flat[0] - ozp) * os)

for yuv_path in captures:
    rgb = load_yuv422_capture_as_rgb(yuv_path, width=IMAGE_SIZE, height=IMAGE_SIZE)

    # 1. fixed training crop
    fixed_crop = rgb[TY0:TY0+TH, TX0:TX0+TW]
    pred_fixed = run_scalar(fixed_crop)

    # 2. rectifier crop
    rect_in = rgb.astype(np.float32)[None]
    raw_box = rectifier.predict(rect_in, verbose=0)
    box = np.array(raw_box).reshape(-1)
    cx, cy, bw, bh = box[0], box[1], min(1.0, box[2]*CROP_SCALE), min(1.0, box[3]*CROP_SCALE)
    x0 = int(max(0, (cx - bw/2) * IMAGE_SIZE))
    y0 = int(max(0, (cy - bh/2) * IMAGE_SIZE))
    x1 = int(min(IMAGE_SIZE, (cx + bw/2) * IMAGE_SIZE))
    y1 = int(min(IMAGE_SIZE, (cy + bh/2) * IMAGE_SIZE))
    rect_crop = rgb[y0:y1, x0:x1]
    pred_rect = run_scalar(rect_crop) if rect_crop.size > 0 else float('nan')

    print(f"{yuv_path.name}: fixed={pred_fixed:.1f}C  rect_box=(x{x0}y{y0}w{x1-x0}h{y1-y0}) rect={pred_rect:.1f}C")

EOF
