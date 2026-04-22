import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "src")
import tensorflow as tf
from embedded_gauge_reading_tinyml.board_crop_compare import (
    load_yuv422_capture_as_rgb,
    resize_with_pad_rgb,
)

SCALAR_TFLITE   = Path("artifacts/deployment/prod_model_v0.2_raw_int8/model_int8.tflite")
RECTIFIER_KERAS = Path("artifacts/training/mobilenetv2_rectifier_zoom_aug_v4/model.keras")
CAPTURES_DIR    = Path("../captured_images")
IMAGE_SIZE = 224
CROP_SCALE = 1.25

# Fixed training crop from app_ai.c APP_AI_TRAINING_CROP_*_RATIO
TX0 = int(0.103 * IMAGE_SIZE)
TY0 = int(0.254 * IMAGE_SIZE)
TX1 = int(0.794 * IMAGE_SIZE)
TY1 = int(0.803 * IMAGE_SIZE)
print(f"Fixed training crop: x={TX0} y={TY0} w={TX1-TX0} h={TY1-TY0}")

interp = tf.lite.Interpreter(model_path=str(SCALAR_TFLITE))
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

print("Loading rectifier...")
rectifier = tf.keras.models.load_model(str(RECTIFIER_KERAS), compile=False)
print("Ready.")

captures = sorted(CAPTURES_DIR.glob("capture_2026-04-20_16-*.yuv422"))[:8]
if not captures:
    captures = sorted(CAPTURES_DIR.glob("*.yuv422"))[-8:]


def run_scalar(rgb_full, x0, y0, x1, y1):
    crop_box = (float(x0), float(y0), float(x1), float(y1))
    resized = resize_with_pad_rgb(rgb_full, crop_box, IMAGE_SIZE)
    s  = inp_det['quantization_parameters']['scales'][0]
    zp = inp_det['quantization_parameters']['zero_points'][0]
    q  = np.clip(np.round(resized.astype(np.float32) / s + zp), -128, 127).astype(np.int8)[None]
    interp.set_tensor(inp_det['index'], q)
    interp.invoke()
    raw = interp.get_tensor(out_det['index'])
    os_  = out_det['quantization_parameters']['scales'][0]
    ozp  = out_det['quantization_parameters']['zero_points'][0]
    return float((raw.flat[0] - ozp) * os_)


for yuv_path in captures:
    rgb = load_yuv422_capture_as_rgb(yuv_path, image_width=IMAGE_SIZE, image_height=IMAGE_SIZE)

    pred_fixed = run_scalar(rgb, TX0, TY0, TX1, TY1)

    raw_pred = rectifier.predict(rgb.astype(np.float32)[None], verbose=0)
    if isinstance(raw_pred, dict):
        raw_pred = list(raw_pred.values())[0]
    box = np.array(raw_pred).reshape(-1)
    cx, cy = float(box[0]), float(box[1])
    bw = min(1.0, float(np.clip(box[2], 0.05, 1.0)) * CROP_SCALE)
    bh = min(1.0, float(np.clip(box[3], 0.05, 1.0)) * CROP_SCALE)
    rx0 = int(max(0, (cx - bw / 2) * IMAGE_SIZE))
    ry0 = int(max(0, (cy - bh / 2) * IMAGE_SIZE))
    rx1 = int(min(IMAGE_SIZE, (cx + bw / 2) * IMAGE_SIZE))
    ry1 = int(min(IMAGE_SIZE, (cy + bh / 2) * IMAGE_SIZE))
    pred_rect = run_scalar(rgb, rx0, ry0, rx1, ry1) if (rx1 > rx0 and ry1 > ry0) else float('nan')

    print(f"{yuv_path.name}: fixed={pred_fixed:.1f}C  "
          f"rect_box=(x{rx0}y{ry0}w{rx1-rx0}h{ry1-ry0}) rect={pred_rect:.1f}C")
