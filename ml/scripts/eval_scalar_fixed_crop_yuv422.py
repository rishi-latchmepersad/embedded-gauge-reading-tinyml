"""Evaluate the scalar TFLite model on raw YUV422 captures using the fixed training crop.

Mirrors the firmware's fixed-crop path exactly (luma replicated to 3 channels,
training crop ratios from app_ai.c) so we can isolate whether the scalar model
itself is correct on close-up frames before blaming the rectifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent

TRAINING_CROP_X_MIN = 0.1027
TRAINING_CROP_Y_MIN = 0.2573
TRAINING_CROP_X_MAX = 0.7987
TRAINING_CROP_Y_MAX = 0.8071
IMAGE_SIZE = 224


def load_luma(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE // 2, 4)
    luma = np.empty((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    luma[:, 0::2] = yuyv[:, :, 0]
    luma[:, 1::2] = yuyv[:, :, 2]
    return luma


def crop_and_resize(luma: np.ndarray, x0r: float, y0r: float, x1r: float, y1r: float) -> np.ndarray:
    x0 = int(x0r * IMAGE_SIZE)
    y0 = int(y0r * IMAGE_SIZE)
    x1 = int(x1r * IMAGE_SIZE)
    y1 = int(y1r * IMAGE_SIZE)
    crop = luma[y0:y1, x0:x1]
    rgb = np.repeat(crop[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    return tf.image.resize_with_pad(rgb, IMAGE_SIZE, IMAGE_SIZE).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "artifacts/deployment/scalar_full_finetune_from_best_piecewise_calibrated_int8/model_int8.tflite")
    parser.add_argument("--captures-dir", type=Path, default=REPO_ROOT / "captured_images")
    parser.add_argument("--pattern", type=str, default="capture_2026-04-18*.yuv422")
    parser.add_argument("--true-value", type=float, default=14.0)
    args = parser.parse_args()

    interp = tf.lite.Interpreter(model_path=str(args.model), num_threads=2)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    captures = sorted(args.captures_dir.glob(args.pattern))
    if not captures:
        print(f"No captures found matching {args.captures_dir / args.pattern}")
        return

    print(f"Scalar model: {args.model.name}")
    print(f"Fixed training crop: x=[{TRAINING_CROP_X_MIN},{TRAINING_CROP_X_MAX}] y=[{TRAINING_CROP_Y_MIN},{TRAINING_CROP_Y_MAX}]")
    print(f"True value: {args.true_value}°C  |  {len(captures)} captures\n")

    preds = []
    for cap in captures:
        luma = load_luma(cap)
        img = crop_and_resize(luma, TRAINING_CROP_X_MIN, TRAINING_CROP_Y_MIN, TRAINING_CROP_X_MAX, TRAINING_CROP_Y_MAX)
        q = np.clip(np.round(img / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None])
        interp.invoke()
        raw = int(interp.get_tensor(out["index"])[0][0])
        pred = float(out_scale * (raw - out_zp))
        preds.append(pred)
        print(f"  {cap.name}:  pred={pred:7.2f}  err={pred - args.true_value:+.2f}")

    preds_arr = np.array(preds)
    print(f"\nMean={preds_arr.mean():.2f}  Std={preds_arr.std():.2f}  MAE={np.abs(preds_arr - args.true_value).mean():.2f}  (true={args.true_value})")


if __name__ == "__main__":
    main()
