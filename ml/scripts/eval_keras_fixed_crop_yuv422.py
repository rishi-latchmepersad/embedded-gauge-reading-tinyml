"""Evaluate a Keras scalar model on YUV422 captures using the fixed training crop.
Useful for comparing Keras checkpoints before quantization.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

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


def crop_and_resize(luma: np.ndarray, x0r, y0r, x1r, y1r) -> np.ndarray:
    x0 = int(x0r * IMAGE_SIZE)
    y0 = int(y0r * IMAGE_SIZE)
    x1 = int(x1r * IMAGE_SIZE)
    y1 = int(y1r * IMAGE_SIZE)
    crop = luma[y0:y1, x0:x1]
    rgb = np.repeat(crop[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    return tf.image.resize_with_pad(rgb, IMAGE_SIZE, IMAGE_SIZE).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--captures-dir", type=Path, default=REPO_ROOT / "data" / "captured" / "images"
    )
    parser.add_argument("--pattern", type=str, default="capture_2026-04-18_17-*.yuv422")
    parser.add_argument("--true-value", type=float, default=33.0)
    args = parser.parse_args()

    model = tf.keras.models.load_model(
        str(args.model),
        custom_objects={
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
        },
        compile=False,
    )

    captures = sorted(args.captures_dir.glob(args.pattern))
    captures = [c for c in captures if c.stat().st_size > 0]
    print(f"Model: {args.model.name}")
    print(f"True value: {args.true_value}°C  |  {len(captures)} captures\n")

    preds = []
    for cap in captures:
        luma = load_luma(cap)
        img = crop_and_resize(
            luma,
            TRAINING_CROP_X_MIN,
            TRAINING_CROP_Y_MIN,
            TRAINING_CROP_X_MAX,
            TRAINING_CROP_Y_MAX,
        )
        result = model.predict(img[None], verbose=0)
        pred = float(np.asarray(result).flatten()[0])
        preds.append(pred)
        print(f"  {cap.name}:  pred={pred:7.2f}  err={pred - args.true_value:+.2f}")

    arr = np.array(preds)
    print(
        f"\nMean={arr.mean():.2f}  Std={arr.std():.2f}  MAE={np.abs(arr - args.true_value).mean():.2f}  (true={args.true_value})"
    )


if __name__ == "__main__":
    main()
