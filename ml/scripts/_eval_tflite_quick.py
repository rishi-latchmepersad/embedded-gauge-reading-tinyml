"""Evaluate TFLite scalar model on hard cases manifest."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

TRAINING_CROP_X_MIN = 0.1027
TRAINING_CROP_Y_MIN = 0.2573
TRAINING_CROP_X_MAX = 0.7987
TRAINING_CROP_Y_MAX = 0.8071
IMAGE_SIZE = 224


def load_rgb_png(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img, dtype=np.uint8)


def crop_and_resize(img_hw3: np.ndarray) -> np.ndarray:
    h, w = img_hw3.shape[:2]
    x0, x1 = int(TRAINING_CROP_X_MIN * w), int(TRAINING_CROP_X_MAX * w)
    y0, y1 = int(TRAINING_CROP_Y_MIN * h), int(TRAINING_CROP_Y_MAX * h)
    crop = img_hw3[y0:y1, x0:x1]
    rgb = crop.astype(np.float32) / 255.0
    return tf.image.resize_with_pad(rgb, IMAGE_SIZE, IMAGE_SIZE).numpy()


def main():
    model_path = Path("artifacts/deployment/prod_model_v0.2_raw_int8/model_int8.tflite")
    manifest_path = Path("data/hard_cases_plus_board30_valid_with_new6.csv")

    print(f"Loading TFLite model from {model_path}...", flush=True)
    interp = tf.lite.Interpreter(model_path=str(model_path), num_threads=2)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    print(f"Input: {inp['shape']} quant={inp['quantization']}", flush=True)
    print(f"Output: {out['shape']} quant={out['quantization']}", flush=True)

    rows = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            p = Path(row["image_path"])
            if not p.is_absolute():
                p = REPO_ROOT / p
            rows.append((p, float(row["value"])))

    print(f"Manifest: {manifest_path.name}  ({len(rows)} images)\n", flush=True)

    errors = []
    for path, true_val in rows:
        if not path.exists() or path.stat().st_size == 0:
            print(f"  SKIP: {path.name}", flush=True)
            continue
        img = load_rgb_png(path)
        inp_img = crop_and_resize(img)
        q = np.clip(np.round(inp_img / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None])
        interp.invoke()
        raw = interp.get_tensor(out["index"]).flatten()[0]
        pred = (raw.astype(np.float32) - out_zp) * out_scale
        err = pred - true_val
        errors.append((true_val, pred, err))
        print(
            f"  true={true_val:6.1f}  pred={pred:7.2f}  err={err:+.2f}  {path.name}",
            flush=True,
        )

    if errors:
        arr = np.array([e[2] for e in errors])
        print(
            f"\nn={len(arr)}  MAE={np.abs(arr).mean():.2f}  bias={arr.mean():.2f}  std={arr.std():.2f}",
            flush=True,
        )
        print(f"Cases > 5C: {sum(1 for e in errors if abs(e[2]) > 5)}", flush=True)
        errors.sort(key=lambda x: -abs(x[2]))
        print("\nWorst failures:", flush=True)
        for e in errors[:10]:
            print(f"  err={e[2]:+.2f}  true={e[0]:6.1f}  pred={e[1]:7.2f}", flush=True)


if __name__ == "__main__":
    main()
