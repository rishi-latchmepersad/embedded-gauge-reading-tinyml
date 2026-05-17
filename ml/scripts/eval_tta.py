"""Evaluate a Keras scalar model with Test-Time Augmentation (TTA).

Applies mild augmentations at inference time and averages predictions.
Can squeeze 5-15%% improvement without retraining.
"""
from __future__ import annotations
import argparse
import csv
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


def load_luma_yuv422(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE // 2, 4)
    luma = np.empty((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    luma[:, 0::2] = yuyv[:, :, 0]
    luma[:, 1::2] = yuyv[:, :, 2]
    return luma


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


def tta_augment(img: tf.Tensor) -> list[tf.Tensor]:
    """Generate mild test-time augmentations of a single image tensor.
    
    Returns list of augmented images to predict on.
    """
    imgs = [img]
    # horizontal flip
    imgs.append(tf.image.flip_left_right(img))
    # brightness +/- 0.08
    imgs.append(tf.image.adjust_brightness(img, 0.08))
    imgs.append(tf.image.adjust_brightness(img, -0.08))
    # contrast +/- 0.1
    imgs.append(tf.image.adjust_contrast(img, 1.1))
    imgs.append(tf.image.adjust_contrast(img, 0.9))
    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "data/hard_cases_plus_board30_valid_with_new6.csv")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--tta", action="store_true", default=True, help="Enable TTA (default)")
    parser.add_argument("--no-tta", dest="tta", action="store_false", help="Disable TTA")
    args = parser.parse_args()

    model = tf.keras.models.load_model(
        str(args.model),
        custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
        compile=False,
    )

    rows = []
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            img_col = next((k for k in row if "image" in k.lower() or "path" in k.lower() or "file" in k.lower()), list(row.keys())[0])
            val_col = next((k for k in row if "value" in k.lower() or "label" in k.lower() or "temp" in k.lower()), list(row.keys())[1])
            p = Path(row[img_col])
            if not p.is_absolute():
                p = args.repo_root / p
            rows.append((p, float(row[val_col])))

    print(f"Model: {args.model.name}")
    print(f"Manifest: {args.manifest.name}  ({len(rows)} images)")
    print(f"TTA: {args.tta}\n")

    errors = []
    tta_errors = []
    for path, true_val in rows:
        if not path.exists() or path.stat().st_size == 0:
            continue
        if path.suffix == ".yuv422":
            luma = load_luma_yuv422(path)
            img = np.repeat(luma[:, :, None], 3, axis=2)
        else:
            img = load_rgb_png(path)
        inp = crop_and_resize(img)
        
        if args.tta:
            aug_imgs = tta_augment(tf.convert_to_tensor(inp))
            batch = tf.stack(aug_imgs)
            preds = model.predict(batch, verbose=0)
            pred = float(np.mean(preds))
        else:
            pred = float(np.asarray(model.predict(inp[None], verbose=0)).flatten()[0])
        
        err = pred - true_val
        errors.append((true_val, pred, err))
        print(f"  true={true_val:6.1f}  pred={pred:7.2f}  err={err:+.2f}  {path.name}")

    arr = np.array([e[2] for e in errors])
    print(f"\nn={len(arr)}  MAE={np.abs(arr).mean():.2f}  bias={arr.mean():.2f}  std={arr.std():.2f}")


if __name__ == "__main__":
    main()
