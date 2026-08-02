"""Evaluate v9 center/tip on all test sets using the EXACT training pipeline.

Key: the training data PADS the image with black when the crop extends
beyond boundaries (not clips).  This preserves the ellipse-centered
coordinate system.  My earlier evaluation clipped, which broke it.
"""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CT_MODEL = ROOT / "artifacts" / "gauge_center_tip_littlegood_v9" / "gauge_center_tip_v8_int8.tflite"
LABELLED = ROOT / "data" / "labelled"
INPUT_SIZE = 160
HEATMAP_SIZE = 80
SOURCE_CROP_SCALE = 1.18  # matches prepare_gauge_center_tip_v1_dataset.py
MASK_CROP_SCALE = 1.35    # matches train_gauge_center_tip_v1.py


def load_ct_model():
    ct = tf.lite.Interpreter(model_path=str(CT_MODEL))
    ct.allocate_tensors()
    return ct, ct.get_input_details()[0], ct.get_output_details()[0]


def extract_gt(image_elem, w, h):
    center_xy = tip_xy = None
    for elem in image_elem:
        label = elem.get("label", "")
        if label in ("Center", "temp_center"):
            if elem.tag == "box":
                xtl, ytl = float(elem.get("xtl")), float(elem.get("ytl"))
                xbr, ybr = float(elem.get("xbr")), float(elem.get("ybr"))
                center_xy = np.array([(xtl + xbr) / 2, (ytl + ybr) / 2])
            elif elem.tag == "points":
                pts = elem.get("points", "").split(",")
                center_xy = np.array([float(pts[0]), float(pts[1])])
            elif elem.tag == "ellipse":
                center_xy = np.array([float(elem.get("cx")), float(elem.get("cy"))])
        elif label in ("Tip", "temp_tip"):
            if elem.tag == "box":
                xtl, ytl = float(elem.get("xtl")), float(elem.get("ytl"))
                xbr, ybr = float(elem.get("xbr")), float(elem.get("ybr"))
                tip_xy = np.array([(xtl + xbr) / 2, (ytl + ybr) / 2])
            elif elem.tag == "points":
                pts = elem.get("points", "").split(",")
                tip_xy = np.array([float(pts[0]), float(pts[1])])
            elif elem.tag == "ellipse":
                tip_xy = np.array([float(elem.get("cx")), float(elem.get("cy"))])
    return center_xy, tip_xy


def make_crop_with_padding(image_gray, cx, cy, rx, ry):
    """Crop with PAD when extending beyond image — matches training pipeline.

    Returns: (crop_160, left, top, side) where left/top are in padded coords.
    """
    w, h = image_gray.size
    side = max(2.0 * rx, 2.0 * ry) * SOURCE_CROP_SCALE
    left = cx - side / 2.0
    top = cy - side / 2.0

    # Pad if crop extends beyond image boundaries
    pad = int(math.ceil(max(0.0, -left, -top, left + side - w, top + side - h)))
    if pad:
        canvas = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
        canvas.paste(image_gray, (pad, pad))
        image_gray = canvas
        left += pad
        top += pad

    crop = image_gray.crop((
        int(round(left)), int(round(top)),
        int(round(left + side)), int(round(top + side)),
    ))
    crop_160 = np.asarray(crop.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR), np.float32) / 255.0
    return crop_160, left, top, side


def predict_center_tip(ct_interp, ct_in, ct_out, crop_160, cx, cy, rx, ry, left, top, side):
    """Run center/tip model on a pre-cropped 160x160 image."""
    # Map ellipse to crop coordinates
    cx_crop = (cx - left) / side * INPUT_SIZE
    cy_crop = (cy - top) / side * INPUT_SIZE
    rx_crop = rx / side * INPUT_SIZE
    ry_crop = ry / side * INPUT_SIZE

    # Draw mask with MASK_CROP_SCALE in crop space
    mask_side = max(2 * rx_crop, 2 * ry_crop) * MASK_CROP_SCALE
    xs = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / float(INPUT_SIZE) * mask_side + cx_crop - mask_side / 2
    ys = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / float(INPUT_SIZE) * mask_side + cy_crop - mask_side / 2
    xx, yy = np.meshgrid(xs, ys)
    mask = (((xx - cx_crop) / max(rx_crop, 1)) ** 2 + ((yy - cy_crop) / max(ry_crop, 1)) ** 2 <= 1).astype(np.float32)
    inp = np.stack([crop_160 * 2 - 1, mask * 2 - 1], axis=-1).astype(np.float32)

    s, zp = ct_in["quantization"]
    t = np.clip(np.round(inp[None] / float(s) + float(zp)), -128, 127).astype(np.int8)
    ct_interp.set_tensor(ct_in["index"], t)
    ct_interp.invoke()
    raw = ct_interp.get_tensor(ct_out["index"]).astype(np.float32)
    s, zp = ct_out["quantization"]
    hm = (raw - float(zp)) * float(s)

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

    # Map back to original image coordinates
    pred_c = np.array([left, top]) + decoded[0] * side
    pred_t = np.array([left, top]) + decoded[1] * side
    return pred_c, pred_t


def evaluate_zip(zip_name, ct_interp, ct_in, ct_out, max_images=100):
    zip_path = LABELLED / zip_name
    if not zip_path.exists():
        print(f"  {zip_name}: not found")
        return

    z = zipfile.ZipFile(zip_path)
    root = ET.parse(z.open("annotations.xml")).getroot()
    images = root.findall("image")
    if not images:
        print(f"  {zip_name}: no images")
        return

    indices = np.linspace(0, len(images) - 1, min(max_images, len(images)), dtype=int)
    errors_c, errors_t = [], []

    for idx in indices:
        img_elem = images[idx]
        name = img_elem.get("name")
        w = int(img_elem.get("width", "640"))
        h = int(img_elem.get("height", "640"))

        gt_c, gt_t = extract_gt(img_elem, w, h)
        if gt_c is None or gt_t is None:
            continue

        try:
            data = z.read(f"images/{name}")
            image = Image.open(io.BytesIO(data)).convert("L")
        except Exception:
            continue

        # Resize to 640x640 if needed
        if image.size != (640, 640):
            gt_c = gt_c * np.array([640 / w, 640 / h])
            gt_t = gt_t * np.array([640 / w, 640 / h])
            image = image.resize((640, 640), Image.Resampling.LANCZOS)

        # Use GT ellipse for crop (isolates center/tip model performance)
        # Find GaugeFace ellipse
        ellipse = None
        for elem in img_elem:
            if elem.get("label") == "GaugeFace" and elem.tag == "ellipse":
                ellipse = {
                    "cx": float(elem.get("cx")),
                    "cy": float(elem.get("cy")),
                    "rx": float(elem.get("rx")),
                    "ry": float(elem.get("ry")),
                }
                break

        if ellipse is None:
            # No ellipse annotation — use image center as fallback
            ellipse = {"cx": 320, "cy": 320, "rx": 200, "ry": 200}

        # Scale ellipse to 640px if needed
        if w != 640:
            for k in ("cx", "cy"):
                ellipse[k] *= 640.0 / w
            for k in ("rx", "ry"):
                ellipse[k] *= 640.0 / w

        # Make crop with padding (matches training pipeline)
        crop_160, left, top, side = make_crop_with_padding(
            image, ellipse["cx"], ellipse["cy"], ellipse["rx"], ellipse["ry"],
        )

        # Run center/tip model
        pred_c, pred_t = predict_center_tip(
            ct_interp, ct_in, ct_out, crop_160,
            ellipse["cx"], ellipse["cy"], ellipse["rx"], ellipse["ry"],
            left, top, side,
        )

        c_err = float(np.linalg.norm(pred_c - gt_c))
        t_err = float(np.linalg.norm(pred_t - gt_t))
        errors_c.append(c_err)
        errors_t.append(t_err)

    if not errors_c:
        print(f"  {zip_name}: no valid images")
        return

    ec = np.array(errors_c)
    et = np.array(errors_t)

    print(f"\n  {zip_name} ({len(ec)} images):")
    print(f"    Center <=8px:  {np.mean(ec <= 8):.1%} ({int(np.sum(ec <= 8))}/{len(ec)})")
    print(f"    Tip <=8px:     {np.mean(et <= 8):.1%} ({int(np.sum(et <= 8))}/{len(et)})")
    print(f"    Center mean:   {ec.mean():.1f}px  median: {np.median(ec):.1f}px  p90: {np.percentile(ec, 90):.1f}px")
    print(f"    Tip mean:      {et.mean():.1f}px  median: {np.median(et):.1f}px  p90: {np.percentile(et, 90):.1f}px")


def main():
    print("Loading models...")
    ct_interp, ct_in, ct_out = load_ct_model()

    print("\nv9 model + GT ellipse + PAD pipeline (matches training):")
    print("=" * 60)
    for zip_name in ["test_1.zip", "test_2.zip", "test_3.zip"]:
        evaluate_zip(zip_name, ct_interp, ct_in, ct_out, max_images=1000)


if __name__ == "__main__":
    main()
