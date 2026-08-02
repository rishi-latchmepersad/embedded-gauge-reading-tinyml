"""Evaluate full pipeline with adaptive crop scales on all test sets."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ELLIPSE_MODEL = ROOT / "artifacts" / "gauge_ellipse_qat_linear_v1" / "ellipse_qat_linear_int8.tflite"
CT_MODEL = ROOT / "artifacts" / "gauge_center_tip_adaptive_v2" / "gauge_center_tip_adaptive_v1_int8.tflite"
LABELLED = ROOT / "data" / "labelled"
INPUT_SIZE = 160
ELL_INPUT_SIZE = 224
TARGET_FILL = 0.65


def adaptive_crop_scale(gauge_fill: float) -> float:
    if gauge_fill <= 0:
        return 1.35
    ideal = 1.0 / TARGET_FILL
    if gauge_fill > 0.80:
        return min(ideal, 0.95 / gauge_fill)
    elif gauge_fill > 0.40:
        return ideal
    else:
        return min(ideal, 1.6)


def load_models():
    ell = tf.lite.Interpreter(model_path=str(ELLIPSE_MODEL))
    ell.allocate_tensors()
    ct = tf.lite.Interpreter(model_path=str(CT_MODEL))
    ct.allocate_tensors()
    return (
        ell, ell.get_input_details()[0], ell.get_output_details(),
        ct, ct.get_input_details()[0], ct.get_output_details()[0],
    )


def predict_ellipse(ell_interp, ell_in, ell_outs, image_640):
    img_224 = np.asarray(Image.fromarray(image_640.astype(np.uint8)).convert("L").resize((224, 224)), np.float32) / 255.0
    s, zp = ell_in["quantization"]
    t = np.clip(np.round(img_224[None, ..., None] / float(s) + float(zp)), -128, 127).astype(np.int8)
    ell_interp.set_tensor(ell_in["index"], t)
    ell_interp.invoke()
    preds = {}
    for od in ell_outs:
        raw = ell_interp.get_tensor(od["index"]).astype(np.float32)
        s, zp = od["quantization"]
        preds[od["name"]] = (raw - float(zp)) * float(s)
    center = preds["StatefulPartitionedCall:0"][0]
    radius = preds["StatefulPartitionedCall:2"][0]
    return center, radius


def predict_keypoints(ct_interp, ct_in, ct_out, image_640, ell_center, ell_radius):
    ecx, ecy = ell_center * 640
    erx, ery = ell_radius * 640
    gauge_radius = max(erx, ery)
    gauge_fill = 2.0 * gauge_radius / 640.0
    crop_scale = adaptive_crop_scale(gauge_fill)
    side = 2.0 * gauge_radius * crop_scale
    max_side = 640 * 0.98
    if side > max_side:
        side = max_side

    left, top = ecx - side / 2, ecy - side / 2
    if left < 0: left = 0
    if top < 0: top = 0
    if left + side > 640: left = 640 - side
    if top + side > 640: top = 640 - side

    left_i, top_i = int(max(0, left)), int(max(0, top))
    right_i = int(min(640, left + side))
    bottom_i = int(min(640, top + side))
    crop = Image.fromarray(image_640.astype(np.uint8)).crop((left_i, top_i, right_i, bottom_i))
    crop160 = np.asarray(crop.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR), np.float32) / 255.0

    xs = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / float(INPUT_SIZE) * side + left
    ys = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / float(INPUT_SIZE) * side + top
    xx, yy = np.meshgrid(xs, ys)
    mask = (((xx - ecx) / max(erx, 1)) ** 2 + ((yy - ecy) / max(ery, 1)) ** 2 <= 1).astype(np.float32)
    ct_input = np.stack([crop160 * 2 - 1, mask * 2 - 1], axis=-1).astype(np.float32)

    s, zp = ct_in["quantization"]
    t = np.clip(np.round(ct_input[None] / float(s) + float(zp)), -128, 127).astype(np.int8)
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

    pred_center = np.array([left, top]) + decoded[0] * side
    pred_tip = np.array([left, top]) + decoded[1] * side
    return pred_center, pred_tip, crop_scale


def extract_gt(image_elem, w, h):
    center_xy = tip_xy = None
    for elem in image_elem:
        label = elem.get("label", "")
        if label in ("Center", "temp_center"):
            if elem.tag == "ellipse":
                center_xy = np.array([float(elem.get("cx")), float(elem.get("cy"))])
            elif elem.tag == "points":
                pts = elem.get("points", "").split(",")
                center_xy = np.array([float(pts[0]), float(pts[1])])
            elif elem.tag == "box":
                xtl, ytl = float(elem.get("xtl")), float(elem.get("ytl"))
                xbr, ybr = float(elem.get("xbr")), float(elem.get("ybr"))
                center_xy = np.array([(xtl + xbr) / 2, (ytl + ybr) / 2])
        elif label in ("Tip", "temp_tip"):
            if elem.tag == "ellipse":
                tip_xy = np.array([float(elem.get("cx")), float(elem.get("cy"))])
            elif elem.tag == "points":
                pts = elem.get("points", "").split(",")
                tip_xy = np.array([float(pts[0]), float(pts[1])])
            elif elem.tag == "box":
                xtl, ytl = float(elem.get("xtl")), float(elem.get("ytl"))
                xbr, ybr = float(elem.get("xbr")), float(elem.get("ybr"))
                tip_xy = np.array([(xtl + xbr) / 2, (ytl + ybr) / 2])
    return center_xy, tip_xy


def evaluate_zip(zip_name, ell_interp, ell_in, ell_outs, ct_interp, ct_in, ct_out, max_images=50):
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
    errors_c, errors_t, crop_scales = [], [], []

    for idx in indices:
        img_elem = images[idx]
        name = img_elem.get("name")
        w = int(img_elem.get("width", "640"))
        h = int(img_elem.get("height", "640"))

        gt_center, gt_tip = extract_gt(img_elem, w, h)
        if gt_center is None or gt_tip is None:
            continue

        try:
            data = z.read(f"images/{name}")
            image = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32)
        except Exception:
            continue

        if image.shape[0] != 640 or image.shape[1] != 640:
            image = np.asarray(Image.fromarray(image.astype(np.uint8)).resize((640, 640)), np.float32)
            gt_center = gt_center * np.array([640 / w, 640 / h])
            gt_tip = gt_tip * np.array([640 / w, 640 / h])

        ell_center, ell_radius = predict_ellipse(ell_interp, ell_in, ell_outs, image)
        pred_center, pred_tip, scale = predict_keypoints(
            ct_interp, ct_in, ct_out, image, ell_center, ell_radius,
        )

        c_err = float(np.linalg.norm(pred_center - gt_center))
        t_err = float(np.linalg.norm(pred_tip - gt_tip))
        errors_c.append(c_err)
        errors_t.append(t_err)
        crop_scales.append(scale)

    if not errors_c:
        print(f"  {zip_name}: no valid images")
        return

    ec = np.array(errors_c)
    et = np.array(errors_t)
    cs = np.array(crop_scales)

    print(f"\n  {zip_name} ({len(ec)} images):")
    print(f"    Center <=8px:  {np.mean(ec <= 8):.1%} ({int(np.sum(ec <= 8))}/{len(ec)})")
    print(f"    Tip <=8px:     {np.mean(et <= 8):.1%} ({int(np.sum(et <= 8))}/{len(et)})")
    print(f"    Center mean:   {ec.mean():.1f}px  median: {np.median(ec):.1f}px  p90: {np.percentile(ec, 90):.1f}px")
    print(f"    Tip mean:      {et.mean():.1f}px  median: {np.median(et):.1f}px  p90: {np.percentile(et, 90):.1f}px")
    print(f"    Crop scale:    min={cs.min():.2f} max={cs.max():.2f} mean={cs.mean():.2f}")


def main():
    print("Loading models...")
    ell_interp, ell_in, ell_outs, ct_interp, ct_in, ct_out = load_models()

    print("\nFull pipeline (adaptive crop scale):")
    print("=" * 60)
    for zip_name in ["test_1.zip", "test_2.zip", "test_3.zip"]:
        evaluate_zip(zip_name, ell_interp, ell_in, ell_outs, ct_interp, ct_in, ct_out, max_images=50)


if __name__ == "__main__":
    main()
