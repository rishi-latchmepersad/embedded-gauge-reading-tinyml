"""Evaluate full pipeline on all labelled test sets.

Tests the linear radius ellipse + center/tip model on:
  - test_1.zip: 915 images with Center/End/GaugeFace/Start/Tip annotations
  - test_2.zip: 11 augmented images
  - test_3.zip: 11 board captures

Uses ground truth ellipse for center/tip evaluation (to isolate center/tip
model performance), and also tests the full pipeline with predicted ellipse.
"""

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
CT_MODEL = ROOT / "artifacts" / "gauge_center_tip_universal_v1" / "gauge_center_tip_full_v1_int8.tflite"
LABELLED = ROOT / "data" / "labelled"
INPUT_SIZE = 160
HEATMAP_SIZE = 80
CROP_SCALE = 1.35
ELL_INPUT_SIZE = 224


def load_models():
    """Load both int8 models."""
    ell = tf.lite.Interpreter(model_path=str(ELLIPSE_MODEL))
    ell.allocate_tensors()
    ell_in = ell.get_input_details()[0]
    ell_outs = ell.get_output_details()

    ct = tf.lite.Interpreter(model_path=str(CT_MODEL))
    ct.allocate_tensors()
    ct_in = ct.get_input_details()[0]
    ct_out = ct.get_output_details()[0]

    return ell, ell_in, ell_outs, ct, ct_in, ct_out


def predict_ellipse(ell_interp, ell_in, ell_outs, image_640: np.ndarray) -> tuple:
    """Predict ellipse from 640x640 grayscale image."""
    img_224 = np.asarray(
        Image.fromarray(image_640.astype(np.uint8)).resize((224, 224)),
        dtype=np.float32,
    ) / 255.0
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


def predict_keypoints(ct_interp, ct_in, ct_out, image_640: np.ndarray,
                      ellipse_center: np.ndarray, ellipse_radius: np.ndarray) -> tuple:
    """Predict center/tip keypoints given an ellipse."""
    ecx, ecy = ellipse_center * 640
    erx, ery = ellipse_radius * 640
    side = max(2 * erx, 2 * ery) * CROP_SCALE
    left, top = ecx - side / 2, ecy - side / 2

    left_i, top_i = int(max(0, left)), int(max(0, top))
    right_i = int(min(640, left + side))
    bottom_i = int(min(640, top + side))
    crop = Image.fromarray(image_640.astype(np.uint8)).crop((left_i, top_i, right_i, bottom_i))
    crop160 = np.asarray(crop.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

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
    return pred_center, pred_tip


def extract_gt_from_cvat(image_elem, image_w: int, image_h: int) -> tuple:
    """Extract ground truth center and tip from CVAT annotations.

    Handles three annotation types:
    - ellipse: use center (cx, cy)
    - points: use the point coordinates
    - box: use the center of the bounding box ((xtl+xbr)/2, (ytl+ybr)/2)
    """
    center_xy = None
    tip_xy = None
    ellipse_params = None

    for elem in image_elem:
        label = elem.get("label", "")

        if label in ("Center", "temp_center"):
            if elem.tag == "ellipse":
                cx, cy = float(elem.get("cx")), float(elem.get("cy"))
                center_xy = np.array([cx, cy])
            elif elem.tag == "points":
                pts = elem.get("points", "").split(",")
                center_xy = np.array([float(pts[0]), float(pts[1])])
            elif elem.tag == "box":
                xtl, ytl = float(elem.get("xtl")), float(elem.get("ytl"))
                xbr, ybr = float(elem.get("xbr")), float(elem.get("ybr"))
                center_xy = np.array([(xtl + xbr) / 2, (ytl + ybr) / 2])

        elif label in ("Tip", "temp_tip"):
            if elem.tag == "ellipse":
                cx, cy = float(elem.get("cx")), float(elem.get("cy"))
                tip_xy = np.array([cx, cy])
            elif elem.tag == "points":
                pts = elem.get("points", "").split(",")
                tip_xy = np.array([float(pts[0]), float(pts[1])])
            elif elem.tag == "box":
                xtl, ytl = float(elem.get("xtl")), float(elem.get("ytl"))
                xbr, ybr = float(elem.get("xbr")), float(elem.get("ybr"))
                tip_xy = np.array([(xtl + xbr) / 2, (ytl + ybr) / 2])

        elif label in ("GaugeFace", "temp_dial"):
            if elem.tag == "ellipse":
                ellipse_params = {
                    "cx": float(elem.get("cx")),
                    "cy": float(elem.get("cy")),
                    "rx": float(elem.get("rx")),
                    "ry": float(elem.get("ry")),
                }

    return center_xy, tip_xy, ellipse_params


def evaluate_zip(
    zip_name: str,
    ell_interp,
    ell_in,
    ell_outs,
    ct_interp,
    ct_in,
    ct_out,
    max_images: int = 50,
    use_ground_truth_ellipse: bool = False,
):
    """Evaluate full pipeline on images from one zip file."""
    zip_path = LABELLED / zip_name
    if not zip_path.exists():
        print(f"  {zip_name}: not found, skipping")
        return

    z = zipfile.ZipFile(zip_path)
    tree = ET.parse(z.open("annotations.xml"))
    root = tree.getroot()
    images = root.findall("image")

    if not images:
        print(f"  {zip_name}: no images")
        return

    # Sample evenly across the dataset
    indices = np.linspace(0, len(images) - 1, min(max_images, len(images)), dtype=int)

    errors_c, errors_t = [], []
    ellipse_errors = []
    count = 0

    for idx in indices:
        img_elem = images[idx]
        name = img_elem.get("name")
        w = int(img_elem.get("width", "640"))
        h = int(img_elem.get("height", "640"))

        gt_center, gt_tip, gt_ellipse = extract_gt_from_cvat(img_elem, w, h)
        if gt_center is None or gt_tip is None:
            continue

        # Load image
        try:
            data = z.read(f"images/{name}")
            image = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32)
        except Exception:
            continue

        # Resize to 640x640 if needed
        if image.shape[0] != 640 or image.shape[1] != 640:
            image = np.asarray(
                Image.fromarray(image.astype(np.uint8)).resize((640, 640)),
                dtype=np.float32,
            )
            # Scale GT coordinates
            gt_center = gt_center * np.array([640 / w, 640 / h])
            gt_tip = gt_tip * np.array([640 / w, 640 / h])
            if gt_ellipse is not None:
                gt_ellipse = {
                    key: value * (640.0 / (w if key in ("cx", "rx") else h))
                    for key, value in gt_ellipse.items()
                }

        # Full pipeline: ellipse → crop → center/tip
        if use_ground_truth_ellipse and gt_ellipse is not None:
            ell_center = np.array([gt_ellipse["cx"], gt_ellipse["cy"]], dtype=np.float32) / 640.0
            ell_radius = np.array([gt_ellipse["rx"], gt_ellipse["ry"]], dtype=np.float32) / 640.0
        else:
            ell_center, ell_radius = predict_ellipse(ell_interp, ell_in, ell_outs, image)
        pred_center, pred_tip = predict_keypoints(
            ct_interp, ct_in, ct_out, image, ell_center, ell_radius,
        )

        c_err = float(np.linalg.norm(pred_center - gt_center))
        t_err = float(np.linalg.norm(pred_tip - gt_tip))
        errors_c.append(c_err)
        errors_t.append(t_err)

        if gt_ellipse:
            ecx = gt_ellipse["cx"]
            ecy = gt_ellipse["cy"]
            ell_err = np.sqrt((ell_center[0] * 640 - ecx) ** 2 + (ell_center[1] * 640 - ecy) ** 2)
            ellipse_errors.append(ell_err)

        count += 1

    if count == 0:
        print(f"  {zip_name}: no valid images")
        return

    ec = np.array(errors_c)
    et = np.array(errors_t)

    print(f"\n  {zip_name} ({count} images):")
    print(f"    Center ≤8px:  {np.mean(ec <= 8):.1%} ({int(np.sum(ec <= 8))}/{count})")
    print(f"    Tip ≤8px:     {np.mean(et <= 8):.1%} ({int(np.sum(et <= 8))}/{count})")
    print(f"    Center mean:  {ec.mean():.1f}px  median: {np.median(ec):.1f}px  p90: {np.percentile(ec, 90):.1f}px")
    print(f"    Tip mean:     {et.mean():.1f}px  median: {np.median(et):.1f}px  p90: {np.percentile(et, 90):.1f}px")

    if ellipse_errors:
        ee = np.array(ellipse_errors)
        print(f"    Ellipse center err: {ee.mean():.1f}px median: {np.median(ee):.1f}px")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth-ellipse",
        action="store_true",
        help="Use annotated ellipses to isolate keypoint performance from ellipse errors.",
    )
    args = parser.parse_args()

    print("Loading models...")
    ell_interp, ell_in, ell_outs, ct_interp, ct_in, ct_out = load_models()

    print("\nEvaluating full pipeline (linear radius ellipse + center/tip):")
    print("=" * 60)

    for zip_name in ["test_1.zip", "test_2.zip", "test_3.zip"]:
        evaluate_zip(
            zip_name,
            ell_interp,
            ell_in,
            ell_outs,
            ct_interp,
            ct_in,
            ct_out,
            max_images=50,
            use_ground_truth_ellipse=args.ground_truth_ellipse,
        )


if __name__ == "__main__":
    main()
