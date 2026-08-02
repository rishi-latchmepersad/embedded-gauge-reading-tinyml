#!/usr/bin/env python3
"""Evaluate keypoint model per test split (test_1, test_2, test_3).

Loads the int8 TFLite model and evaluates separately on each test zip
to identify which gauge type or data domain is the weak point.

Usage:
    python scripts/eval_keypoint_per_split.py \
        --model artifacts/gauge_keypoint_unet_224g_v5/model_int8.tflite
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data" / "labelled"
HEATMAP_SIZE = 56
CROP_SCALE = 1.35  # must match the data prep used for training

# Label name variants (same as prepare_gauge_keypoint_224_data.py)
_ELLIPSE_LABELS = {"GaugeFace", "temp_dial"}
_CENTER_LABELS = {"Center", "temp_center"}
_TIP_LABELS = {"Tip", "temp_tip"}

SPLITS = {
    "test_1": "test_1.zip",
    "test_2": "test_2.zip",
    "test_3": "test_3.zip",
}


def _decode_heatmap_peak(heatmap: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel keypoint from heatmap using local softargmax."""
    h, w = heatmap.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weights = np.maximum(heatmap - 0.03, 0.0) ** 2
    total = weights.sum()
    if total < 1e-6:
        return 0.5, 0.5, 0.0
    cx = (weights * xx).sum() / total / (w - 1)
    cy = (weights * yy).sum() / total / (h - 1)
    return float(cx), float(cy), float(heatmap.max())


def _extract_records(zip_path: Path) -> list[dict]:
    """Extract annotation records from a CVAT zip, handling all formats."""
    records = []
    with zipfile.ZipFile(zip_path) as z:
        try:
            xml_bytes = z.read("annotations.xml")
        except KeyError:
            return records
        root = ET.fromstring(xml_bytes)
        for img_node in root.findall("image"):
            width = int(img_node.get("width", 640))
            height = int(img_node.get("height", 640))
            name = img_node.get("name")
            ellipse = None
            center_pt = None
            tip_pt = None
            for el in img_node:
                label = el.get("label", "")
                if label in _ELLIPSE_LABELS and el.tag == "ellipse":
                    ellipse = (
                        float(el.get("cx")), float(el.get("cy")),
                        float(el.get("rx")), float(el.get("ry")),
                    )
                elif label in _CENTER_LABELS:
                    if el.tag == "box":
                        xtl, ytl = float(el.get("xtl")), float(el.get("ytl"))
                        xbr, ybr = float(el.get("xbr")), float(el.get("ybr"))
                        center_pt = ((xtl + xbr) / 2, (ytl + ybr) / 2)
                    elif el.tag == "ellipse":
                        center_pt = (float(el.get("cx")), float(el.get("cy")))
                    elif el.tag == "points":
                        pts = el.get("points", "").split(",")
                        center_pt = (float(pts[0]), float(pts[1]))
                elif label in _TIP_LABELS:
                    if el.tag == "box":
                        xtl, ytl = float(el.get("xtl")), float(el.get("ytl"))
                        xbr, ybr = float(el.get("xbr")), float(el.get("ybr"))
                        tip_pt = ((xtl + xbr) / 2, (ytl + ybr) / 2)
                    elif el.tag == "ellipse":
                        tip_pt = (float(el.get("cx")), float(el.get("cy")))
                    elif el.tag == "points":
                        pts = el.get("points", "").split(",")
                        tip_pt = (float(pts[0]), float(pts[1]))
            if ellipse is not None and center_pt is not None and tip_pt is not None:
                records.append({
                    "zip": str(zip_path), "name": name,
                    "width": width, "height": height,
                    "cx": ellipse[0], "cy": ellipse[1],
                    "rx": ellipse[2], "ry": ellipse[3],
                    "center_x": center_pt[0], "center_y": center_pt[1],
                    "tip_x": tip_pt[0], "tip_y": tip_pt[1],
                })
    return records


def evaluate_split(
    interp: tf.lite.Interpreter,
    in_det: dict, out_det: dict,
    zip_name: str,
) -> dict:
    """Evaluate the model on one test split."""
    zip_path = LABELLED / zip_name
    if not zip_path.exists():
        print(f"  {zip_name}: not found")
        return {}

    records = _extract_records(zip_path)
    if not records:
        print(f"  {zip_name}: no valid records")
        return {}

    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    center_errs, tip_errs = [], []
    for rec in records:
        # Load image from zip
        with zipfile.ZipFile(zip_path) as z:
            basename = Path(rec["name"]).name
            matches = [n for n in z.namelist() if Path(n).name == basename]
            if not matches:
                continue
            raw = z.read(matches[0])
        img = Image.open(io.BytesIO(raw)).convert("L")
        w_img, h_img = img.size

        # Crop around ellipse
        side = max(2.0 * rec["rx"], 2.0 * rec["ry"]) * CROP_SCALE
        left = rec["cx"] - side / 2.0
        top = rec["cy"] - side / 2.0
        crop_left = max(0, int(left))
        crop_top = max(0, int(top))
        crop_right = min(w_img, int(left + side))
        crop_bottom = min(h_img, int(top + side))
        actual_side = max(crop_right - crop_left, crop_bottom - crop_top)
        if actual_side < 1:
            continue

        cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        resized = cropped.resize((224, 224), Image.Resampling.BILINEAR)
        x = np.asarray(resized, dtype=np.float32) / 255.0

        # Quantize and run inference
        xq = np.clip(np.round(x[None, ..., None] / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        raw_out = interp.get_tensor(out_det["index"])
        hm = ((raw_out.astype(np.float32) - out_zp) * out_scale)[0]

        # Decode predicted keypoints (in crop-normalized [0,1] space)
        pcx, pcy, _ = _decode_heatmap_peak(hm[..., 0])
        ptx, pty, _ = _decode_heatmap_peak(hm[..., 1])

        # Map back to source image coordinates
        pred_center = np.array([crop_left + pcx * actual_side, crop_top + pcy * actual_side])
        pred_tip = np.array([crop_left + ptx * actual_side, crop_top + pty * actual_side])

        # Ground truth in source image coordinates
        gt_center = np.array([rec["center_x"], rec["center_y"]])
        gt_tip = np.array([rec["tip_x"], rec["tip_y"]])

        center_errs.append(float(np.linalg.norm(pred_center - gt_center)))
        tip_errs.append(float(np.linalg.norm(pred_tip - gt_tip)))

    c = np.array(center_errs)
    t = np.array(tip_errs)
    metrics = {
        "n": len(c),
        "center_mae": float(c.mean()),
        "center_median": float(np.median(c)),
        "center_p90": float(np.percentile(c, 90)),
        "center_le4": float((c <= 4).mean() * 100),
        "center_le8": float((c <= 8).mean() * 100),
        "tip_mae": float(t.mean()),
        "tip_median": float(np.median(t)),
        "tip_p90": float(np.percentile(t, 90)),
        "tip_le4": float((t <= 4).mean() * 100),
        "tip_le8": float((t <= 8).mean() * 100),
    }
    print(f"\n  {zip_name} ({metrics['n']} images):")
    print(f"    Center MAE: {metrics['center_mae']:.2f}px  "
          f"≤4px: {metrics['center_le4']:.1f}%  ≤8px: {metrics['center_le8']:.1f}%")
    print(f"    Tip    MAE: {metrics['tip_mae']:.2f}px  "
          f"≤4px: {metrics['tip_le4']:.1f}%  ≤8px: {metrics['tip_le8']:.1f}%")
    print(f"    Center p90: {metrics['center_p90']:.1f}px  Tip p90: {metrics['tip_p90']:.1f}px")
    return metrics


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to int8 TFLite model")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    interp = tf.lite.Interpreter(model_path=str(args.model))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f"  Input:  {in_det['shape']}  scale={in_det['quantization'][0]:.6f}")
    print(f"  Output: {out_det['shape']}  scale={out_det['quantization'][0]:.6f}")

    all_metrics = {}
    for split_name, zip_name in SPLITS.items():
        metrics = evaluate_split(interp, in_det, out_det, zip_name)
        if metrics:
            all_metrics[split_name] = metrics

    # Summary
    print("\n" + "=" * 60)
    print("PER-SPLIT SUMMARY")
    print("=" * 60)
    for split, m in all_metrics.items():
        print(f"  {split}: center={m['center_mae']:.2f}px tip={m['tip_mae']:.2f}px "
              f"(tip ≤8px: {m['tip_le8']:.1f}%)")
    print()

    # Save report
    report_path = args.model.parent / "per_split_report.json"
    report_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    import json
    sys.exit(main())
