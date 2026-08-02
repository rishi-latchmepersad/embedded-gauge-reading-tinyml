#!/usr/bin/env python3
"""Visualize worst and best tip predictions from the v6 int8 keypoint UNet.

For each of test_1/test_2/test_3:
  - Pick the 5 images with WORST tip error
  - Draw GT center (green), GT tip (red), predicted center (blue),
    predicted tip (yellow), and error in pixels
  - Save annotated images to /tmp/v6_failures/

Also pick 5 BEST tip-error images from test_1 for comparison.

Uses CROP_SCALE=1.35 for ellipse crop and local softargmax decode
(weights = max(hm-0.03, 0)^2, full-heatmap window).

Usage:
    poetry run python scripts/analyze_v6_failures.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "gauge_keypoint_unet_224g_v6" / "model_int8.tflite"
LABELLED = ROOT / "data" / "labelled"
OUTPUT_DIR = Path("/tmp/v6_failures")
CROP_SCALE = 1.35
HEATMAP_SIZE = 56
N_WORST = 5
N_BEST = 5

# Label name variants
_ELLIPSE_LABELS = {"GaugeFace", "temp_dial"}
_CENTER_LABELS = {"Center", "temp_center"}
_TIP_LABELS = {"Tip", "temp_tip"}

SPLITS = [
    ("test_1", "test_1.zip"),
    ("test_2", "test_2.zip"),
    ("test_3", "test_3.zip"),
]


def _load_model() -> tuple[tf.lite.Interpreter, dict, dict]:
    """Load the int8 TFLite model and return interpreter + I/O metadata."""
    interp = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    return interp, in_det, out_det


def _extract_records(zip_path: Path) -> list[dict]:
    """Extract annotation records from a CVAT zip, handling all 3 formats."""
    records: list[dict] = []
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


def _decode_heatmap_peak(heatmap: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel keypoint from heatmap using local softargmax.

    Weights = max(hm - 0.03, 0)^2, evaluated over the full heatmap.
    Returns (x_norm, y_norm, peak_value) in [0,1] normalized space.
    """
    h, w = heatmap.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weights = np.maximum(heatmap - 0.03, 0.0) ** 2
    total = float(weights.sum())
    if total < 1e-6:
        return 0.5, 0.5, 0.0
    cx = float((weights * xx).sum() / total / (w - 1))
    cy = float((weights * yy).sum() / total / (h - 1))
    return cx, cy, float(heatmap.max())


def _run_inference(
    interp: tf.lite.Interpreter,
    in_det: dict,
    out_det: dict,
    crop_image: Image.Image,
) -> np.ndarray:
    """Run TFLite inference on a cropped grayscale image. Returns 56x56x2 heatmap."""
    resized = crop_image.resize((224, 224), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    # Quantize to int8
    in_scale, in_zp = in_det["quantization"]
    xq = np.clip(np.round(x[None, ..., None] / float(in_scale) + float(in_zp)),
                 -128, 127).astype(np.int8)
    interp.set_tensor(int(in_det["index"]), xq)
    interp.invoke()
    raw_out = interp.get_tensor(int(out_det["index"]))
    # Dequantize output
    out_scale, out_zp = out_det["quantization"]
    hm = ((raw_out.astype(np.float32) - float(out_zp)) * float(out_scale))[0]  # 56x56x2
    return hm


def _compute_errors(records: list[dict], interp, in_det, out_det) -> list[dict]:
    """Run inference on every record, compute tip error. Returns enriched records."""
    results: list[dict] = []
    zip_path = Path(records[0]["zip"])
    with zipfile.ZipFile(zip_path) as z:
        for rec in records:
            basename = Path(rec["name"]).name
            matches = [n for n in z.namelist() if Path(n).name == basename]
            if not matches:
                continue
            raw = z.read(matches[0])
            img_pil = Image.open(io.BytesIO(raw)).convert("L")
            w_img, h_img = img_pil.size

            # Ellipse crop: square side = 2 * max(rx, ry) * CROP_SCALE
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

            cropped = img_pil.crop((crop_left, crop_top, crop_right, crop_bottom))

            # Run model
            hm = _run_inference(interp, in_det, out_det, cropped)

            # Decode predicted keypoints in crop-normalized [0,1] space
            pcx, pcy, c_peak = _decode_heatmap_peak(hm[..., 0])
            ptx, pty, t_peak = _decode_heatmap_peak(hm[..., 1])

            # Map predicted coords back to source image pixels
            pred_center = np.array([crop_left + pcx * actual_side,
                                    crop_top + pcy * actual_side])
            pred_tip = np.array([crop_left + ptx * actual_side,
                                 crop_top + pty * actual_side])

            # Ground truth in source image pixels
            gt_center = np.array([rec["center_x"], rec["center_y"]])
            gt_tip = np.array([rec["tip_x"], rec["tip_y"]])

            center_err = float(np.linalg.norm(pred_center - gt_center))
            tip_err = float(np.linalg.norm(pred_tip - gt_tip))

            # Also compute the raw RGB image for annotation
            img_rgb = Image.open(io.BytesIO(raw)).convert("RGB")

            results.append({
                **rec,
                "pred_center": pred_center,
                "pred_tip": pred_tip,
                "gt_center": gt_center,
                "gt_tip": gt_tip,
                "center_err": center_err,
                "tip_err": tip_err,
                "c_peak": c_peak,
                "t_peak": t_peak,
                "crop_left": crop_left,
                "crop_top": crop_top,
                "actual_side": actual_side,
                "img_rgb": img_rgb,
                "img_width": w_img,
                "img_height": h_img,
            })
    return results


def _annotate_image(result: dict, title_suffix: str) -> np.ndarray:
    """Draw GT and predicted keypoints on the original image. Returns RGB array."""
    img = np.array(result["img_rgb"].copy())
    h_img, w_img = img.shape[:2]

    gt_c = result["gt_center"]
    gt_t = result["gt_tip"]
    pred_c = result["pred_center"]
    pred_t = result["pred_tip"]
    c_err = result["center_err"]
    t_err = result["tip_err"]

    # Draw on a figure with the image
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
    ax.imshow(img)

    # Draw crop box (dashed white)
    cl, ct = result["crop_left"], result["crop_top"]
    side = result["actual_side"]
    from matplotlib.patches import Rectangle
    rect = Rectangle((cl, ct), side, side, linewidth=1.5, edgecolor="white",
                      facecolor="none", linestyle="--", alpha=0.7)
    ax.add_patch(rect)

    # GT needle line (white, thin)
    ax.plot([gt_c[0], gt_t[0]], [gt_c[1], gt_t[1]],
            color="white", linewidth=1.5, alpha=0.8, zorder=2)

    # GT center (green circle)
    ax.scatter([gt_c[0]], [gt_c[1]], c="lime", s=80, marker="o",
               edgecolors="black", linewidths=1.0, zorder=3, label="GT center")
    # GT tip (red X)
    ax.scatter([gt_t[0]], [gt_t[1]], c="red", s=80, marker="x",
               linewidths=2.0, zorder=3, label="GT tip")

    # Predicted needle line (cyan)
    ax.plot([pred_c[0], pred_t[0]], [pred_c[1], pred_t[1]],
            color="deepskyblue", linewidth=1.5, alpha=0.8, zorder=2)

    # Predicted center (blue circle)
    ax.scatter([pred_c[0]], [pred_c[1]], c="blue", s=80, marker="o",
               edgecolors="white", linewidths=1.0, zorder=3, label="Pred center")
    # Predicted tip (yellow X)
    ax.scatter([pred_t[0]], [pred_t[1]], c="yellow", s=80, marker="x",
               linewidths=2.0, edgecolors="black",
               zorder=3, label="Pred tip")

    # Error text
    ax.set_title(
        f"{result['name']}{title_suffix}\n"
        f"center err: {c_err:.1f}px | tip err: {t_err:.1f}px | "
        f"center peak: {result['c_peak']:.3f} | tip peak: {result['t_peak']:.3f}",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_axis_off()

    fig.tight_layout()

    # Render to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return arr


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {MODEL_PATH}")
    interp, in_det, out_det = _load_model()
    print(f"  Input:  {in_det['shape']}  dtype={in_det['dtype']}  "
          f"scale={in_det['quantization'][0]:.6f}")
    print(f"  Output: {out_det['shape']}  scale={out_det['quantization'][0]:.6f}")

    all_results: dict[str, list[dict]] = {}

    for split_name, zip_name in SPLITS:
        zip_path = LABELLED / zip_name
        if not zip_path.exists():
            print(f"\n  {zip_name}: not found, skipping")
            continue

        records = _extract_records(zip_path)
        if not records:
            print(f"\n  {zip_name}: no valid records")
            continue

        print(f"\n  {zip_name}: {len(records)} images, running inference...")
        results = _compute_errors(records, interp, in_det, out_det)
        print(f"    Evaluated {len(results)} images")

        # Print stats
        tip_errs = np.array([r["tip_err"] for r in results])
        c_errs = np.array([r["center_err"] for r in results])
        print(f"    Tip    MAE: {tip_errs.mean():.2f}px  "
              f"median: {np.median(tip_errs):.2f}px  "
              f"p90: {np.percentile(tip_errs, 90):.2f}px  "
              f"max: {tip_errs.max():.2f}px")
        print(f"    Center MAE: {c_errs.mean():.2f}px  "
              f"median: {np.median(c_errs):.2f}px  "
              f"p90: {np.percentile(c_errs, 90):.2f}px  "
              f"max: {c_errs.max():.2f}px")

        all_results[split_name] = results

        # --- Worst 5 by tip error ---
        sorted_by_tip = sorted(results, key=lambda r: r["tip_err"], reverse=True)
        worst = sorted_by_tip[:N_WORST]
        out_dir = OUTPUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, res in enumerate(worst):
            ann = _annotate_image(res, f"  [WORST #{i+1}]")
            out_path = out_dir / f"worst_{i+1}_tip{res['tip_err']:.1f}px_{Path(res['name']).stem}.png"
            Image.fromarray(ann).save(out_path)
            print(f"    Saved worst #{i+1}: {out_path.name}")

        # --- Best 5 from test_1 only ---
        if split_name == "test_1":
            sorted_by_tip_asc = sorted(results, key=lambda r: r["tip_err"])
            best = sorted_by_tip_asc[:N_BEST]
            for i, res in enumerate(best):
                ann = _annotate_image(res, f"  [BEST #{i+1}]")
                out_path = out_dir / f"best_{i+1}_tip{res['tip_err']:.1f}px_{Path(res['name']).stem}.png"
                Image.fromarray(ann).save(out_path)
                print(f"    Saved best  #{i+1}: {out_path.name}")

    # --- Print summary of failure patterns ---
    print("\n" + "=" * 70)
    print("FAILURE PATTERN ANALYSIS")
    print("=" * 70)
    for split_name, results in all_results.items():
        tip_errs = np.array([r["tip_err"] for r in results])
        c_errs = np.array([r["center_err"] for r in results])
        worst = sorted(results, key=lambda r: r["tip_err"], reverse=True)[:5]
        best = sorted(results, key=lambda r: r["tip_err"])[:5]

        print(f"\n--- {split_name} ---")
        print(f"  Total images: {len(results)}")
        print(f"  Tip MAE: {tip_errs.mean():.2f}px, Center MAE: {c_errs.mean():.2f}px")
        print(f"\n  WORST 5 tip errors:")
        for i, r in enumerate(worst):
            side = r["actual_side"]
            rx, ry = r["rx"], r["ry"]
            # Check if tip is near edge of crop
            gt_t = r["gt_tip"]
            cl, ct = r["crop_left"], r["crop_top"]
            rel_x = (gt_t[0] - cl) / side if side > 0 else 0.5
            rel_y = (gt_t[1] - ct) / side if side > 0 else 0.5
            edge_dist = min(rel_x, rel_y, 1 - rel_x, 1 - rel_y)
            print(f"    #{i+1} tip_err={r['tip_err']:.1f}px  "
                  f"center_err={r['center_err']:.1f}px  "
                  f"tip_peak={r['t_peak']:.3f}  "
                  f"gauge={rx:.0f}x{ry:.0f}px  "
                  f"crop_side={side:.0f}px  "
                  f"tip_in_crop=({rel_x:.2f},{rel_y:.2f})  "
                  f"edge_dist={edge_dist:.2f}  "
                  f"file={r['name']}")
        print(f"\n  BEST 5 tip errors:")
        for i, r in enumerate(best):
            print(f"    #{i+1} tip_err={r['tip_err']:.1f}px  "
                  f"center_err={r['center_err']:.1f}px  "
                  f"tip_peak={r['t_peak']:.3f}  "
                  f"gauge={r['rx']:.0f}x{r['ry']:.0f}px  "
                  f"crop_side={r['actual_side']:.0f}px  "
                  f"file={r['name']}")

    # Save a JSON report
    report = {}
    for split_name, results in all_results.items():
        report[split_name] = {
            "n_images": len(results),
            "worst_5": [
                {"name": r["name"], "tip_err": r["tip_err"], "center_err": r["center_err"],
                 "t_peak": r["t_peak"], "c_peak": r["c_peak"],
                 "rx": r["rx"], "ry": r["ry"], "crop_side": r["actual_side"]}
                for r in sorted(results, key=lambda r: r["tip_err"], reverse=True)[:5]
            ],
            "best_5": [
                {"name": r["name"], "tip_err": r["tip_err"], "center_err": r["center_err"],
                 "t_peak": r["t_peak"], "c_peak": r["c_peak"],
                 "rx": r["rx"], "ry": r["ry"], "crop_side": r["actual_side"]}
                for r in sorted(results, key=lambda r: r["tip_err"])[:5]
            ],
            "tip_err_stats": {
                "mean": float(tip_errs.mean()),
                "median": float(np.median(tip_errs)),
                "p90": float(np.percentile(tip_errs, 90)),
                "max": float(tip_errs.max()),
            },
        }
    report_path = OUTPUT_DIR / "failure_analysis.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nJSON report saved to {report_path}")
    print(f"Annotated images saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
