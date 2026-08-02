#!/usr/bin/env python3
"""Evaluate the full ellipse detector + needle UNet pipeline on all test sets.

Pipeline:
1. Load test images from pre-extracted data
2. Run ellipse detector to get (cx, cy, rx, ry) for each image
3. Crop gauge face from each image using predicted ellipse
4. Run needle UNet on cropped gauge face to get center/tip heatmaps
5. Decode heatmaps via softargmax to get pixel coordinates
6. Report per-split center/tip MAE in original image pixel space

This evaluates the END-TO-END pipeline, not individual models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR = ROOT / "data" / "needle_pipeline"
ELLIPSE_MODEL = ROOT / "artifacts" / "ellipse_detector_224_v1" / "model_int8.tflite"
NEEDLE_MODEL = ROOT / "artifacts" / "needle_unet_224_v1" / "model_int8.tflite"
DEFAULT_OUTPUT = ROOT / "artifacts" / "pipeline_eval_224_v1"
CROP_SCALE = 1.35  # Matches training crop scale
SEED = 42


def _decode_heatmap_peak(heatmap: np.ndarray) -> tuple[float, float]:
    """Sub-pixel keypoint from heatmap using weighted softargmax."""
    h, w = heatmap.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weights = np.maximum(heatmap - 0.03, 0.0) ** 2
    total = weights.sum()
    if total < 1e-6:
        return 0.5, 0.5
    cx = (weights * xx).sum() / total / (w - 1)
    cy = (weights * yy).sum() / total / (h - 1)
    return float(cx), float(cy)


def _crop_gauge_face(
    img: np.ndarray, face: dict, padding: float = CROP_SCALE
) -> tuple[np.ndarray, dict]:
    """Crop and resize the gauge face region from the image.

    Returns the crop and the transform info needed to map coordinates back.
    """
    h, w = img.shape[:2]
    cx = face["cx"] * w
    cy = face["cy"] * h
    rx = face["rx"] * w * padding
    ry = face["ry"] * h * padding

    x1 = max(0, int(cx - rx))
    y1 = max(0, int(cy - ry))
    x2 = min(w, int(cx + rx))
    y2 = min(h, int(cy + ry))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((224, 224), dtype=np.float32), {"x1": 0, "y1": 0, "cw": 1, "ch": 1}

    pil_crop = Image.fromarray(crop.astype(np.uint8), mode="L")
    pil_crop = pil_crop.resize((224, 224), Image.BILINEAR)

    transform = {"x1": x1, "y1": y1, "cw": x2 - x1, "ch": y2 - y1}
    return np.array(pil_crop, dtype=np.float32), transform


def _crop_to_original(
    crop_x: float, crop_y: float, transform: dict
) -> tuple[float, float]:
    """Map normalized crop coordinates back to original image space."""
    orig_x = transform["x1"] + crop_x * transform["cw"]
    orig_y = transform["y1"] + crop_y * transform["ch"]
    return orig_x, orig_y


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ellipse-model", type=Path, default=ELLIPSE_MODEL)
    parser.add_argument("--needle-model", type=Path, default=NEEDLE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crop-scale", type=float, default=CROP_SCALE)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Load TFLite models ─────────────────────────────────────────────
    print("Loading models...")
    ell_interp = tf.lite.Interpreter(model_path=str(args.ellipse_model))
    ell_interp.allocate_tensors()
    ell_in = ell_interp.get_input_details()[0]
    ell_out = ell_interp.get_output_details()[0]
    ell_scale, ell_zp = ell_in["quantization"]

    needle_interp = tf.lite.Interpreter(model_path=str(args.needle_model))
    needle_interp.allocate_tensors()
    needle_in = needle_interp.get_input_details()[0]
    needle_out = needle_interp.get_output_details()[0]
    needle_scale, needle_zp = needle_in["quantization"]

    print(f"  Ellipse: {args.ellipse_model}")
    print(f"  Needle:  {args.needle_model}")

    # ── Load test splits ───────────────────────────────────────────────
    results = {}
    for split in ["test"]:
        images = np.load(DATA_DIR / split / "images.npy")
        center_hm = np.load(DATA_DIR / split / "center_heatmaps.npy")
        tip_hm = np.load(DATA_DIR / split / "tip_heatmaps.npy")
        has_needle = np.load(DATA_DIR / split / "has_needle.npy")
        ellipse_labels = np.load(DATA_DIR / split / "ellipse_labels.npy")

        n = len(images)
        print(f"\n{'='*60}")
        print(f"Evaluating {split}: {n} images, {int(has_needle.sum())} with needle labels")
        print(f"{'='*60}")

        center_errs = []
        tip_errs = []
        ellipse_center_errs = []

        for i in range(n):
            img = images[i, ..., 0]  # (224, 224)

            # ── Stage 1: Ellipse detection ─────────────────────────────
            xq = np.clip(np.round(img[None, ..., None] / ell_scale + ell_zp), -128, 127).astype(np.int8)
            ell_interp.set_tensor(ell_in["index"], xq)
            ell_interp.invoke()
            ell_raw = ell_interp.get_tensor(ell_out["index"])
            if ell_out["dtype"] == np.int8:
                s, z = ell_out["quantization"]
                pred_ellipse = ((ell_raw.astype(np.float32) - z) * s)[0]
            else:
                pred_ellipse = ell_raw.astype(np.float32)[0]

            pcx, pcy, prx, pry = pred_ellipse
            gcx, gcy, grx, gry = ellipse_labels[i]

            # Ellipse center error (in 224px space)
            ell_err = np.sqrt(((pcx - gcx) * 224) ** 2 + ((pcy - gcy) * 224) ** 2)
            ellipse_center_errs.append(ell_err)

            if not has_needle[i]:
                continue

            # ── Stage 2: Crop using predicted ellipse ──────────────────
            face = {"cx": pcx, "cy": pcy, "rx": prx, "ry": pry}
            crop, transform = _crop_gauge_face(img, face, padding=args.crop_scale)
            crop_norm = crop / 255.0

            # ── Stage 3: Needle heatmap prediction ─────────────────────
            xq = np.clip(np.round(crop_norm[None, ..., None] / needle_scale + needle_zp), -128, 127).astype(np.int8)
            needle_interp.set_tensor(needle_in["index"], xq)
            needle_interp.invoke()
            needle_raw = needle_interp.get_tensor(needle_out["index"])
            if needle_out["dtype"] == np.int8:
                s, z = needle_out["quantization"]
                hm = ((needle_raw.astype(np.float32) - z) * s)[0]
            else:
                hm = needle_raw.astype(np.float32)[0]

            # Decode center and tip from predicted heatmaps
            pcx_crop, pcy_crop = _decode_heatmap_peak(hm[..., 0])
            ptx_crop, pty_crop = _decode_heatmap_peak(hm[..., 1])

            # Map back to original image space
            pcx_orig, pcy_orig = _crop_to_original(pcx_crop, pcy_crop, transform)
            ptx_orig, pty_orig = _crop_to_original(ptx_crop, pty_crop, transform)

            # Ground truth in original image space
            gcx_orig = center_hm[i]  # Need to decode from heatmap
            # Actually, we stored heatmaps in crop space. Let me decode GT heatmaps too.
            gcx_c, gcy_c = _decode_heatmap_peak(center_hm[i])
            gtx_c, gty_c = _decode_heatmap_peak(tip_hm[i])

            # Map GT back to original image space
            gcx_orig, gcy_orig = _crop_to_original(gcx_c, gcy_c, transform)
            gtx_orig, gty_orig = _crop_to_original(gtx_c, gty_c, transform)

            # Euclidean error in original pixel space
            c_err = np.sqrt((pcx_orig - gcx_orig) ** 2 + (pcy_orig - gcy_orig) ** 2)
            t_err = np.sqrt((ptx_orig - gtx_orig) ** 2 + (pty_orig - gty_orig) ** 2)
            center_errs.append(c_err)
            tip_errs.append(t_err)

        # ── Report ─────────────────────────────────────────────────────
        ell_c = np.array(ellipse_center_errs)
        print(f"\nEllipse detection ({n}):")
        print(f"  Center MAE: {ell_c.mean():.2f}px")
        print(f"  Center <=8px: {(ell_c <= 8).mean() * 100:.1f}%")
        print(f"  Center <=4px: {(ell_c <= 4).mean() * 100:.1f}%")

        if center_errs:
            c = np.array(center_errs)
            t = np.array(tip_errs)
            print(f"\nFull pipeline needle ({len(c)} with needle labels):")
            print(f"  Center MAE: {c.mean():.2f}px, median: {np.median(c):.2f}px")
            print(f"  Center <=8px: {(c <= 8).mean() * 100:.1f}%, <=4px: {(c <= 4).mean() * 100:.1f}%")
            print(f"  Tip MAE:     {t.mean():.2f}px, median: {np.median(t):.2f}px")
            print(f"  Tip <=8px:    {(t <= 8).mean() * 100:.1f}%, <=4px: {(t <= 4).mean() * 100:.1f}%")

            results[split] = {
                "n": n,
                "n_needle": len(c),
                "ellipse_center_mae": float(ell_c.mean()),
                "ellipse_center_le8": float((ell_c <= 8).mean() * 100),
                "center_mae": float(c.mean()),
                "center_median": float(np.median(c)),
                "center_le8": float((c <= 8).mean() * 100),
                "center_le4": float((c <= 4).mean() * 100),
                "tip_mae": float(t.mean()),
                "tip_median": float(np.median(t)),
                "tip_le8": float((t <= 8).mean() * 100),
                "tip_le4": float((t <= 4).mean() * 100),
            }
        else:
            results[split] = {
                "n": n,
                "n_needle": 0,
                "ellipse_center_mae": float(ell_c.mean()),
                "ellipse_center_le8": float((ell_c <= 8).mean() * 100),
                "note": "No needle labels available for evaluation",
            }

    # ── Save report ────────────────────────────────────────────────────
    report = {
        "pipeline": "ellipse_detector_224 + needle_unet_224",
        "crop_scale": args.crop_scale,
        "ellipse_model": str(args.ellipse_model),
        "needle_model": str(args.needle_model),
        "results": results,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {args.output / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
