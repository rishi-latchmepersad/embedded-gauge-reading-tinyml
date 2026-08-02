"""Full eval of the 384x384 QAT encoder TFLite on val and test sets.

Note: This script loads images at the model's native input size (384x384).
The TFLite int8 outputs are dequantized using the model's per-output scale
and zero_point.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "repvgg_ellipse"
ART = ROOT / "artifacts" / f"gauge_ellipse_qat_encoder_{IMAGE_SIZE}g_cvat_v1"
TFLITE_PATH = ART / "model_int8.tflite"
FP32_PATH = ART / "model_fp32.keras"
IMAGE_SIZE = 384


def _load_split(split: str):
    """Load split at 384x384."""
    import json as _json
    from PIL import Image
    labels = _json.loads((DATA_DIR / split / "labels.json").read_text())
    images = np.zeros((len(labels), IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    targets = {
        "center_xy": np.zeros((len(labels), 2), dtype=np.float32),
        "radius_xy": np.zeros((len(labels), 2), dtype=np.float32),
    }
    img_dir = DATA_DIR / split / "images"
    for i, lab in enumerate(labels):
        img = np.asarray(Image.open(img_dir / lab["image"]).convert("L"), dtype=np.float32)
        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img = tf.image.resize(img[..., None], (IMAGE_SIZE, IMAGE_SIZE),
                                  method="bilinear").numpy().squeeze(-1)
        images[i, ..., 0] = img / 255.0
        targets["center_xy"][i] = [lab["cx"], lab["cy"]]
        targets["radius_xy"][i] = [lab["rx"], lab["ry"]]
    return images, targets


def main():
    print(f"Loading TFLite from {TFLITE_PATH}...")
    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]
    print(f"  Input: scale={in_scale}, zp={in_zp}")
    print(f"  Output: scale={out_scale}, zp={out_zp}")
    print(f"  Input shape: {in_det['shape']}")
    print(f"  Output shape: {out_det['shape']}")

    results = {}
    for split in ["val", "test"]:
        print(f"\nEvaluating {split}...")
        images, targets = _load_split(split)
        n = len(images)
        # Single 5-vector output, with cx/cy/rx/ry/conf
        preds = np.zeros((n, 5), dtype=np.float32)
        for i, img in enumerate(images):
            xq = np.clip(np.round(img[None] / in_scale + in_zp), -128, 127).astype(np.int8)
            interp.set_tensor(in_det["index"], xq)
            interp.invoke()
            raw = interp.get_tensor(out_det["index"])
            dequant = (raw.astype(np.float32) - out_zp) * out_scale
            preds[i] = dequant.flatten()

        # Center MAE in pixels (predictions are in [0, 1] of the 384x384 image,
        # but GT is in [0, 1] of the ORIGINAL 640x640 image. We need to scale
        # predictions to 640x640 since the model was trained with 640x640 GT
        # resized to 384x384).
        # The model output is normalized in [0, 1] of 384x384 space, which
        # is the same normalized [0, 1] of 640x640 space (just downsampled).
        # So no scale conversion needed for normalized coords; just multiply
        # by IMAGE_SIZE for pixel error.
        center_pred_px = preds[:, 0:2] * IMAGE_SIZE
        radius_pred_px = preds[:, 2:4] * IMAGE_SIZE
        center_gt_px = targets["center_xy"] * IMAGE_SIZE
        radius_gt_px = targets["radius_xy"] * IMAGE_SIZE

        center_err_px = np.linalg.norm(center_pred_px - center_gt_px, axis=1)
        radius_err_px = np.linalg.norm(radius_pred_px - radius_gt_px, axis=1)

        results[split] = {
            "n": n,
            "model": "qat_encoder_384g_int8",
            "center_mae_px": float(np.mean(center_err_px)),
            "center_median_px": float(np.median(center_err_px)),
            "center_pct_le_4px": float(np.mean(center_err_px <= 4.0)),
            "center_pct_le_8px": float(np.mean(center_err_px <= 8.0)),
            "center_pct_le_16px": float(np.mean(center_err_px <= 16.0)),
            "radius_mae_px": float(np.mean(radius_err_px)),
            "radius_median_px": float(np.median(radius_err_px)),
            "radius_pred_variance_px": float(np.var(radius_pred_px[:, 0])),
            "radius_gt_variance_px": float(np.var(radius_gt_px[:, 0])),
        }
        r = results[split]
        print(f"  {split}: center MAE={r['center_mae_px']:.2f}px, "
              f"%within4={r['center_pct_le_4px']*100:.1f}%, "
              f"%within8={r['center_pct_le_8px']*100:.1f}%, "
              f"%within16={r['center_pct_le_16px']*100:.1f}%")
        print(f"          radius MAE={r['radius_mae_px']:.2f}px, "
              f"radius var pred={r['radius_pred_variance_px']:.1f} (gt={r['radius_gt_variance_px']:.1f})")

    print(f"\nSummary: {json.dumps(results, indent=2)}")
    out_path = ART / "eval_report.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
