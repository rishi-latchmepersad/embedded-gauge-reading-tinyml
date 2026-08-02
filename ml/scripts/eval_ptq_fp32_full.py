"""Full eval of the PTQ-FP32 TFLite model on val and test sets.

Run after try_alt_quantization.py to see if PTQ on the FP32 model
produces a usable TFLite model on the full 640x640 grayscale data.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import tf_keras as keras

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from embedded_gauge_reading_tinyml.qat_encoder_640g import build_qat_encoder_640g  # noqa

IMAGE_SIZE = 640
DATA_DIR = ROOT / "data" / "repvgg_ellipse"
ART = ROOT / "artifacts" / "gauge_ellipse_qat_encoder_640g_v2"
TFLITE_PATH = Path("/tmp/ptq_fp32_eval.tflite")


def _load_split(split: str):
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
            img = tf.image.resize(img[..., None], (IMAGE_SIZE, IMAGE_SIZE), method="bilinear").numpy().squeeze(-1)
        images[i, ..., 0] = img / 255.0
        targets["center_xy"][i] = [lab["cx"], lab["cy"]]
        targets["radius_xy"][i] = [lab["rx"], lab["ry"]]
    return images, targets


def main():
    print("Loading FP32 model and converting to int8 with PTQ...")
    fp32 = keras.models.load_model(ART / "model_fp32.keras", compile=False)
    train_images, _ = _load_split("train")
    print(f"  {len(train_images)} training images for representative dataset")

    def rep():
        rng = np.random.default_rng(42)
        for idx in rng.choice(len(train_images), size=min(1024, len(train_images)), replace=False):
            yield [train_images[idx:idx+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(fp32)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
    blob = converter.convert()
    TFLITE_PATH.write_bytes(blob)
    print(f"  TFLite size: {len(blob) / 1024:.1f} KB")

    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    in_scale, in_zp = in_det["quantization"]
    print(f"  Input: scale={in_scale}, zp={in_zp}")
    for d in out_dets:
        print(f"  Output idx {d['index']} shape={d['shape']} dtype={d['dtype']}")

    # The TFLite output order is [center_xy, confidence, radius_xy] based on
    # the Keras model output names (center_xy, radius_xy, confidence) but
    # the TFLite may re-order them. We must match by output shape: dim 2 = center
    # or radius, dim 1 = confidence.
    two_dim_outputs = [d for d in out_dets if int(d["shape"][-1]) == 2]
    one_dim_outputs = [d for d in out_dets if int(d["shape"][-1]) == 1]

    # Eval on val and test
    results = {}
    for split in ["val", "test"]:
        print(f"\nEvaluating {split}...")
        images, targets = _load_split(split)
        n = len(images)
        # Pre-allocate output buffers
        center_buf = np.zeros((n, 2), dtype=np.float32)
        radius_buf = np.zeros((n, 2), dtype=np.float32)
        for i, img in enumerate(images):
            xq = np.clip(np.round(img[None] / in_scale + in_zp), -128, 127).astype(np.int8)
            interp.set_tensor(in_det["index"], xq)
            interp.invoke()
            for d in two_dim_outputs:
                raw = interp.get_tensor(d["index"])
                s, z = d["quantization"]
                if d["dtype"] == np.int8:
                    dequant = (raw.astype(np.float32) - z) * s
                else:
                    dequant = raw.astype(np.float32)
                dequant = dequant[0]
                # First 2-dim output is center_xy, second is radius_xy
                if d["index"] == two_dim_outputs[0]["index"]:
                    center_buf[i] = dequant
                else:
                    radius_buf[i] = dequant

        # Compute pixel error
        center_err_px = np.linalg.norm(
            (center_buf - targets["center_xy"]) * IMAGE_SIZE, axis=1,
        )
        radius_err_px = np.linalg.norm(
            (radius_buf - targets["radius_xy"]) * IMAGE_SIZE, axis=1,
        )
        center_pred_variance_px = float(np.var(radius_buf[:, 0] * IMAGE_SIZE))
        radius_gt_variance_px = float(np.var(targets["radius_xy"][:, 0] * IMAGE_SIZE))

        results[split] = {
            "n": n,
            "model": "ptq_fp32_int8",
            "center_mae_px": float(np.mean(center_err_px)),
            "center_median_px": float(np.median(center_err_px)),
            "center_pct_le_4px": float(np.mean(center_err_px <= 4.0)),
            "center_pct_le_8px": float(np.mean(center_err_px <= 8.0)),
            "radius_mae_px": float(np.mean(radius_err_px)),
            "radius_pred_variance_px": center_pred_variance_px,
            "radius_gt_variance_px": radius_gt_variance_px,
        }
        print(f"  {split}: center MAE={results[split]['center_mae_px']:.2f}px, "
              f"% within 8px={results[split]['center_pct_le_8px']*100:.1f}%, "
              f"radius MAE={results[split]['radius_mae_px']:.2f}px, "
              f"radius pred var={center_pred_variance_px:.1f} (gt var={radius_gt_variance_px:.1f})")

    print(f"\nSummary: {json.dumps(results, indent=2)}")
    out_path = Path("/tmp/ptq_fp32_eval_report.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
