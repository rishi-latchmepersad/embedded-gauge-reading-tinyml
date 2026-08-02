"""Try alternative TFLite int8 quantization approaches on the FP32 model.

Why this script exists:
- The default TFLiteConverter PTQ path produces a TFLite int8 model
  where the radius output collapses to a constant.
- The default QAT path also produces a collapsed radius output.
- We are NOT doing conversion-rescue experiments blindly; we are
  testing three specific options the AI memory does not forbid:
  1. Larger representative dataset (more calibration samples)
  2. PTQ on the QAT model (use QAT-calibrated weights as the seed)
  3. Per-channel quantization on weights
- Each option is a single conversion pass with a different setting.
  We compare int8 vs fp32 predictions on a small held-out set.

If any approach produces a TFLite model where the radius output
varies across inputs, we have a path forward. Otherwise, the
"Family has shown TFLite mismatch" rule from the AI memory still
applies and we should fall back to the 224x224 + resize path.
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

from embedded_gauge_reading_tinyml.qat_encoder_640g import build_qat_encoder_640g  # noqa: E402

DATA_DIR = ROOT / "data" / "repvgg_ellipse"
ART = ROOT / "artifacts" / "gauge_ellipse_qat_encoder_640g_v2"


def _load_n(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Load n val images and their labels."""
    import json as _json
    from PIL import Image
    labels = _json.loads((DATA_DIR / "val" / "labels.json").read_text())[:n]
    images = np.zeros((n, 640, 640, 1), dtype=np.float32)
    targets = np.zeros((n, 5), dtype=np.float32)
    img_dir = DATA_DIR / "val" / "images"
    for i, lab in enumerate(labels):
        img = np.asarray(Image.open(img_dir / lab["image"]).convert("L"), dtype=np.float32)
        if img.shape != (640, 640):
            img = tf.image.resize(img[..., None], (640, 640), method="bilinear").numpy().squeeze(-1)
        images[i, ..., 0] = img / 255.0
        targets[i] = [lab["cx"], lab["cy"], lab["rx"], lab["ry"], 1.0]
    return images, targets


def _convert(model, rep_images, **opts):
    """Convert a Keras model to int8 TFLite with the given converter options."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if "rep_size" in opts:
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(rep_images), size=min(opts["rep_size"], len(rep_images)), replace=False)
        rep_imgs = rep_images[idxs]
    else:
        rep_imgs = rep_images
    converter.representative_dataset = lambda: (
        [img[None].astype(np.float32)] for img in rep_imgs
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if opts.get("per_channel", True):
        # Per-channel on weights is the default; explicitly enable.
        converter._experimental_disable_per_channel = False
    else:
        converter._experimental_disable_per_channel = True
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32  # keep output float for easy read
    return converter.convert()


def _eval_tflite(tflite_bytes, sample):
    """Run the TFLite on a sample of images and return per-output variance."""
    path = Path("/tmp/_probe.tflite")
    path.write_bytes(tflite_bytes)
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    in_scale, in_zp = in_det["quantization"]

    all_out = {d["index"]: [] for d in out_dets}
    for img in sample:
        xq = np.clip(np.round(img[None] / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        for d in out_dets:
            raw = interp.get_tensor(d["index"])
            s, z = d["quantization"]
            if d["dtype"] == np.int8:
                dequant = (raw.astype(np.float32) - z) * s
            else:
                dequant = raw.astype(np.float32)
            all_out[d["index"]].append(dequant.flatten())
    return {idx: np.array(outs) for idx, outs in all_out.items()}


def _report(name, tflite_bytes, tflite_outputs, fp32_preds):
    """Print a per-output summary and return a dict."""
    fp32_by_dim = {p.shape[1]: p for p in fp32_preds}
    print(f"\n=== {name} ===")
    print(f"  TFLite size: {len(tflite_bytes) / 1024:.1f} KB")
    print(f"  Outputs: {len(tflite_outputs)} tensors")
    rows = []
    for idx, arr in tflite_outputs.items():
        # Find the matching fp32 output by last dim.
        dim = arr.shape[1]
        fp32 = fp32_by_dim.get(dim)
        if fp32 is None:
            continue
        diff = np.max(np.abs(arr - fp32[:len(arr)]))
        var = float(arr.std())
        max_min = float(arr.max() - arr.min())
        rows.append({
            "name": f"output_dim_{dim}",
            "tflite_std": var,
            "tflite_max_min": max_min,
            "max_diff_vs_fp32": float(diff),
        })
        print(f"  dim={dim}: tflite std={var:.5f}, max-min={max_min:.5f}, max diff vs fp32={diff:.5f}")
    return {"name": name, "size_kb": round(len(tflite_bytes) / 1024, 1), "rows": rows}


def main():
    print("Loading FP32 Keras model...")
    fp32 = keras.models.load_model(ART / "model_fp32.keras", compile=False)
    rep_all_images, _ = _load_n(n=512)
    sample_images = rep_all_images[:50]
    sample_targets = None  # We don't need targets for the variance check

    # FP32 baseline on sample
    print("\nFP32 baseline on 50 sample images:")
    fp32_preds = fp32.predict(sample_images, batch_size=8, verbose=0)
    for name, p in zip(["center_xy", "radius_xy", "confidence"], fp32_preds):
        print(f"  {name}: std={p.std():.5f}, max-min={p.max() - p.min():.5f}")

    results = []

    # Approach 1: PTQ with 256 rep samples (baseline)
    print("\n--- Approach 1: PTQ, 256 rep, per-channel on ---")
    blob = _convert(fp32, rep_all_images, rep_size=256, per_channel=True)
    out = _eval_tflite(blob, sample_images)
    r = _report("PTQ-256-perchan-ON", blob, out, fp32_preds)
    results.append(r)

    # Approach 2: PTQ with 1024 rep samples
    print("\n--- Approach 2: PTQ, 1024 rep ---")
    # Need 1024 reps. Use full rep_all if we don't have 1024.
    blob = _convert(fp32, rep_all_images, rep_size=min(1024, len(rep_all_images)), per_channel=True)
    out = _eval_tflite(blob, sample_images)
    r = _report("PTQ-1024-perchan-ON", blob, out, fp32_preds)
    results.append(r)

    # Approach 3: PTQ with per-channel OFF (force per-tensor for weights)
    print("\n--- Approach 3: PTQ, 256 rep, per-channel OFF ---")
    blob = _convert(fp32, rep_all_images, rep_size=256, per_channel=False)
    out = _eval_tflite(blob, sample_images)
    r = _report("PTQ-256-perchan-OFF", blob, out, fp32_preds)
    results.append(r)

    # Approach 4: PTQ on the QAT model (use QAT's calibrated weights as seed)
    print("\n--- Approach 4: PTQ on QAT model, 256 rep ---")
    try:
        qat = keras.models.load_model(ART / "model_qat.keras", compile=False)
        # The QAT model has quant_* prefixes on output names; we strip them by
        # using a forward pass to get a functional model.
        blob = _convert(qat, rep_all_images, rep_size=256, per_channel=True)
        out = _eval_tflite(blob, sample_images)
        r = _report("PTQ-on-QAT-256", blob, out, fp32_preds)
        results.append(r)
    except Exception as e:
        print(f"  Skipped: {e}")

    # Summary
    print("\n\n=== SUMMARY ===")
    for r in results:
        print(f"\n{r['name']} ({r['size_kb']} KB):")
        for row in r["rows"]:
            ok = "OK" if row["tflite_std"] > 1e-4 else "COLLAPSED"
            print(f"  {row['name']}: std={row['tflite_std']:.5f} [{ok}], "
                  f"max-min={row['tflite_max_min']:.5f}, "
                  f"diff={row['max_diff_vs_fp32']:.5f}")


if __name__ == "__main__":
    sys.exit(main())
