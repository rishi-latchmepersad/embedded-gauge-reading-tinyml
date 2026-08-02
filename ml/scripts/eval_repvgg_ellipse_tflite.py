#!/usr/bin/env python3
"""Evaluate the RepVGG ellipse detector: Keras vs TFLite parity, plus test-set metrics.

Three checks, in order of importance:
1. TFLite predictions track Keras predictions within a small error budget.
   A large gap here means the int8 quantization lost too much information
   and the model will misbehave on the board.
2. Center error in pixels on the val and test sets. We report mean absolute
   error and the percentage of predictions within 4 px and 8 px of the
   ground truth. A usable detector should be > 90% within 8 px.
3. Radius variance on the test set. If the radius output has collapsed
   to a constant we will see variance ~ 0; that is the failure mode the
   linear-head trick is supposed to prevent.

Why a separate script: the training script reports the int8 model size and
peak activation, but it does not run the TFLite interpreter. This script
does. Run it after the training script finishes, before you consider the
model a deployment candidate.
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

from embedded_gauge_reading_tinyml.ellipse_repvgg import (  # noqa: E402
    build_repvgg_ellipse_fused,
    reparameterize_model,
)

IMAGE_SIZE = 640
DATA_DIR = ROOT / "data" / "repvgg_ellipse"


def _load_split(split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a single split from disk (mirrors the training script)."""
    labels = json.loads((DATA_DIR / split / "labels.json").read_text())
    from PIL import Image
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


def _quantize_input(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantize a float32 input to int8 using the model's scale + zero point."""
    return np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)


def _dequantize_output(y: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize an int8 output back to float32."""
    return (y.astype(np.float32) - zero_point) * scale


def parity_check(tflite_path: Path, keras_model: keras.Model,
                 sample: np.ndarray, n: int = 100) -> dict:
    """Run the TFLite model and the Keras model on the same `n` images and compare.

    Returns the per-output mean abs error and the max abs error. The TFLite
    outputs are dequantized using the model's output quantization params
    so we are comparing in the float32 domain.
    """
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    in_scale, in_zp = in_det["quantization"]

    n = min(n, len(sample))
    sample = sample[:n]
    y_keras = keras_model.predict(sample, verbose=0)

    # Map each tflite output detail to the matching Keras output by output
    # dimension. The tflite output names are positional "StatefulPartitionedCall:N"
    # so we cannot rely on the name, only on the (batch, dim) shape.
    keras_by_dim = {y.shape[1]: y for y in y_keras}
    tflite_by_dim = {}
    for det in out_dets:
        # det["shape"] is (1, dim) at this point. Use the dim as the key.
        dim = int(det["shape"][-1])
        tflite_by_dim[dim] = det

    # Allocate one (n, dim) buffer per output, keyed by dim.
    buffers = {dim: np.zeros((n, dim), dtype=np.float32) for dim in tflite_by_dim}

    for i in range(n):
        xq = _quantize_input(sample[i:i+1], in_scale, in_zp)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        for dim, det in tflite_by_dim.items():
            raw = interp.get_tensor(det["index"])
            if det["dtype"] == np.int8:
                scale, zp = det["quantization"]
                raw = _dequantize_output(raw, scale, zp)
            # raw has shape (1, dim); assign the per-sample slice.
            buffers[dim][i] = raw[0]

    diffs = []
    names = ["center_xy", "radius_xy", "confidence"]
    for name, y_k in zip(names, y_keras):
        dim = int(y_k.shape[1])
        diffs.append({
            "name": name,
            "mean_abs_diff": float(np.mean(np.abs(y_k - buffers[dim]))),
            "max_abs_diff": float(np.max(np.abs(y_k - buffers[dim]))),
        })
    return {
        "n_samples": n,
        "per_output": diffs,
        "max_abs_diff_overall": float(max(d["max_abs_diff"] for d in diffs)),
    }


def eval_split_tflite(tflite_path: Path, x: np.ndarray,
                       y: dict[str, np.ndarray], name: str) -> dict:
    """Score the int8 TFLite model on a split, in pixel units.

    This is what the board will actually see — the int8 model, not the
    Keras float32 model. We re-quantize each input on the fly and read
    the dequantized outputs.
    """
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    in_scale, in_zp = in_det["quantization"]

    # Map outputs to the (n, dim) buffer.
    buffers = {int(d["shape"][-1]): np.zeros((len(x), int(d["shape"][-1])), np.float32)
               for d in out_dets}
    for i in range(len(x)):
        xq = _quantize_input(x[i:i+1], in_scale, in_zp)
        interp.set_tensor(in_det["index"], xq)
        interp.invoke()
        for det in out_dets:
            dim = int(det["shape"][-1])
            raw = interp.get_tensor(det["index"])
            if det["dtype"] == np.int8:
                scale, zp = det["quantization"]
                raw = _dequantize_output(raw, scale, zp)
            buffers[dim][i] = raw[0]

    center_pred = buffers[2]  # 2-dim output
    radius_pred = buffers[2]  # also 2-dim - we need to disambiguate by output index
    confidence_pred = buffers[1]

    # We have two outputs of dim 2 (center_xy and radius_xy). Re-run the
    # interpreter with a known input to figure out which is which by
    # matching against the Keras output ordering: center is the first
    # 2-dim output, radius is the second.
    center_pred = buffers[2]
    # Find both 2-dim outputs in the order they appear in out_dets.
    two_dim_outputs = [d for d in out_dets if int(d["shape"][-1]) == 2]
    if len(two_dim_outputs) >= 2:
        first_det = two_dim_outputs[0]
        second_det = two_dim_outputs[1]
        center_buf = np.zeros((len(x), 2), np.float32)
        radius_buf = np.zeros((len(x), 2), np.float32)
        for i in range(len(x)):
            xq = _quantize_input(x[i:i+1], in_scale, in_zp)
            interp.set_tensor(in_det["index"], xq)
            interp.invoke()
            for det, buf in [(first_det, center_buf), (second_det, radius_buf)]:
                raw = interp.get_tensor(det["index"])
                if det["dtype"] == np.int8:
                    s, z = det["quantization"]
                    raw = _dequantize_output(raw, s, z)
                buf[i] = raw[0]
        center_pred, radius_pred = center_buf, radius_buf

    # Pixel error for the centre.
    center_err_px = np.linalg.norm(
        (center_pred - y["center_xy"]) * IMAGE_SIZE, axis=1,
    )
    radius_pred_px = radius_pred * IMAGE_SIZE
    radius_gt_px = y["radius_xy"] * IMAGE_SIZE
    radius_err_px = np.linalg.norm(radius_pred_px - radius_gt_px, axis=1)

    return {
        "split": name,
        "n": int(len(x)),
        "model": "tflite_int8",
        "center_mae_px": float(np.mean(center_err_px)),
        "center_median_px": float(np.median(center_err_px)),
        "center_pct_le_4px": float(np.mean(center_err_px <= 4.0)),
        "center_pct_le_8px": float(np.mean(center_err_px <= 8.0)),
        "radius_mae_px": float(np.mean(radius_err_px)),
        "radius_pred_variance": float(np.var(radius_pred_px[:, 0])),
        "radius_gt_variance": float(np.var(radius_gt_px[:, 0])),
    }


def eval_split(keras_model: keras.Model, x: np.ndarray, y: dict[str, np.ndarray],
                name: str) -> dict:
    """Score the Keras (float32) model on a split, in pixel units."""
    preds = keras_model.predict(x, batch_size=8, verbose=0)
    center_pred = preds[0]  # (N, 2) in [0,1]
    radius_pred = preds[1]  # (N, 2) in normalised radii

    # Pixel error for the centre.
    center_err_px = np.linalg.norm(
        (center_pred - y["center_xy"]) * IMAGE_SIZE, axis=1,
    )
    radius_pred_px = radius_pred * IMAGE_SIZE
    radius_gt_px = y["radius_xy"] * IMAGE_SIZE
    radius_err_px = np.linalg.norm(radius_pred_px - radius_gt_px, axis=1)

    return {
        "split": name,
        "n": int(len(x)),
        "model": "keras_fp32",
        "center_mae_px": float(np.mean(center_err_px)),
        "center_median_px": float(np.median(center_err_px)),
        "center_pct_le_4px": float(np.mean(center_err_px <= 4.0)),
        "center_pct_le_8px": float(np.mean(center_err_px <= 8.0)),
        "radius_mae_px": float(np.mean(radius_err_px)),
        "radius_pred_variance": float(np.var(radius_pred_px[:, 0])),
        "radius_gt_variance": float(np.var(radius_gt_px[:, 0])),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path,
                        default=ROOT / "artifacts" / "gauge_ellipse_repvgg_640g_v1")
    parser.add_argument("--parity-n", type=int, default=100)
    parser.add_argument("--tflite", type=str, default="model_int8.tflite",
                        help="Which TFLite artifact to evaluate.")
    args = parser.parse_args()

    # Load the trained model and the matching TFLite. The eval script is
    # named for the RepVGG effort but works for any Keras model that has
    # a `model_fp32.keras` or `model_fused.keras` artifact. We load with
    # compile=False to avoid the LR-schedule deserialization problem.
    fused_path = args.artifacts / "model_fused.keras"
    fp32_path = args.artifacts / "model_fp32.keras"
    if fused_path.exists():
        fused = keras.models.load_model(fused_path, compile=False)
    elif fp32_path.exists():
        fused = keras.models.load_model(fp32_path, compile=False)
    else:
        raise FileNotFoundError(f"No model_fused.keras or model_fp32.keras in {args.artifacts}")
    tflite_path = args.artifacts / args.tflite
    if not tflite_path.exists():
        raise FileNotFoundError(f"Missing {tflite_path}")

    # Parity check.
    print("Running Keras vs TFLite parity check...")
    val_x, val_y = _load_split("val")
    parity = parity_check(tflite_path, fused, val_x, n=args.parity_n)
    print(json.dumps(parity, indent=2))
    if parity["max_abs_diff_overall"] > 0.02:
        print(f"  WARNING: TFLite diverges from Keras by "
              f"{parity['max_abs_diff_overall']:.3f} on the worst output. "
              f"Inspect the int8 representative dataset.")

    # Score on val and test.
    print("\nEvaluating on val and test splits (Keras float32 model)...")
    test_x, test_y = _load_split("test")
    results = {
        "parity": parity,
        "val_keras": eval_split(fused, val_x, val_y, "val"),
        "test_keras": eval_split(fused, test_x, test_y, "test"),
    }
    print("\nEvaluating on val and test splits (TFLite int8 model)...")
    results["val_tflite"] = eval_split_tflite(tflite_path, val_x, val_y, "val")
    results["test_tflite"] = eval_split_tflite(tflite_path, test_x, test_y, "test")
    print(json.dumps(results, indent=2))

    # Persist the eval report.
    out_path = args.artifacts / "eval_report.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
