#!/usr/bin/env python3
"""Compare saved QAT Keras outputs with fully integer TFLite outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from eval_ellipse_all_test_sets import _load_zip
from train_ellipse_center_heatmap_640 import build_model as build_high
from train_ellipse_domain_classifier_640 import build_model as build_classifier
from train_ellipse_extrema_heatmaps_640 import build_model as build_extrema
from train_ellipse_domain_heatmaps_640 import DomainHeatmapLoss, build_model as build_domain
from train_ellipse_mask_all_domains_384 import build_model as build_low
from train_ellipse_mask_640_center import build_model as build_mask_640
from train_ellipse_scalar_640 import resize_cpu


def main() -> None:
    """Run parity for one model family and print absolute-error statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("low", "high", "domain", "classifier", "extrema", "mask640"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    builder = {"low": build_low, "high": build_high, "domain": build_domain, "classifier": build_classifier, "extrema": build_extrema, "mask640": build_mask_640}[args.kind]
    # Prefer the complete saved QAT graph for the domain family.  Rebuilding
    # and loading only weights is unsafe when TFMOT wrapper variables differ.
    saved_model = args.artifact / "model_qat.keras"
    if args.kind == "domain" and saved_model.exists():
        qat = keras.models.load_model(saved_model, custom_objects={"DomainHeatmapLoss": DomainHeatmapLoss})
    else:
        qat = tfmot.quantization.keras.quantize_model(builder())
        qat.load_weights(args.artifact / "model_qat.weights.h5")
    images, _ = _load_zip("test_3.zip")
    if args.kind == "low":
        images = images[:8]
    elif args.kind == "mask640":
        images = tf.image.resize(images[:8], (640, 640)).numpy()
    else:
        images = resize_cpu(images[:8])
    keras_values = qat(images, training=False).numpy()
    interpreter = tf.lite.Interpreter(model_path=str(args.artifact / "model_int8.tflite"))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    tflite_values = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])[0].astype(np.float32)
        tflite_values.append((raw - out_zero) * out_scale)
    tflite_values = np.asarray(tflite_values)
    error = np.abs(keras_values - tflite_values)
    print({"kind": args.kind, "samples": len(images), "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()), "p99_abs_error": float(np.quantile(error, 0.99))})


if __name__ == "__main__":
    main()
