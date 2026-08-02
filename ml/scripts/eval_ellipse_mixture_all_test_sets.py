#!/usr/bin/env python3
"""Evaluate a domain-selecting int8 mixture ellipse model on all test zips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import FRAME_SIZE, _load_zip, _metrics


def predict_mixture(
    interpreter: tf.lite.Interpreter, images: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model and return selected ellipses, domains, and all heads."""
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    ellipses = np.zeros((len(images), 5), dtype=np.float32)
    all_heads = np.zeros((len(images), 3, 5), dtype=np.float32)
    domains = np.zeros(len(images), dtype=np.int32)
    for index, image in enumerate(images):
        quantized = np.clip(
            np.round(image[None] / input_scale + input_zero), -128, 127
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
        values = (raw - output_zero) * output_scale
        domain = int(np.argmax(values[15:18]))
        domains[index] = domain
        all_heads[index] = values[:15].reshape(3, 5)
        ellipses[index] = values[domain * 5 : domain * 5 + 5]
    return ellipses, domains, all_heads


def main() -> None:
    """Evaluate all held-out zips and write selected-domain geometry metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    report = {"model": str(args.model), "image_size": 384, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        predictions, domains, all_heads = predict_mixture(interpreter, images)
        metrics = _metrics(predictions, targets)
        metrics["selected_domain_counts"] = np.bincount(domains, minlength=3).tolist()
        metrics["forced_head_metrics"] = [
            _metrics(all_heads[:, head], targets) for head in range(3)
        ]
        report["tests"][zip_name] = metrics
        print(zip_name, json.dumps(metrics, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
