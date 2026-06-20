#!/usr/bin/env python3
"""Export FP32 warmup model to TFLite int8 (post-training quantization).

Since QARepVGGBlock contains only standard layers (Conv2D, BN, ReLU, Add),
the TFLite converter should handle int8 quantization natively.

Usage:
  cd ml && poetry run python scripts/export_tflite_qarepvgg.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras

os.environ["TF_GPU_MEMORY_LIMIT_MB"] = "2048"
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)],
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.tf_models import build_qarepvgg_mini


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
HEATMAP_SIZE = 32


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="artifacts/training/qat_qarepvgg_mini_20260613_203310/best_warmup.keras")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}"); sys.exit(1)
    run_dir = ckpt_path.parent

    print(f"Building model...")
    model = build_qarepvgg_mini((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    model.load_weights(ckpt_path)
    print(f"  Params: {model.count_params():,}")
    print(f"  Estimated FP32 size: {model.count_params() * 4 / 1024:.0f} KB")

    # Build representative dataset from manifest images
    # Paths in manifest are relative to repo root
    REPO_ROOT = PROJECT_ROOT.parent  # ml/ → repo root
    manifest_path = REPO_ROOT / "ml" / "data" / "merged_geometry_board_manifest.csv"
    rep_paths = []
    if manifest_path.exists():
        import csv
        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fpath = REPO_ROOT / row["image_path"]
                if fpath.exists():
                    rep_paths.append(str(fpath))
                    if len(rep_paths) >= 100:
                        break

    print(f"  Representative dataset: {len(rep_paths)} images")

    def rep_dataset():
        for p in rep_paths[:100]:
            img = keras.utils.load_img(p, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
            arr = keras.utils.img_to_array(img) / 255.0
            yield [arr[tf.newaxis, ...].astype(np.float32)]

    print("\nConverting to TFLite int8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_buf = converter.convert()
    tflite_path = run_dir / "qarepvgg_mini_int8.tflite"
    tflite_path.write_bytes(tflite_buf)
    size_kb = len(tflite_buf) / 1024
    print(f"\n  TFLite int8: {tflite_path}")
    print(f"  Size: {size_kb:.1f} KB ({size_kb / 1024:.2f} MB)")
    print(f"  Under 2.5 MB: {'YES' if size_kb < 2560 else 'NO'}")


if __name__ == "__main__":
    main()
