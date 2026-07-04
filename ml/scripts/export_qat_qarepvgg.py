#!/usr/bin/env python3
"""Export TFLite int8 from QARepVGG QAT checkpoint.

Usage: cd ml && poetry run python scripts/export_qat_qarepvgg.py
"""

import csv, os, sys
from pathlib import Path

import numpy as np
os.environ["TF_GPU_MEMORY_LIMIT_MB"] = "3900"
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.tf_models import build_qarepvgg_mini

IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
HEATMAP_SIZE = 40

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str,
        default="artifacts/training/qat_qarepvgg_mini_20260613_220243")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    qat_ckpt = run_dir / "best_qat.keras"
    if not qat_ckpt.exists():
        print(f"Checkpoint not found: {qat_ckpt}"); sys.exit(1)

    print("Building FP32 model...")
    fp32 = build_qarepvgg_mini((IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    print("Wrapping with QAT...")
    qat_model = tfmot.quantization.keras.quantize_model(fp32)
    qat_model.load_weights(qat_ckpt)
    print(f"  QAT params: {qat_model.count_params():,}")

    # Representative dataset from manifest
    REPO_ROOT = PROJECT_ROOT.parent
    manifest_path = REPO_ROOT / "ml" / "data" / "merged_geometry_board_manifest.csv"
    rep_paths = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fpath = REPO_ROOT / row["image_path"]
            if fpath.exists():
                rep_paths.append(str(fpath))
                if len(rep_paths) >= 100:
                    break
    print(f"  Representative: {len(rep_paths)} images")

    def rep_dataset():
        for p in rep_paths:
            img = keras.utils.load_img(p, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
            arr = keras.utils.img_to_array(img) / 255.0
            yield [arr[tf.newaxis, ...].astype(np.float32)]

    print("Converting to TFLite int8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_buf = converter.convert()
    tflite_path = run_dir / "qarepvgg_mini_qat_int8.tflite"
    tflite_path.write_bytes(tflite_buf)
    size_kb = len(tflite_buf) / 1024
    print(f"\n  TFLite: {tflite_path}")
    print(f"  Size:   {size_kb:.1f} KB ({size_kb/1024:.2f} MB)")
    print(f"  Under 2.5 MB: {'YES' if size_kb < 2560 else 'NO'}")

if __name__ == "__main__":
    main()
