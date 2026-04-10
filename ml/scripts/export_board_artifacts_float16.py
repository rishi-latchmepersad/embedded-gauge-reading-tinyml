"""Export the calibrated scalar CNN to a float16 TFLite artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import keras
import tensorflow as tf

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the float16 export job."""
    parser = argparse.ArgumentParser(
        description="Export the calibrated scalar CNN to a float16 TFLite artifact."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "training"
        / "scalar_full_finetune_from_best_board30_piecewise_calibrated"
        / "model.keras",
        help="Path to the calibrated scalar Keras model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "deployment"
        / "scalar_full_finetune_from_best_board30_piecewise_calibrated_float16",
        help="Directory where the float16 TFLite model should be written.",
    )
    parser.add_argument(
        "--legacy-mobilenetv2-preprocess",
        action="store_true",
        help="Load the saved model using the legacy MobileNetV2 preprocess Lambda symbol.",
    )
    return parser.parse_args()


def _load_model(model_path: Path, *, legacy_mobilenetv2_preprocess: bool) -> keras.Model:
    """Load the calibrated scalar model artifact."""
    custom_objects = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
    }
    if legacy_mobilenetv2_preprocess:
        print("[EXPORT] Legacy MobileNetV2 preprocess support enabled.", flush=True)
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )
    print(f"[EXPORT] Loaded model '{model.name}' from {model_path}.", flush=True)
    return model


def main() -> None:
    """Export the calibrated scalar CNN as a float16 TFLite model."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[EXPORT] Stage: load-model", flush=True)
    model = _load_model(
        args.model,
        legacy_mobilenetv2_preprocess=args.legacy_mobilenetv2_preprocess,
    )
    print("[EXPORT] Stage: convert-to-float16", flush=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_path = args.output_dir / "model_float16.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"[EXPORT] Wrote float16 TFLite model to {tflite_path}.", flush=True)


if __name__ == "__main__":
    main()
