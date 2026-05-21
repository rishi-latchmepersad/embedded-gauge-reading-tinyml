"""Export the repacked V28 polar-vote model to int8 TFLite for STM32N6 deployment.

Loads the .keras checkpoint (which uses custom ReduceMeanAxis / ReduceMaxAxis
layers), quantises the full graph to int8 using a small random representative
dataset, and writes the resulting .tflite flatbuffer to the deployment
artefacts directory.  Tensor details (name, shape, dtype, scale, zero-point)
and the final model size in bytes are printed at the end.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

# Suppress noisy TF / CUDA logs before any TF import side-effects.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project root -- resolves relative to this script so the defaults work
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL = (
    _REPO_ROOT
    / "ml"
    / "artifacts"
    / "runtime"
    / "polar_vote_circular_v28_repack"
    / "model.keras"
)
_DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT
    / "ml"
    / "artifacts"
    / "deployment"
    / "polar_vote_circular_v28_int8"
)


# -- Custom serialisable layers (must match the repack script) ---------------


class ReduceMeanAxis(tf.keras.layers.Layer):
    """Compute tf.reduce_mean over a configurable axis.

    Registered as a custom object so that load_model can deserialise the
    repacked V28 checkpoint which uses this layer instead of a Lambda.
    """

    def __init__(self, axis: int = -1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.axis = axis  # axis along which to average

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Collapse the stored axis by averaging.
        return tf.reduce_mean(inputs, axis=self.axis)

    def get_config(self) -> dict:
        # Include axis so the layer can be re-instantiated from a saved model.
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class ReduceMaxAxis(tf.keras.layers.Layer):
    """Compute tf.reduce_max over a configurable axis.

    Mirrors ReduceMeanAxis but takes the maximum instead of the mean.
    """

    def __init__(self, axis: int = -1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.axis = axis  # axis along which to take the max

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Collapse the stored axis by taking the maximum.
        return tf.reduce_max(inputs, axis=self.axis)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


# -- Representative dataset ------------------------------------------------


def _representative_dataset(
    num_samples: int = 32,
    input_shape: tuple[int, ...] = (1, 224, 224, 7),
) -> tf.lite.RepresentativeDataset:
    """Yield num_samples random float32 tensors for int8 calibration.

    Uses a fixed seed so the calibration is reproducible across runs.  The
    random data is not meaningful for the model, but it provides a reasonable
    distribution of activations that the TFLite converter uses to determine
    quantisation ranges.
    """

    rng = np.random.default_rng(seed=42)
    samples = [rng.random(input_shape).astype(np.float32) for _ in range(num_samples)]

    def _gen():
        for sample in samples:
            yield [sample]

    return _gen


# -- Main entry point -----------------------------------------------------


def main() -> None:
    """Load, convert, and save the int8 TFLite model."""
    parser = argparse.ArgumentParser(
        description="Export repacked V28 polar-vote model to int8 TFLite."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_DEFAULT_MODEL,
        help="Path to the repacked .keras model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory to write the int8 TFLite model into",
    )
    args = parser.parse_args()

    source_path: Path = args.source
    output_dir: Path = args.output_dir
    output_path = output_dir / "model_int8.tflite"

    # --- 1. Load the repacked model with custom layers ------------------------
    print(f"[EXPORT] Loading model from {source_path} ...")
    model = tf.keras.models.load_model(
        str(source_path),
        custom_objects={
            "ReduceMeanAxis": ReduceMeanAxis,
            "ReduceMaxAxis": ReduceMaxAxis,
        },
        compile=False,
    )
    print(f"[EXPORT] Model: {model.name}")
    print(f"[EXPORT] Inputs:  {[i.shape for i in model.inputs]}")
    print(f"[EXPORT] Outputs: {[o.shape for o in model.outputs]}")

    # --- 2. Convert to int8 TFLite ------------------------------------
    print("[EXPORT] Converting to int8 TFLite ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Default optimisation with int8 weights and activations.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Representative dataset for activation quantisation ranges.
    converter.representative_dataset = _representative_dataset(
        num_samples=32, input_shape=(1, 224, 224, 7)
    )
    # Require full int8 ops (no hybrid fallback).
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Force int8 input / output tensors.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model: bytes = converter.convert()

    # --- 3. Save ------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[EXPORT] Saved int8 TFLite model to {output_path}")

    # --- 4. Print tensor details ---------------------------------------------
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n--- Input tensors ---")
    for info in input_details:
        name = info["name"]
        shape = info["shape"]
        dtype = info["dtype"]
        scales = info["quantization_parameters"]["scales"]
        zps = info["quantization_parameters"]["zero_points"]
        # Pull the per-tensor scale and zero-point (channelwise is empty for
        # fully-quantised int8 with a single scale per tensor).
        scale = float(scales[0]) if len(scales) > 0 else 0.0
        zp = int(zps[0]) if len(zps) > 0 else 0
        print(f"  {name}: shape={shape}, dtype={dtype}, scale={scale}, zero_point={zp}")

    print("\n--- Output tensors ---")
    for info in output_details:
        name = info["name"]
        shape = info["shape"]
        dtype = info["dtype"]
        scales = info["quantization_parameters"]["scales"]
        zps = info["quantization_parameters"]["zero_points"]
        scale = float(scales[0]) if len(scales) > 0 else 0.0
        zp = int(zps[0]) if len(zps) > 0 else 0
        print(f"  {name}: shape={shape}, dtype={dtype}, scale={scale}, zero_point={zp}")

    # --- 5. Print model size ------------------------------------------------
    model_size_bytes = len(tflite_model)
    print(f"\n[EXPORT] Model size: {model_size_bytes} bytes ({model_size_bytes / 1024:.1f} KB)")
    print("[EXPORT] Done.")


if __name__ == "__main__":
    main()
