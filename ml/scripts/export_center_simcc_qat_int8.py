#!/usr/bin/env python3
"""Export the board center/tip SimCC QAT model to an int8 TFLite artifact.

The QAT checkpoint loads successfully with ``tf.keras`` in this environment, so
we use that loader here and convert the trained model into a deployment-ready
int8 flatbuffer with a small representative dataset from the grouped manifest.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"

import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from embedded_gauge_reading_tinyml.quantize_compat import quantize_load_scope  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "training" / "obb_board_simcc_kd_qat_v2" / "center_simcc_qat.keras"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "tmp" / "labelled_captured_images_board_center_tip_v2.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "deployment" / "center_simcc_qat_int8"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_VAL_FRACTION = 0.15
IMAGE_SIZE = 224


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the export job."""

    parser = argparse.ArgumentParser(description="Export the board SimCC QAT model to int8 TFLite.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representative-batches", type=int, default=4)
    return parser.parse_args()


def _split_samples(
    samples: list[object],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[object], list[object], list[object]]:
    """Mirror the trainer's stratified split so the export sees the same mix."""

    from sklearn.model_selection import train_test_split

    labels = np.array([1 if sample.tip_xy is not None else 0 for sample in samples], dtype=np.int32)
    indices = np.arange(len(samples), dtype=np.int32)
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=max(test_fraction, 0.05),
        random_state=seed,
        shuffle=True,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    relative_val_fraction = val_fraction / max(1.0 - test_fraction, 1e-6)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=max(relative_val_fraction, 0.05),
        random_state=seed + 1,
        shuffle=True,
        stratify=labels[train_val_indices] if len(np.unique(labels[train_val_indices])) > 1 else None,
    )
    return (
        [samples[int(index)] for index in train_indices],
        [samples[int(index)] for index in val_indices],
        [samples[int(index)] for index in test_indices],
    )


def _load_model(model_path: Path) -> tf.keras.Model:
    """Load the saved QAT model with the quantization wrappers enabled."""

    with quantize_load_scope():
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)


def _representative_dataset(
    sequence: Any,
    *,
    max_batches: int,
) -> Iterable[list[np.ndarray]]:
    """Yield representative batches for TFLite calibration/export."""

    batch_limit = min(len(sequence), max_batches)
    for batch_index in range(batch_limit):
        batch_x, _batch_y, _weights = sequence[batch_index]
        for sample in batch_x:
            yield [np.asarray(sample[np.newaxis, ...], dtype=np.float32)]


def _export_int8_tflite(
    model: tf.keras.Model,
    output_path: Path,
    *,
    representative_sequence: Any,
    representative_batches: int,
) -> Path:
    """Convert the loaded QAT model into an int8 TFLite flatbuffer."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(
        representative_sequence,
        max_batches=representative_batches,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    return output_path


def main() -> None:
    """Load the trained QAT model and export an int8 TFLite file."""

    args = _parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    if args.representative_batches < 1:
        raise ValueError("--representative-batches must be >= 1.")

    with quantize_load_scope():
        model = tf.keras.models.load_model(args.model_path, compile=False, safe_mode=False)
    model = tfmot.quantization.keras.strip_quantization(model)

    os.environ.setdefault("TF_SKIP_EXPLICIT_GPU_CONFIG", "1")
    training = importlib.import_module("train_qat_obb_simcc_combined")
    CombinedGeometrySequence = training.CombinedGeometrySequence
    load_combined_samples = training.load_combined_samples

    samples = load_combined_samples(args.manifest, include_temperature_head=True)
    if not samples:
        raise ValueError(f"No usable samples were found in {args.manifest}.")

    train_samples, _val_samples, _test_samples = _split_samples(
        samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    representative_sequence = CombinedGeometrySequence(
        train_samples,
        batch_size=args.batch_size,
        include_temperature_head=True,
        augment=False,
        seed=args.seed,
    )
    tflite_path = args.output_dir / "center_simcc_qat_int8.tflite"
    _export_int8_tflite(
        model,
        tflite_path,
        representative_sequence=representative_sequence,
        representative_batches=args.representative_batches,
    )

    payload: dict[str, Any] = {
        "model_path": str(args.model_path),
        "manifest": str(args.manifest),
        "output_path": str(tflite_path),
        "sample_count": len(samples),
        "train_count": len(train_samples),
        "image_size": IMAGE_SIZE,
        "representative_batches": args.representative_batches,
    }
    summary_path = args.output_dir / "export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
