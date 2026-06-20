#!/usr/bin/env python3
"""QAT fine-tune the v4 112 geometry heatmap model to reduce INT8 quantization drift.

Rebuilds the model in tf_keras (Keras v2), copies weights from the Keras v3
checkpoint, applies tfmot quantization-aware training, fine-tunes, and exports.
"""

from __future__ import annotations

import argparse
import os as _os
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Cap GPU allocation before TensorFlow initializes the runtime, then load the
# tf_keras stack so tfmot sees the same configured device.
import tensorflow as tf
_GPU_MEMORY_LIMIT_MB = int(_os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "3800"))
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=_GPU_MEMORY_LIMIT_MB)],
    )
del _os, _GPU_MEMORY_LIMIT_MB

# Force tf_keras as keras for tfmot compatibility.
import tf_keras as keras
import tensorflow_model_optimization as tfmot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import summarize_tflite_contract
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    load_clean_geometry_examples,
    load_heatmap_sample,
    sample_jitter_params,
    select_examples_from_split,
)

SELECTED_PREPROCESSING_MODE = "python_training_rgb_bilinear"
INPUT_SIZE = 224
HEATMAP_SIZE = 112
SIGMA_PIXELS = 2.5
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 5e-5


def _build_tfkeras_model() -> keras.Model:
    """Build the MobileNetV2 geometry heatmap v4_112 model with tf_keras layers."""
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name="input_image")

    # Backbone: MobileNetV2 alpha=0.35
    backbone = keras.applications.MobileNetV2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        alpha=0.35,
        include_top=False,
        weights=None,
        pooling=None,
        input_tensor=inputs,
    )
    # Extract skip and final features
    skip_56 = backbone.get_layer("block_3_expand_relu").output  # 56x56
    x = backbone.output  # 7x7x1280

    # Decoder
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="geometry_decoder_conv_1")(x)
    x = keras.layers.UpSampling2D(2, interpolation="bilinear", name="geometry_decoder_up_1")(x)  # 14x14
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="geometry_decoder_conv_2")(x)
    x = keras.layers.UpSampling2D(2, interpolation="bilinear", name="geometry_decoder_up_2")(x)  # 28x28
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="geometry_decoder_conv_3")(x)
    x = keras.layers.UpSampling2D(2, interpolation="bilinear", name="geometry_decoder_up_3")(x)  # 56x56

    # Skip connection
    skip = keras.layers.Conv2D(16, 1, padding="same", activation="relu", name="geometry_decoder_skip_56")(skip_56)
    x = keras.layers.Concatenate(name="geometry_decoder_concat_56")([x, skip])

    # Refine
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="geometry_decoder_refine_112")(x)
    x = keras.layers.UpSampling2D(2, interpolation="bilinear", name="geometry_decoder_up_4")(x)  # 112x112

    # Output heads
    center_heatmap = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="center_heatmap")(x)
    tip_heatmap = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="tip_heatmap")(x)
    gap = keras.layers.GlobalAveragePooling2D(name="geometry_confidence_gap")(x)
    confidence = keras.layers.Dense(1, activation="sigmoid", name="confidence")(gap)

    return keras.Model(inputs=inputs, outputs=[center_heatmap, tip_heatmap, confidence],
                       name="mobilenetv2_geometry_heatmap_v4_112")


def _find_v3_layer(v3_model: Any, name: str) -> Any | None:
    """Recursively search for a layer by name in v3_model (handles sub-models)."""
    for layer in v3_model.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, 'layers'):
            found = _find_v3_layer(layer, name)
            if found is not None:
                return found
    return None


def _copy_weights_keras3_to_tfkeras(v3_model: Any, tfk_model: keras.Model) -> None:
    """Copy weights from a Keras v3 model to a tf_keras model layer by layer."""
    copied = 0
    skipped = 0
    failed = 0
    for tfk_layer in tfk_model.layers:
        name = tfk_layer.name
        v3_layer = _find_v3_layer(v3_model, name)
        if v3_layer is None:
            skipped += 1
            continue
        v3_weights = v3_layer.get_weights()
        if not v3_weights:
            continue
        try:
            tfk_layer.set_weights(v3_weights)
            copied += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"  Weight transfer: {copied} OK, {failed} failed, {skipped} skipped (no weights)")


def _build_training_generator(train_examples: list[Any], base_path: Path, *, batch_size: int):
    rng = np.random.default_rng(42)
    while True:
        batch_inp, batch_center, batch_tip, batch_conf = [], [], [], []
        for example in train_examples:
            jitter = sample_jitter_params(
                np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
                shift_min_px=3, shift_max_px=7,
                scale_min=0.95, scale_max=1.08,
                aspect_min=0.93, aspect_max=1.07,
            )
            sample = load_heatmap_sample(
                example, base_path, input_size=INPUT_SIZE, heatmap_size=HEATMAP_SIZE,
                sigma_pixels=SIGMA_PIXELS, jitter=jitter,
            )
            batch_inp.append(np.asarray(sample.crop_image, dtype=np.float32) * 2.0 - 1.0)
            batch_center.append(np.asarray(sample.center_heatmap, dtype=np.float32))
            batch_tip.append(np.asarray(sample.tip_heatmap, dtype=np.float32))
            batch_conf.append(np.asarray([1.0], dtype=np.float32))  # shape (1,) -> stack to (batch, 1)
            if len(batch_inp) >= batch_size:
                yield (np.stack(batch_inp, axis=0),
                       (np.stack(batch_center, axis=0),
                        np.stack(batch_tip, axis=0),
                        np.stack(batch_conf, axis=0)))
                batch_inp, batch_center, batch_tip, batch_conf = [], [], [], []
        if batch_inp:
            yield (np.stack(batch_inp, axis=0),
                   (np.stack(batch_center, axis=0),
                    np.stack(batch_tip, axis=0),
                    np.stack(batch_conf, axis=0)))


def _representative_dataset(rep_examples: list[Any], base_path: Path) -> Iterable[list[np.ndarray]]:
    for example in rep_examples:
        sample = load_heatmap_sample(
            example, base_path, input_size=INPUT_SIZE, heatmap_size=HEATMAP_SIZE,
            sigma_pixels=SIGMA_PIXELS, jitter=None,
        )
        yield [np.expand_dims(np.asarray(sample.crop_image, dtype=np.float32), axis=0)]


def main() -> None:
    parser = argparse.ArgumentParser(description="QAT fine-tune v4 112 geometry heatmap model")
    parser.add_argument("--model-path", type=str,
                        default="ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras")
    parser.add_argument("--output-dir", type=str, default="ml/artifacts/deployment/geometry_heatmap_v4_112_qat")
    parser.add_argument("--manifest-path", type=str, default="ml/data/geometry_reader_manifest_v2_clean.csv")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = repo_root / args.model_path if not Path(args.model_path).is_absolute() else Path(args.model_path)
    manifest_path = repo_root / args.manifest_path if not Path(args.manifest_path).is_absolute() else Path(args.manifest_path)
    output_dir = repo_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Keras v3 model to extract weights
    print("[QAT] Loading Keras v3 model for weight transfer...")
    from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import load_geometry_heatmap_keras_model
    v3_model = load_geometry_heatmap_keras_model(model_path)
    print(f"  Loaded: {type(v3_model).__name__}")

    # Step 2: Rebuild in tf_keras
    print("[QAT] Rebuilding model in tf_keras...")
    tfk_model = _build_tfkeras_model()
    print("  Copying weights...")
    _copy_weights_keras3_to_tfkeras(v3_model, tfk_model)
    del v3_model  # free memory

    # Step 3: Verify
    print("  Verifying forward pass...")
    dummy = np.random.randn(1, 224, 224, 3).astype(np.float32)
    out = tfk_model(dummy, training=False)
    print(f"  Output shapes: center={out[0].shape}, tip={out[1].shape}, conf={out[2].shape}")

    # Step 4: Apply QAT
    print("[QAT] Applying quantization-aware training wrapper...")
    qat_model = tfmot.quantization.keras.quantize_model(tfk_model)
    print("  QAT model ready")

    # Step 5: Compile
    # Use positional loss list (order matches output heads: center, tip, confidence)
    # The QAT wrapper renames outputs with quant_ prefix, so positional is more robust
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=[
            keras.losses.MeanSquaredError(),
            keras.losses.MeanSquaredError(),
            keras.losses.BinaryCrossentropy(),
        ],
        loss_weights=[1.0, 1.0, 0.1],
    )

    # Step 6: Load training data
    print("[QAT] Loading train split...")
    examples = load_clean_geometry_examples(manifest_path)
    train_examples = select_examples_from_split(examples, split="train")
    print(f"  {len(train_examples)} clean train rows")

    # Step 7: Fine-tune
    train_gen = _build_training_generator(train_examples, repo_root, batch_size=args.batch_size)
    print(f"[QAT] Fine-tuning {args.epochs} epochs, {args.steps_per_epoch} steps/epoch, LR={args.learning_rate}...")
    history = qat_model.fit(train_gen, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, verbose=1)

    # Save training history
    history_path = output_dir / "qat_history.json"
    history_path.write_text(json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}), encoding="utf-8")

    # Step 8: Convert to INT8 TFLite (QAT-aware)
    print("[QAT] Converting to INT8 TFLite...")
    rep_examples = list(train_examples)
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(rep_examples, repo_root)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_bytes = converter.convert()
    int8_path = output_dir / "model_v4_112_qat_int8.tflite"
    int8_path.write_bytes(int8_bytes)
    print(f"  INT8 -> {int8_path} ({int8_path.stat().st_size / 1024:.0f} KB)")

    # Also FP32
    print("[QAT] Converting to FP32 TFLite...")
    fp32_bytes = tf.lite.TFLiteConverter.from_keras_model(qat_model).convert()
    fp32_path = output_dir / "model_v4_112_qat_float32.tflite"
    fp32_path.write_bytes(fp32_bytes)
    print(f"  FP32 -> {fp32_path} ({fp32_path.stat().st_size / 1024:.0f} KB)")

    # Save QAT Keras model
    qat_keras_path = output_dir / "model_v4_112_qat.keras"
    qat_model.save(str(qat_keras_path))
    print(f"  QAT Keras -> {qat_keras_path}")

    # Contract
    contract = summarize_tflite_contract(int8_path)
    # Detect output ordering: find which raw output is the confidence scalar
    conf_idx = next(i for i, o in enumerate(contract["outputs"]) if len(o["shape"]) == 2 and o["shape"][1] == 1)
    heatmap_idxs = [i for i, o in enumerate(contract["outputs"]) if len(o["shape"]) == 4]
    contract["semantic_output_order_indices"] = [heatmap_idxs[0], heatmap_idxs[1], conf_idx]
    contract["decoder"] = {"decode_method": "softargmax", "window_size": 3}
    contract["semantic_output_names"] = ["center_heatmap", "tip_heatmap", "confidence"]
    contract["qat_epochs"] = args.epochs
    (output_dir / "tflite_tensor_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

    # Config
    (output_dir / "qat_config.json").write_text(
        json.dumps({
            "model_source": str(model_path),
            "qat_epochs": args.epochs,
            "qat_batch_size": args.batch_size,
            "qat_learning_rate": args.learning_rate,
            "steps_per_epoch": args.steps_per_epoch,
            "input_size": INPUT_SIZE,
            "heatmap_size": HEATMAP_SIZE,
            "sigma_pixels": SIGMA_PIXELS,
            "preprocessing_mode": SELECTED_PREPROCESSING_MODE,
            "train_samples": len(train_examples),
        }, indent=2, sort_keys=True), encoding="utf-8")

    print("[QAT] Done")


if __name__ == "__main__":
    main()
