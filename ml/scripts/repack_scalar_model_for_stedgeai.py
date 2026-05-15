"""Repack the scalar gauge model into a CubeAI-friendly Keras artifact.

This script loads the best calibrated scalar MobileNetV2 model, rebuilds an
equivalent graph that uses only standard serializable layers, and copies the
trained weights across. The goal is to remove the legacy preprocess Lambda so
the STM32 X-CUBE-AI toolchain can load and compress the model cleanly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

import keras
import h5py
import tensorflow as tf
import numpy as np

# Add `ml/src` to sys.path so this script works from the `ml/` directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.presets import DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the repack job."""
    parser = argparse.ArgumentParser(
        description=(
            "Repack the calibrated scalar MobileNetV2 model into a Keras artifact "
            "that uses only standard serializable layers."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "training"
        / "scalar_hardcase_boost_v1_calibrated"
        / "model.keras",
        help="Path to the source calibrated scalar Keras model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "runtime"
        / "scalar_hardcase_boost_v1_calibrated_clean",
        help="Directory where the CubeAI-friendly model should be written.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Input image height for the rebuilt model.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Input image width for the rebuilt model.",
    )
    parser.add_argument(
        "--legacy-mobilenetv2-preprocess",
        action="store_true",
        help=(
            "Load older MobileNetV2 artifacts that use a non-serializable preprocess "
            "Lambda symbol."
        ),
    )
    return parser.parse_args()


def _load_model(model_path: Path, *, legacy_mobilenetv2_preprocess: bool) -> keras.Model:
    """Load the source model artifact from disk."""
    print(f"[REPACK] Loading source model from {model_path}...", flush=True)
    custom_objects: dict[str, Any] = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
    }
    if legacy_mobilenetv2_preprocess:
        print("[REPACK] Legacy MobileNetV2 preprocess support enabled.", flush=True)
    # We only need the trained weights for deployment, so avoid deserializing
    # the training-time compile state and any custom loss symbols.
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )
    print(f"[REPACK] Loaded model '{model.name}'.", flush=True)
    model.summary()
    return model


def _infer_piecewise_knot_count(source_model: keras.Model) -> int:
    """Infer how many learned spline knots the calibrated source model uses."""
    return sum(
        1 for layer in source_model.layers if layer.name.startswith("value_calibration_shift_")
    )


def _weight_key(name: str) -> str:
    """Normalize a TensorFlow variable name down to a stable layer-local key."""
    # Keras variables may be exposed as either bare names such as `kernel` or
    # scoped names such as `dense/kernel:0`. We only care about the layer-local
    # suffix when matching quantized source tensors to the clean deployment graph.
    return name.rsplit("/", 1)[-1].split(":", 1)[0]


def _build_clean_model(
    image_height: int,
    image_width: int,
    *,
    piecewise_knot_count: int,
) -> keras.Model:
    """Build a serializable graph that mirrors the loaded scalar model.

    Older prod v0.4 exports carried a piecewise-linear calibration head, but the
    current calibration-free prod model does not. We therefore keep the common
    MobileNetV2 backbone and only add the calibration basis layers when the
    source model actually contains them.
    """
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    # Mirror the legacy preprocessing contract in a serializable way.
    x = keras.layers.Rescaling(255.0, name="to_255")(inputs)
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(x)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(image_height, image_width, 3),
        name="mobilenetv2_1.00_224",
    )
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = keras.layers.Dropout(0.2, name="dropout")(x)
    x = keras.layers.Dense(128, activation="swish", name="dense")(x)
    x = keras.layers.Dropout(0.2, name="dropout_1")(x)
    base_output = keras.layers.Dense(1, name="gauge_value")(x)

    if piecewise_knot_count < 1:
        output = base_output
    else:
        basis_terms: list[keras.layers.Layer] = [base_output]
        for knot_index in range(piecewise_knot_count):
            shifted = keras.layers.Dense(
                1,
                use_bias=True,
                kernel_initializer=keras.initializers.Constant([[1.0]]),
                bias_initializer="zeros",
                trainable=False,
                name=f"value_calibration_shift_{knot_index}",
            )(base_output)
            basis_terms.append(
                keras.layers.ReLU(name=f"value_calibration_relu_{knot_index}")(shifted)
            )

        features = keras.layers.Concatenate(name="value_calibration_features")(basis_terms)
        output = keras.layers.Dense(1, name="value_calibration")(features)

    return keras.Model(
        inputs=inputs,
        outputs=output,
        name="mobilenetv2_gauge_regressor_piecewise_calibrated_clean",
    )


def _transfer_weights(source_model: keras.Model, target_model: keras.Model) -> None:
    """Copy trained weights from the loaded model into the clean graph.

    The source artifact carries a few extra bookkeeping tensors from its QAT
    serialization path, so we transfer weights layer-by-layer and allow the
    clean deployment graph to ignore source-only extras.
    """
    transferred_tensors = 0
    skipped_layers: list[str] = []

    def _transfer_layer(source_layer: keras.layers.Layer, target_layer: keras.layers.Layer) -> None:
        """Transfer one layer, recursing into nested models when needed."""
        nonlocal transferred_tensors

        if isinstance(source_layer, keras.Model) and isinstance(target_layer, keras.Model):
            source_nested_layers_by_name: dict[str, keras.layers.Layer] = {
                layer.name: layer for layer in source_layer.layers
            }
            for nested_target_layer in target_layer.layers:
                nested_source_layer = source_nested_layers_by_name.get(nested_target_layer.name)
                if nested_source_layer is None:
                    continue
                _transfer_layer(nested_source_layer, nested_target_layer)
            return

        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()
        if not source_weights and not target_weights:
            return

        source_weight_map: dict[str, np.ndarray] = {
            _weight_key(variable.name): array
            for variable, array in zip(source_layer.weights, source_weights)
        }

        remapped_target_weights: list[np.ndarray] = []
        missing_weight_names: list[str] = []
        for target_variable in target_layer.weights:
            target_key = _weight_key(target_variable.name)
            source_value = source_weight_map.get(target_key)
            if source_value is None:
                missing_weight_names.append(target_key)
                continue
            remapped_target_weights.append(source_value)

        if missing_weight_names:
            raise RuntimeError(
                f"Layer '{target_layer.name}' is missing source tensor(s) {missing_weight_names}."
            )

        if len(source_weights) < len(target_weights):
            raise RuntimeError(
                "Source layer has fewer weights than target layer: "
                f"{target_layer.name} ({len(source_weights)} < {len(target_weights)})"
            )

        if any(
            src_array.shape != tgt_array.shape
            for src_array, tgt_array in zip(remapped_target_weights, target_weights)
        ):
            raise RuntimeError(
                f"Weight shape mismatch for layer '{target_layer.name}'."
            )

        target_layer.set_weights(remapped_target_weights)
        transferred_tensors += len(remapped_target_weights)

        source_only_tensors = len(source_weights) - len(remapped_target_weights)
        if source_only_tensors > 0:
            print(
                f"[REPACK] Layer '{target_layer.name}' has {source_only_tensors} "
                "extra source tensor(s); copied the matching named weights.",
                flush=True,
            )

    source_layers_by_name: dict[str, keras.layers.Layer] = {
        layer.name: layer for layer in source_model.layers
    }

    for target_layer in target_model.layers:
        source_layer = source_layers_by_name.get(target_layer.name)
        if source_layer is None:
            skipped_layers.append(target_layer.name)
            continue

        _transfer_layer(source_layer, target_layer)

    print(
        f"[REPACK] Transferred {transferred_tensors} weight tensors into clean model.",
        flush=True,
    )
    if skipped_layers:
        print(
            f"[REPACK] Skipped {len(skipped_layers)} clean-layer placeholders without source matches.",
            flush=True,
        )


def _strip_key_recursive(value: Any, key: str) -> int:
    """Remove a key from nested JSON-like structures and count the removals."""
    removed = 0
    if isinstance(value, dict):
        if key in value:
            del value[key]
            removed += 1
        for nested in value.values():
            removed += _strip_key_recursive(nested, key)
    elif isinstance(value, list):
        for nested in value:
            removed += _strip_key_recursive(nested, key)
    return removed


def _strip_quantization_metadata(keras_path: Path) -> None:
    """Rewrite a `.keras` archive with Keras 3 quantization metadata removed."""
    print(f"[REPACK] Stripping quantization metadata from {keras_path}...", flush=True)
    temp_path = keras_path.with_suffix(".tmp.keras")

    with zipfile.ZipFile(keras_path, mode="r") as zin:
        archive_members = {name: zin.read(name) for name in zin.namelist()}

    config_bytes = archive_members.get("config.json")
    if config_bytes is None:
        raise RuntimeError(f"{keras_path} does not contain config.json")

    config = json.loads(config_bytes.decode("utf-8"))
    removed = _strip_key_recursive(config, "quantization_config")
    if removed == 0:
        print("[REPACK] No quantization metadata was found.", flush=True)
    else:
        print(
            f"[REPACK] Removed {removed} quantization metadata field(s).",
            flush=True,
        )
    archive_members["config.json"] = json.dumps(config, indent=2).encode("utf-8")

    with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zout:
        for member_name, payload in archive_members.items():
            zout.writestr(member_name, payload)

    shutil.move(str(temp_path), str(keras_path))
    print(f"[REPACK] Rewrote sanitized archive at {keras_path}.", flush=True)


def _strip_h5_quantization_metadata(h5_path: Path) -> None:
    """Remove Keras 3 quantization metadata from a legacy HDF5 model file."""
    print(f"[REPACK] Stripping H5 quantization metadata from {h5_path}...", flush=True)
    with h5py.File(h5_path, mode="r+") as handle:
        model_config = handle.attrs.get("model_config")
        if model_config is None:
            raise RuntimeError(f"{h5_path} does not contain a model_config attribute")

        if isinstance(model_config, bytes):
            config_text = model_config.decode("utf-8")
        else:
            config_text = str(model_config)

        config = json.loads(config_text)
        removed = _strip_key_recursive(config, "quantization_config")
        if removed == 0:
            print("[REPACK] No H5 quantization metadata was found.", flush=True)
        else:
            print(
                f"[REPACK] Removed {removed} quantization metadata field(s) from H5.",
                flush=True,
            )
        handle.attrs.modify("model_config", json.dumps(config))


def main() -> None:
    """Load, rebuild, and save the clean scalar model artifact."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_model = _load_model(
        args.model,
        legacy_mobilenetv2_preprocess=args.legacy_mobilenetv2_preprocess,
    )
    knot_count = _infer_piecewise_knot_count(source_model)
    print(f"[REPACK] Source piecewise knot count: {knot_count}", flush=True)
    clean_model = _build_clean_model(
        args.image_height,
        args.image_width,
        piecewise_knot_count=knot_count,
    )
    print("[REPACK] Built clean serializable model graph.", flush=True)
    _transfer_weights(source_model, clean_model)

    output_path = args.output_dir / "model.keras"
    clean_model.save(output_path)
    print(f"[REPACK] Saved clean model to {output_path}", flush=True)
    _strip_quantization_metadata(output_path)

    h5_path = args.output_dir / "model.h5"
    clean_model.save(h5_path)
    print(f"[REPACK] Saved legacy H5 model to {h5_path}", flush=True)
    _strip_h5_quantization_metadata(h5_path)


if __name__ == "__main__":
    main()
