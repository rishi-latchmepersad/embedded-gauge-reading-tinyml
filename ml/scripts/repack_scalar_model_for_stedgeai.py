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
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"[REPACK] Loaded model '{model.name}'.", flush=True)
    model.summary()
    return model


def _build_clean_model(image_height: int, image_width: int) -> keras.Model:
    """Build an equivalent model graph that uses only serializable layers.

    The original artifact already expects normalized `[0, 1]` image inputs.
    We keep that contract, then map the input to the MobileNetV2 domain with
    two standard `Rescaling` layers instead of the legacy Lambda helper.
    """
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    # Mirror the legacy artifact: first restore pixel scale, then apply the
    # MobileNetV2 [-1, 1] normalization in a serializable way.
    x = keras.layers.Rescaling(255.0, name="to_255")(inputs)
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(
        x
    )

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
    x = keras.layers.Dense(1, name="gauge_value")(x)
    output = keras.layers.Dense(1, name="value_calibration")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name="mobilenetv2_gauge_regressor_calibrated_clean",
    )
    return model


def _transfer_weights(source_model: keras.Model, target_model: keras.Model) -> None:
    """Copy trained weights from the loaded model into the clean graph."""
    source_weights = source_model.get_weights()
    target_weights = target_model.get_weights()
    if len(source_weights) != len(target_weights):
        raise RuntimeError(
            "Weight tensor count mismatch between source and clean model: "
            f"{len(source_weights)} vs {len(target_weights)}"
        )
    target_model.set_weights(source_weights)
    print(
        f"[REPACK] Transferred {len(source_weights)} weight tensors into clean model.",
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
    clean_model = _build_clean_model(args.image_height, args.image_width)
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
