#!/usr/bin/env python3
"""Export the geometry_heatmap_v4_112 quant-native model to TFLite."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import load_clean_geometry_examples
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    load_geometry_heatmap_keras_model,
    load_tflite_model,
    summarize_tflite_contract,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import load_heatmap_sample, sample_jitter_params, select_examples_from_split


SELECTED_PREPROCESSING_MODE = "python_training_rgb_bilinear"
DEFAULT_MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras")
DEFAULT_MANIFEST_PATH = Path("ml/data/geometry_reader_manifest_v2_clean.csv")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json")
DEFAULT_DECODER_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")
DEFAULT_OUTPUT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite")


def _resolve_path(base_path: Path, path: Path) -> Path:
    """Resolve a repository-relative path."""

    return path if path.is_absolute() else base_path / path


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a deterministic CSV file."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _temperature_bin(temperature_c: float) -> str:
    """Assign one temperature bin for stratified representative coverage."""

    if temperature_c < -5.0:
        return "cold"
    if temperature_c < 10.0:
        return "cool"
    if temperature_c < 25.0:
        return "warm"
    return "hot"


def _write_representative_manifest(
    *,
    examples: Iterable[Any],
    base_path: Path,
    output_path: Path,
    input_size: int,
    heatmap_size: int,
    sigma_pixels: float,
) -> int:
    """Record the train-only representative dataset used for int8 conversion."""

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(7)
    ordered_examples = sorted(
        list(examples),
        key=lambda example: (float(example.temperature_c), str(example.source_manifest), str(example.image_path)),
    )
    for index, example in enumerate(ordered_examples):
        identity_sample = load_heatmap_sample(
            example,
            base_path,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
        )
        rows.append(
            {
                "strategy": "identity",
                "image_path": str(example.image_path),
                "split": str(example.split),
                "temperature_c": float(example.temperature_c),
                "temperature_bin": _temperature_bin(float(example.temperature_c)),
                "quality_flag": str(example.quality_flag),
                "source_manifest": str(example.source_manifest),
                "source_width": int(example.source_width),
                "source_height": int(example.source_height),
                "crop_x1": int(identity_sample.metadata["crop_x1"]),
                "crop_y1": int(identity_sample.metadata["crop_y1"]),
                "crop_x2": int(identity_sample.metadata["crop_x2"]),
                "crop_y2": int(identity_sample.metadata["crop_y2"]),
                "crop_width": int(identity_sample.metadata["crop_width"]),
                "crop_height": int(identity_sample.metadata["crop_height"]),
                "center_x_224": float(identity_sample.metadata["center_x_224"]),
                "center_y_224": float(identity_sample.metadata["center_y_224"]),
                "tip_x_224": float(identity_sample.metadata["tip_x_224"]),
                "tip_y_224": float(identity_sample.metadata["tip_y_224"]),
                "preprocessing_mode": SELECTED_PREPROCESSING_MODE,
                "resize_method": "bilinear",
                "channel_strategy": "rgb",
                "normalization": "uint8_to_float32_0_1",
                "source_kind": "rgb",
            }
        )

        jitter_seed = int(rng.integers(0, 2**31 - 1))
        jitter_rng = np.random.default_rng(jitter_seed + index)
        jitter = sample_jitter_params(
            jitter_rng,
            shift_min_px=3,
            shift_max_px=7,
            scale_min=0.97,
            scale_max=1.03,
            aspect_min=0.99,
            aspect_max=1.01,
        )
        jitter_sample = load_heatmap_sample(
            example,
            base_path,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
            jitter=jitter,
        )
        rows.append(
            {
                "strategy": "mild_jitter",
                "image_path": str(example.image_path),
                "split": str(example.split),
                "temperature_c": float(example.temperature_c),
                "temperature_bin": _temperature_bin(float(example.temperature_c)),
                "quality_flag": str(example.quality_flag),
                "source_manifest": str(example.source_manifest),
                "source_width": int(example.source_width),
                "source_height": int(example.source_height),
                "crop_x1": int(jitter_sample.metadata["crop_x1"]),
                "crop_y1": int(jitter_sample.metadata["crop_y1"]),
                "crop_x2": int(jitter_sample.metadata["crop_x2"]),
                "crop_y2": int(jitter_sample.metadata["crop_y2"]),
                "crop_width": int(jitter_sample.metadata["crop_width"]),
                "crop_height": int(jitter_sample.metadata["crop_height"]),
                "center_x_224": float(jitter_sample.metadata["center_x_224"]),
                "center_y_224": float(jitter_sample.metadata["center_y_224"]),
                "tip_x_224": float(jitter_sample.metadata["tip_x_224"]),
                "tip_y_224": float(jitter_sample.metadata["tip_y_224"]),
                "preprocessing_mode": SELECTED_PREPROCESSING_MODE,
                "resize_method": "bilinear",
                "channel_strategy": "rgb",
                "normalization": "uint8_to_float32_0_1",
                "source_kind": "rgb",
                "jitter_seed": int(jitter_seed),
            }
        )
        del jitter_rng

    _write_csv(rows, output_path)
    return len(rows)


def _representative_dataset(
    *,
    examples: Iterable[Any],
    base_path: Path,
    input_size: int,
    heatmap_size: int,
    sigma_pixels: float,
) -> Iterable[list[np.ndarray]]:
    """Yield representative batches in the exact format TFLite expects."""

    rng = np.random.default_rng(13)
    for index, example in enumerate(sorted(list(examples), key=lambda item: (float(item.temperature_c), str(item.image_path)))):
        identity_sample = load_heatmap_sample(
            example,
            base_path,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
        )
        yield [np.expand_dims(np.asarray(identity_sample.crop_image, dtype=np.float32), axis=0)]

        jitter_seed = int(rng.integers(0, 2**31 - 1))
        jitter_rng = np.random.default_rng(jitter_seed + index)
        jitter = sample_jitter_params(
            jitter_rng,
            shift_min_px=3,
            shift_max_px=7,
            scale_min=0.97,
            scale_max=1.03,
            aspect_min=0.99,
            aspect_max=1.01,
        )
        jitter_sample = load_heatmap_sample(
            example,
            base_path,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
            jitter=jitter,
        )
        yield [np.expand_dims(np.asarray(jitter_sample.crop_image, dtype=np.float32), axis=0)]


def _convert_model(
    model: tf.keras.Model,
    *,
    representative_examples: Iterable[Any] | None = None,
    base_path: Path,
    input_size: int,
    heatmap_size: int,
    sigma_pixels: float,
    int8: bool,
) -> bytes:
    """Convert one Keras model to a TFLite flatbuffer."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        if representative_examples is None:
            raise ValueError("Int8 conversion requires a representative dataset.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: _representative_dataset(
            examples=representative_examples,
            base_path=base_path,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


def main() -> None:
    """Export the quantization-native v3 model to float32 and int8 TFLite artifacts."""

    parser = argparse.ArgumentParser(description="Export geometry_heatmap_v4_112 TFLite artifacts")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--calibration-json-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--selected-thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--decoder-path", type=Path, default=DEFAULT_DECODER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--heatmap-size", type=int, default=112)
    # Keep the 112x112 export aligned with the sharper v4 training targets.
    parser.add_argument("--sigma-pixels", type=float, default=2.5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = _resolve_path(repo_root, args.model_path)
    manifest_path = _resolve_path(repo_root, args.manifest_path)
    calibration_json_path = _resolve_path(repo_root, args.calibration_json_path)
    selected_thresholds_path = _resolve_path(repo_root, args.selected_thresholds_path)
    decoder_path = _resolve_path(repo_root, args.decoder_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with decoder_path.open("r", encoding="utf-8") as handle:
        decoder_payload = json.load(handle)
    decode_method = str(decoder_payload["decode_method"])
    window_size = int(decoder_payload["window_size"])
    if decode_method != "softargmax" or window_size != 3:
        raise RuntimeError(f"Expected corrected decoder softargmax w3, found {decode_method} w{window_size}.")

    print(f"[V4 EXPORT] Loading v4 Keras model from {model_path}", flush=True)
    model = load_geometry_heatmap_keras_model(model_path)
    examples = load_clean_geometry_examples(manifest_path)
    train_examples = select_examples_from_split(examples, split="train")
    print(f"[V4 EXPORT] Using {len(train_examples)} clean train rows as representative data.", flush=True)

    rep_manifest_path = output_dir / "representative_dataset_manifest.csv"
    representative_count = _write_representative_manifest(
        examples=train_examples,
        base_path=repo_root,
        output_path=rep_manifest_path,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
    )

    print("[V4 EXPORT] Converting float32 TFLite model...", flush=True)
    float32_tflite = _convert_model(
        model,
        base_path=repo_root,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        int8=False,
    )
    float32_path = output_dir / "model_v4_112_float32.tflite"
    float32_path.write_bytes(float32_tflite)

    print("[V4 EXPORT] Converting full int8 TFLite model...", flush=True)
    int8_tflite = _convert_model(
        model,
        representative_examples=train_examples,
        base_path=repo_root,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        int8=True,
    )
    int8_path = output_dir / "model_v4_112_int8.tflite"
    int8_path.write_bytes(int8_tflite)

    float32_contract = summarize_tflite_contract(float32_path)
    int8_contract = summarize_tflite_contract(int8_path)

    num_outputs = len(model.outputs)
    keras_output_names = [str(name) for name in model.output_names]

    float32_bundle = load_tflite_model(float32_path)
    keras_index_by_tensor_name: dict[str, int] = {}
    for tensor_index, keras_name in enumerate(keras_output_names):
        keras_index_by_tensor_name[keras_name] = tensor_index

    raw_tensor_names: list[str] = []
    for detail in float32_bundle.output_details:
        name = str(detail.get("name", ""))
        raw_tensor_names.append(name)

    output_details_index_by_keras_index: dict[int, int] = {}
    for raw_detail_index, raw_name in enumerate(raw_tensor_names):
        suffix = raw_name.rsplit(":", maxsplit=1)[-1] if ":" in raw_name else ""
        if suffix.isdigit():
            keras_index = int(suffix)
        else:
            keras_index = raw_detail_index
        output_details_index_by_keras_index[keras_index] = raw_detail_index

    semantic_output_order_indices = [
        output_details_index_by_keras_index[i] for i in range(num_outputs)
    ]
    if num_outputs == 4:
        # Auto-detect aux head type from output names.
        fourth_name = keras_output_names[3] if len(keras_output_names) > 3 else ""
        if "aux_offset_map" in fourth_name:
            semantic_output_names = ["center_heatmap", "tip_heatmap", "confidence", "aux_offset_map"]
        else:
            semantic_output_names = ["center_heatmap", "tip_heatmap", "confidence", "aux_coords"]
    elif num_outputs == 3:
        semantic_output_names = ["center_heatmap", "tip_heatmap", "confidence"]
    else:
        raise RuntimeError(f"Unexpected number of outputs: {num_outputs}")

    config: dict[str, Any] = {
        "source_model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "calibration_json_path": str(calibration_json_path),
        "selected_thresholds_path": str(selected_thresholds_path),
        "decoder_path": str(decoder_path),
        "output_dir": str(output_dir),
        "selected_preprocessing_mode": SELECTED_PREPROCESSING_MODE,
        "input_size": int(args.input_size),
        "heatmap_size": int(args.heatmap_size),
        "sigma_pixels": float(args.sigma_pixels),
        "keras_output_names": [str(name) for name in model.output_names],
        "float32_tflite_path": str(float32_path),
        "int8_tflite_path": str(int8_path),
        "representative_dataset_manifest_path": str(rep_manifest_path),
        "representative_dataset_count": int(representative_count),
        "board_input_contract": {
            "channels": "RGB",
            "resize_method": "bilinear",
            "input_shape": [1, int(args.input_size), int(args.input_size), 3],
            "normalization": "uint8_to_float32_0_1",
            "crop_coordinate_convention": "same loose crop convention as board replay",
        },
        "float32_contract": float32_contract,
        "int8_contract": int8_contract,
        "decoder": {"decode_method": decode_method, "window_size": window_size},
        "semantic_output_order_indices": semantic_output_order_indices,
        "semantic_output_names": semantic_output_names,
        "notes": [
            "The representative dataset is train-only and includes identity plus mild-jitter board crops.",
            "The dataset is stratified by temperature bins to improve quantization coverage.",
            "The corrected decoder remains softargmax with a 3x3 window.",
        ],
    }

    config_path = output_dir / "export_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "tflite_tensor_contract.json").write_text(
        json.dumps(
            {
                "decoder": {"decode_method": decode_method, "window_size": window_size},
                "semantic_output_order_indices": semantic_output_order_indices,
                "semantic_output_names": semantic_output_names,
                "float32_contract": float32_contract,
                "int8_contract": int8_contract,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"[V4 EXPORT] Wrote {config_path}", flush=True)
    print(f"[V4 EXPORT] Float32 model: {float32_path}", flush=True)
    print(f"[V4 EXPORT] Int8 model: {int8_path}", flush=True)


if __name__ == "__main__":
    main()
