#!/usr/bin/env python3
"""Export geometry_heatmap_v2 to float32 and int8 TFLite artifacts."""

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

from embedded_gauge_reading_tinyml.geometry_board_replay import build_board_replay_sample
from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import load_clean_geometry_examples
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    load_geometry_heatmap_keras_model,
    summarize_tflite_contract,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import select_examples_from_split


SELECTED_PREPROCESSING_MODE = "python_training_rgb_bilinear"


def _resolve_path(base_path: Path, path: Path) -> Path:
    """Resolve a relative path against the repository root."""

    return path if path.is_absolute() else base_path / path


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a deterministic CSV file."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_representative_manifest(
    *,
    examples: Iterable[Any],
    base_path: Path,
    output_path: Path,
    input_size: int,
    heatmap_size: int,
    sigma_pixels: float,
) -> int:
    """Record the clean train rows used for int8 calibration."""

    rows: list[dict[str, Any]] = []
    for example in examples:
        sample = build_board_replay_sample(
            example,
            base_path,
            mode=SELECTED_PREPROCESSING_MODE,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
        )
        rows.append(
            {
                "image_path": str(example.image_path),
                "split": str(example.split),
                "temperature_c": float(example.temperature_c),
                "source_manifest": str(example.source_manifest),
                "quality_flag": str(example.quality_flag),
                "source_width": int(example.source_width),
                "source_height": int(example.source_height),
                "crop_x1": int(sample.metadata["crop_x1"]),
                "crop_y1": int(sample.metadata["crop_y1"]),
                "crop_x2": int(sample.metadata["crop_x2"]),
                "crop_y2": int(sample.metadata["crop_y2"]),
                "crop_width": int(sample.metadata["crop_width"]),
                "crop_height": int(sample.metadata["crop_height"]),
                "center_x_224": float(sample.metadata["center_x_224"]),
                "center_y_224": float(sample.metadata["center_y_224"]),
                "tip_x_224": float(sample.metadata["tip_x_224"]),
                "tip_y_224": float(sample.metadata["tip_y_224"]),
                "center_x_norm": float(sample.metadata["center_x_norm"]),
                "center_y_norm": float(sample.metadata["center_y_norm"]),
                "tip_x_norm": float(sample.metadata["tip_x_norm"]),
                "tip_y_norm": float(sample.metadata["tip_y_norm"]),
                "preprocessing_mode": SELECTED_PREPROCESSING_MODE,
                "resize_method": str(sample.metadata["resize_method"]),
                "channel_strategy": str(sample.metadata["channel_strategy"]),
                "normalization": str(sample.metadata["normalization"]),
                "source_kind": str(sample.metadata["source_kind"]),
                "input_size": int(input_size),
                "heatmap_size": int(heatmap_size),
                "sigma_pixels": float(sigma_pixels),
            }
        )

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
    """Yield calibration batches in the format TensorFlow Lite expects."""

    for example in examples:
        sample = build_board_replay_sample(
            example,
            base_path,
            mode=SELECTED_PREPROCESSING_MODE,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
        )
        yield [np.expand_dims(np.asarray(sample.crop_image, dtype=np.float32), axis=0)]


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
    """Convert a Keras model to one of the requested TFLite formats."""

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
    """Export the geometry heatmap model and its quantization artifacts."""

    parser = argparse.ArgumentParser(description="Export geometry_heatmap_v2 to TFLite")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2/model.keras"),
        help="Input Keras model path.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Clean geometry manifest.",
    )
    parser.add_argument(
        "--calibration-json-path",
        type=Path,
        default=Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json"),
        help="Calibration artifact kept for provenance in the export config.",
    )
    parser.add_argument(
        "--selected-thresholds-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json"),
        help="Selected board guardrail thresholds kept for provenance.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite"),
        help="Output directory for TFLite artifacts.",
    )
    parser.add_argument("--input-size", type=int, default=224, help="Model input size.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap output size.")
    parser.add_argument("--sigma-pixels", type=float, default=5.0, help="Heatmap sigma used during training.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = _resolve_path(repo_root, args.model_path)
    manifest_path = _resolve_path(repo_root, args.manifest_path)
    calibration_json_path = _resolve_path(repo_root, args.calibration_json_path)
    selected_thresholds_path = _resolve_path(repo_root, args.selected_thresholds_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] Loading Keras model from {model_path}", flush=True)
    model = load_geometry_heatmap_keras_model(model_path)
    examples = load_clean_geometry_examples(manifest_path)
    train_examples = select_examples_from_split(examples, split="train")
    print(f"[EXPORT] Using {len(train_examples)} clean train rows as representative data.", flush=True)

    rep_manifest_path = output_dir / "representative_dataset_manifest.csv"
    representative_count = _write_representative_manifest(
        examples=train_examples,
        base_path=repo_root,
        output_path=rep_manifest_path,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
    )

    # Export the pure float32 TFLite model first so we can compare conversion drift.
    print("[EXPORT] Converting float32 TFLite model...", flush=True)
    float32_tflite = _convert_model(
        model,
        base_path=repo_root,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        int8=False,
    )
    float32_path = output_dir / "model_float32.tflite"
    float32_path.write_bytes(float32_tflite)

    # Dynamic range quantization is optional, but useful if we want a compact
    # non-integer intermediate artifact for debugging.
    dynamic_range_path: Path | None = None
    try:
        print("[EXPORT] Converting dynamic-range TFLite model...", flush=True)
        dynamic_converter = tf.lite.TFLiteConverter.from_keras_model(model)
        dynamic_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        dynamic_range_model = dynamic_converter.convert()
        dynamic_range_path = output_dir / "model_dynamic_range.tflite"
        dynamic_range_path.write_bytes(dynamic_range_model)
    except Exception as exc:  # pragma: no cover - optional artifact
        print(f"[EXPORT] Dynamic-range export skipped: {exc}", flush=True)
        dynamic_range_path = None

    print("[EXPORT] Converting full int8 TFLite model...", flush=True)
    int8_tflite = _convert_model(
        model,
        representative_examples=train_examples,
        base_path=repo_root,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        int8=True,
    )
    int8_path = output_dir / "model_int8.tflite"
    int8_path.write_bytes(int8_tflite)

    float32_contract = summarize_tflite_contract(float32_path)
    int8_contract = summarize_tflite_contract(int8_path)

    config: dict[str, Any] = {
        "source_model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "calibration_json_path": str(calibration_json_path),
        "selected_thresholds_path": str(selected_thresholds_path),
        "output_dir": str(output_dir),
        "selected_preprocessing_mode": SELECTED_PREPROCESSING_MODE,
        "input_size": int(args.input_size),
        "heatmap_size": int(args.heatmap_size),
        "sigma_pixels": float(args.sigma_pixels),
        "keras_input_names": [str(tensor.name) for tensor in model.inputs],
        "keras_output_names": [str(name) for name in model.output_names],
        "keras_output_tensor_names": [str(tensor.name) for tensor in model.outputs],
        "float32_tflite_path": str(float32_path),
        "dynamic_range_tflite_path": str(dynamic_range_path) if dynamic_range_path is not None else None,
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
        "semantic_output_order_indices": [1, 0, 2],
        "semantic_output_names": ["center_heatmap", "tip_heatmap", "confidence"],
        "notes": [
            "Nearest-neighbor and luma preprocessing are invalid for this model unless retrained.",
            "The calibration artifact and guardrails are recorded only for provenance; they are not re-fit here.",
            "TFLite emits the first two heatmap tensors in reverse semantic order, so consumers must reorder them before decode.",
        ],
    }

    config_path = output_dir / "export_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[EXPORT] Wrote {config_path}", flush=True)
    print(f"[EXPORT] Float32 model: {float32_path}", flush=True)
    if dynamic_range_path is not None:
        print(f"[EXPORT] Dynamic-range model: {dynamic_range_path}", flush=True)
    print(f"[EXPORT] Int8 model: {int8_path}", flush=True)


if __name__ == "__main__":
    main()
