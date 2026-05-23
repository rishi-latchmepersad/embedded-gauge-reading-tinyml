#!/usr/bin/env python3
"""Export controlled geometry heatmap v2 TFLite quantization variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_board_replay import (
    load_board_replay_image,
    preprocess_board_replay_image,
)
from embedded_gauge_reading_tinyml.geometry_crop_dataset import JitterParams, create_jittered_crop
from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import load_clean_geometry_examples
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    load_geometry_heatmap_keras_model,
    summarize_tflite_contract,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import select_examples_from_split


SELECTED_PREPROCESSING_MODE = "python_training_rgb_bilinear"
INPUT_SIZE = 224
HEATMAP_SIZE = 56
SIGMA_PIXELS = 5.0
DEFAULT_MANIFEST_PATH = Path("ml/data/geometry_reader_manifest_v2_clean.csv")
DEFAULT_MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v2/model.keras")
DEFAULT_OUTPUT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2")


RepStrategy = Literal["identity", "identity_mild", "identity_mild_medium", "stratified"]


@dataclass(frozen=True)
class RepresentativeSample:
    """One representative input and its provenance metadata."""

    input_tensor: np.ndarray
    row: dict[str, Any]


@dataclass(frozen=True)
class VariantSpec:
    """One export variant to attempt."""

    name: str
    representative_strategy: RepStrategy
    use_representative_dataset: bool
    supported_ops: tuple[tf.lite.OpsSet, ...] | None
    inference_input_type: tf.dtypes.DType | None
    inference_output_type: tf.dtypes.DType | None
    optimization_default: bool


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        name="variant_a_full_int8_identity",
        representative_strategy="identity",
        use_representative_dataset=True,
        supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS_INT8,),
        inference_input_type=tf.int8,
        inference_output_type=tf.int8,
        optimization_default=True,
    ),
    VariantSpec(
        name="variant_a_full_int8_identity_mild",
        representative_strategy="identity_mild",
        use_representative_dataset=True,
        supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS_INT8,),
        inference_input_type=tf.int8,
        inference_output_type=tf.int8,
        optimization_default=True,
    ),
    VariantSpec(
        name="variant_a_full_int8_identity_mild_medium",
        representative_strategy="identity_mild_medium",
        use_representative_dataset=True,
        supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS_INT8,),
        inference_input_type=tf.int8,
        inference_output_type=tf.int8,
        optimization_default=True,
    ),
    VariantSpec(
        name="variant_a_full_int8_stratified",
        representative_strategy="stratified",
        use_representative_dataset=True,
        supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS_INT8,),
        inference_input_type=tf.int8,
        inference_output_type=tf.int8,
        optimization_default=True,
    ),
    VariantSpec(
        name="variant_b_float_io_internal_int8",
        representative_strategy="stratified",
        use_representative_dataset=True,
        supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS_INT8,),
        inference_input_type=None,
        inference_output_type=None,
        optimization_default=True,
    ),
    VariantSpec(
        name="variant_c_int8_input_float_output",
        representative_strategy="stratified",
        use_representative_dataset=True,
        supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS_INT8,),
        inference_input_type=tf.int8,
        inference_output_type=tf.float32,
        optimization_default=True,
    ),
    VariantSpec(
        name="variant_d_dynamic_range",
        representative_strategy="stratified",
        use_representative_dataset=False,
        supported_ops=None,
        inference_input_type=None,
        inference_output_type=None,
        optimization_default=True,
    ),
)


def _resolve_path(base_path: Path, path: Path) -> Path:
    """Resolve a repository-relative path."""

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


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Persist one JSON payload with stable formatting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sample_jitter(example_index: int, strategy: RepStrategy) -> JitterParams:
    """Build a deterministic jitter configuration for one representative example."""

    rng = np.random.default_rng(7_421 + example_index * 101)
    if strategy == "identity":
        return JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)
    if strategy == "identity_mild":
        return JitterParams(
            shift_x=int(rng.integers(-4, 5)),
            shift_y=int(rng.integers(-4, 5)),
            scale=float(rng.uniform(0.97, 1.03)),
            aspect=float(rng.uniform(0.98, 1.02)),
        )
    if strategy == "identity_mild_medium":
        return JitterParams(
            shift_x=int(rng.integers(-8, 9)),
            shift_y=int(rng.integers(-8, 9)),
            scale=float(rng.uniform(0.93, 1.08)),
            aspect=float(rng.uniform(0.95, 1.05)),
        )
    if strategy == "stratified":
        return JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)
    raise ValueError(f"Unsupported representative strategy: {strategy}")


def _build_representative_samples(
    examples: list[Any],
    base_path: Path,
    *,
    strategy: RepStrategy,
    input_size: int,
    heatmap_size: int,
    sigma_pixels: float,
) -> list[RepresentativeSample]:
    """Build representative calibration inputs for one conversion attempt."""

    if strategy == "stratified":
        temp_bins: dict[tuple[str, int], list[Any]] = {}
        for example in examples:
            temp_bin = int(round(float(example.temperature_c) / 5.0) * 5)
            key = (str(example.source_manifest), temp_bin)
            temp_bins.setdefault(key, []).append(example)
        ordered_examples: list[Any] = []
        grouped_items = sorted(temp_bins.items(), key=lambda item: (item[0][0], item[0][1]))
        while any(group for _, group in grouped_items):
            for _, group in grouped_items:
                if group:
                    ordered_examples.append(group.pop(0))
        examples = ordered_examples

    rep_samples: list[RepresentativeSample] = []
    for example_index, example in enumerate(examples):
        jitter_sequence: list[JitterParams]
        identity_jitter = JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)
        mild_jitter = _sample_jitter(example_index, "identity_mild")
        medium_jitter = _sample_jitter(example_index, "identity_mild_medium")
        if strategy == "identity":
            jitter_sequence = [identity_jitter]
        elif strategy == "identity_mild":
            jitter_sequence = [identity_jitter, mild_jitter]
        elif strategy == "identity_mild_medium":
            jitter_sequence = [identity_jitter, mild_jitter, medium_jitter]
        else:
            jitter_sequence = [identity_jitter]

        for jitter in jitter_sequence:
            crop = create_jittered_crop(example, jitter)
            if not crop.accepted:
                crop = create_jittered_crop(example, JitterParams())
            image_path = base_path / Path(crop.source_image_path)
            source_image, source_kind = load_board_replay_image(
                image_path,
                image_width=int(example.source_width),
                image_height=int(example.source_height),
            )
            input_tensor, preprocess_metadata = preprocess_board_replay_image(
                source_image,
                crop_box_xyxy=(float(crop.crop_x1), float(crop.crop_y1), float(crop.crop_x2), float(crop.crop_y2)),
                mode=SELECTED_PREPROCESSING_MODE,
                input_size=input_size,
            )
            row: dict[str, Any] = {
                "image_path": str(crop.source_image_path),
                "split": str(example.split),
                "temperature_c": float(example.temperature_c),
                "source_manifest": str(example.source_manifest),
                "quality_flag": str(example.quality_flag),
                "source_width": int(example.source_width),
                "source_height": int(example.source_height),
                "crop_x1": int(crop.crop_x1),
                "crop_y1": int(crop.crop_y1),
                "crop_x2": int(crop.crop_x2),
                "crop_y2": int(crop.crop_y2),
                "crop_width": int(crop.crop_x2 - crop.crop_x1),
                "crop_height": int(crop.crop_y2 - crop.crop_y1),
                "center_x_norm": float(crop.center_x_normalized),
                "center_y_norm": float(crop.center_y_normalized),
                "tip_x_norm": float(crop.tip_x_normalized),
                "tip_y_norm": float(crop.tip_y_normalized),
                "center_x_224": float(crop.center_x_224),
                "center_y_224": float(crop.center_y_224),
                "tip_x_224": float(crop.tip_x_224),
                "tip_y_224": float(crop.tip_y_224),
                "jitter_shift_x": int(crop.jitter_shift_x),
                "jitter_shift_y": int(crop.jitter_shift_y),
                "jitter_scale": float(crop.jitter_scale),
                "jitter_aspect": float(crop.jitter_aspect),
                "preprocessing_mode": SELECTED_PREPROCESSING_MODE,
                "resize_method": str(preprocess_metadata["resize_method"]),
                "channel_strategy": str(preprocess_metadata["channel_strategy"]),
                "normalization": str(preprocess_metadata["normalization"]),
                "source_kind": source_kind,
                "input_size": int(input_size),
                "heatmap_size": int(heatmap_size),
                "sigma_pixels": float(sigma_pixels),
                "representative_strategy": strategy,
                "temperature_bin_c": int(round(float(example.temperature_c) / 5.0) * 5),
            }
            rep_samples.append(RepresentativeSample(input_tensor=np.asarray(input_tensor, dtype=np.float32), row=row))
    return rep_samples


def _representative_dataset(samples: list[RepresentativeSample]) -> Iterable[list[np.ndarray]]:
    """Yield calibration inputs in the format expected by the TFLite converter."""

    for sample in samples:
        yield [np.expand_dims(np.asarray(sample.input_tensor, dtype=np.float32), axis=0)]


def _semantic_output_order_indices(contract: dict[str, Any]) -> list[int]:
    """Derive the semantic output order from tensor names."""

    outputs = contract["outputs"]
    name_to_index: dict[str, int] = {}
    for index, output in enumerate(outputs):
        name = str(output["name"]).lower()
        if "center" in name:
            name_to_index["center"] = index
        elif "tip" in name:
            name_to_index["tip"] = index
        elif "confidence" in name:
            name_to_index["confidence"] = index
    if {"center", "tip", "confidence"} <= set(name_to_index):
        return [name_to_index["center"], name_to_index["tip"], name_to_index["confidence"]]
    # The geometry heatmap model is known to emit tip, center, confidence in raw
    # TFLite tensor order, so keep the explicit fallback stable across variants.
    return [1, 0, 2]


def _convert_variant(
    model: tf.keras.Model,
    *,
    spec: VariantSpec,
    representative_samples: list[RepresentativeSample],
) -> tuple[bytes, dict[str, Any]]:
    """Convert one Keras model into a requested TFLite variant."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if spec.optimization_default:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if spec.supported_ops is not None:
        converter.target_spec.supported_ops = list(spec.supported_ops)
    if spec.inference_input_type is not None:
        converter.inference_input_type = spec.inference_input_type
    if spec.inference_output_type is not None:
        converter.inference_output_type = spec.inference_output_type
    if spec.use_representative_dataset:
        converter.representative_dataset = lambda: _representative_dataset(representative_samples)
    converted = converter.convert()
    contract = summarize_tflite_contract(Path("/tmp/geometry_heatmap_v2_variant.tflite"))
    return converted, contract


def _convert_with_contract(model: tf.keras.Model, spec: VariantSpec, representative_samples: list[RepresentativeSample]) -> tuple[bytes, dict[str, Any]]:
    """Convert a model and inspect the contract using a temporary file."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if spec.optimization_default:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if spec.supported_ops is not None:
        converter.target_spec.supported_ops = list(spec.supported_ops)
    if spec.inference_input_type is not None:
        converter.inference_input_type = spec.inference_input_type
    if spec.inference_output_type is not None:
        converter.inference_output_type = spec.inference_output_type
    if spec.use_representative_dataset:
        converter.representative_dataset = lambda: _representative_dataset(representative_samples)
    converted = converter.convert()
    return converted, {}


def main() -> None:
    """Export all requested TFLite variant artifacts."""

    parser = argparse.ArgumentParser(description="Export geometry heatmap v2 TFLite variants")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--heatmap-size", type=int, default=HEATMAP_SIZE)
    parser.add_argument("--sigma-pixels", type=float, default=SIGMA_PIXELS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = _resolve_path(repo_root, args.model_path)
    manifest_path = _resolve_path(repo_root, args.manifest_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] Loading Keras model from {model_path}", flush=True)
    model = load_geometry_heatmap_keras_model(model_path)
    examples = load_clean_geometry_examples(manifest_path)
    train_examples = select_examples_from_split(examples, split="train")

    export_index_rows: list[dict[str, Any]] = []

    for spec in VARIANTS:
        variant_dir = output_dir / spec.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        print(f"[EXPORT] Converting {spec.name} using {spec.representative_strategy} calibration samples...", flush=True)
        rep_samples = _build_representative_samples(
            train_examples,
            repo_root,
            strategy=spec.representative_strategy,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            sigma_pixels=args.sigma_pixels,
        )
        rep_manifest_path = variant_dir / "representative_dataset_manifest.csv"
        _write_csv([sample.row for sample in rep_samples], rep_manifest_path)

        config: dict[str, Any] = {
            "source_model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "output_dir": str(variant_dir),
            "representative_strategy": spec.representative_strategy,
            "representative_dataset_count": len(rep_samples),
            "input_size": int(args.input_size),
            "heatmap_size": int(args.heatmap_size),
            "sigma_pixels": float(args.sigma_pixels),
            "selected_preprocessing_mode": SELECTED_PREPROCESSING_MODE,
            "variant_name": spec.name,
            "converter_settings": {
                "optimization_default": spec.optimization_default,
                "supported_ops": [str(op) for op in spec.supported_ops] if spec.supported_ops is not None else None,
                "inference_input_type": str(spec.inference_input_type) if spec.inference_input_type is not None else None,
                "inference_output_type": str(spec.inference_output_type) if spec.inference_output_type is not None else None,
            },
            "status": "pending",
        }

        try:
            converted = tf.lite.TFLiteConverter.from_keras_model(model)
            if spec.optimization_default:
                converted.optimizations = [tf.lite.Optimize.DEFAULT]
            if spec.supported_ops is not None:
                converted.target_spec.supported_ops = list(spec.supported_ops)
            if spec.inference_input_type is not None:
                converted.inference_input_type = spec.inference_input_type
            if spec.inference_output_type is not None:
                converted.inference_output_type = spec.inference_output_type
            if spec.use_representative_dataset:
                converted.representative_dataset = lambda: _representative_dataset(rep_samples)
            tflite_bytes = converted.convert()
            model_path_out = variant_dir / "model.tflite"
            model_path_out.write_bytes(tflite_bytes)
            contract = summarize_tflite_contract(model_path_out)
            semantic_order = _semantic_output_order_indices(contract)
            contract["semantic_output_order_indices"] = semantic_order
            contract["semantic_output_names"] = ["center_heatmap", "tip_heatmap", "confidence"]
            _write_json(contract, variant_dir / "tflite_tensor_contract.json")
            config["status"] = "success"
            config["model_path"] = str(model_path_out)
            config["tensor_contract_path"] = str(variant_dir / "tflite_tensor_contract.json")
            config["semantic_output_order_indices"] = semantic_order
            config["semantic_output_names"] = ["center_heatmap", "tip_heatmap", "confidence"]
            config["input_tensor_name"] = str(contract["input"]["name"])
            config["input_tensor_shape"] = contract["input"]["shape"]
            config["input_tensor_dtype"] = contract["input"]["dtype"]
            config["output_tensor_names"] = [str(output["name"]) for output in contract["outputs"]]
            config["output_tensor_dtypes"] = [str(output["dtype"]) for output in contract["outputs"]]
            print(f"[EXPORT] Wrote {model_path_out}", flush=True)
            export_index_rows.append(
                {
                    "variant_name": spec.name,
                    "variant_dir": str(variant_dir),
                    "status": "success",
                    "model_path": str(model_path_out),
                    "tensor_contract_path": str(variant_dir / "tflite_tensor_contract.json"),
                    "representative_strategy": spec.representative_strategy,
                    "representative_dataset_count": len(rep_samples),
                }
            )
        except Exception as exc:  # pragma: no cover - variant-specific conversion can fail
            config["status"] = "failed"
            config["error"] = str(exc)
            print(f"[EXPORT] Variant {spec.name} failed: {exc}", flush=True)
            export_index_rows.append(
                {
                    "variant_name": spec.name,
                    "variant_dir": str(variant_dir),
                    "status": "failed",
                    "model_path": "",
                    "tensor_contract_path": "",
                    "representative_strategy": spec.representative_strategy,
                    "representative_dataset_count": len(rep_samples),
                }
            )

        _write_json(config, variant_dir / "export_config.json")

    _write_csv(export_index_rows, output_dir / "variant_index.csv")
    print(f"[EXPORT] Wrote variant index to {output_dir / 'variant_index.csv'}", flush=True)


if __name__ == "__main__":
    main()
