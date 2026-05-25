#!/usr/bin/env python3
"""Sweep representative-dataset strategies for v4 112 INT8 export.

Evaluates six strategies to find the one that minimizes quantization drift.
Does not overwrite Phase 10E exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import keras
import tensorflow as tf

from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    JitterParams,
    load_clean_geometry_examples,
    load_heatmap_sample,
    sample_jitter_params,
    select_examples_from_split,
    softargmax_2d,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    load_geometry_heatmap_keras_model,
    load_tflite_model,
    summarize_tflite_contract,
)

SELECTED_PREPROCESSING_MODE = "python_training_rgb_bilinear"
MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras")
MANIFEST_PATH = Path("ml/data/geometry_reader_manifest_v2_clean.csv")
CALIBRATION_JSON_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json")
DECODER_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")
OUTPUT_DIR = Path("ml/artifacts/deployment/geometry_heatmap_v4_112_tflite_rep_sweep")

INPUT_SIZE = 224
HEATMAP_SIZE = 112
SIGMA_PIXELS = 2.5


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
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
    if temperature_c < -5.0:
        return "cold"
    if temperature_c < 10.0:
        return "cool"
    if temperature_c < 25.0:
        return "warm"
    return "hot"


def _make_manifest_row(example: Any, base_path: Path, strategy: str, jitter: JitterParams | None, index: int, seed: int) -> dict[str, Any]:
    sample = load_heatmap_sample(
        example, base_path, input_size=INPUT_SIZE, heatmap_size=HEATMAP_SIZE, sigma_pixels=SIGMA_PIXELS, jitter=jitter,
    )
    return {
        "strategy": strategy,
        "image_path": str(example.image_path),
        "split": str(example.split),
        "temperature_c": float(example.temperature_c),
        "temperature_bin": _temperature_bin(float(example.temperature_c)),
        "quality_flag": str(example.quality_flag),
        "source_manifest": str(example.source_manifest),
        "source_width": int(example.source_width),
        "source_height": int(example.source_height),
        "crop_x1": int(sample.metadata["crop_x1"]),
        "crop_y1": int(sample.metadata["crop_y1"]),
        "crop_x2": int(sample.metadata["crop_x2"]),
        "crop_y2": int(sample.metadata["crop_y2"]),
        "preprocessing_mode": SELECTED_PREPROCESSING_MODE,
        "resize_method": "bilinear",
        "channel_strategy": "rgb",
        "normalization": "uint8_to_float32_0_1",
        "source_kind": "rgb",
        "jitter_seed": int(seed) if jitter is not None else 0,
        "jitter_shift_x": int(sample.metadata.get("jitter_shift_x", 0)),
        "jitter_shift_y": int(sample.metadata.get("jitter_shift_y", 0)),
        "jitter_scale": float(sample.metadata.get("jitter_scale", 1.0)),
        "jitter_aspect": float(sample.metadata.get("jitter_aspect", 1.0)),
    }


# --- Strategy generators ---

def strategy_A_baseline_identity(
    train_examples: list[Any], rng: np.random.Generator, base_path: Path,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Identity crops only, no jitter."""
    rows: list[dict[str, Any]] = []
    rep_examples: list[Any] = []
    for i, example in enumerate(train_examples):
        rows.append(_make_manifest_row(example, base_path, "baseline_identity", None, i, 0))
        rep_examples.append(example)
    return rows, rep_examples


def strategy_B_identity_mild(
    train_examples: list[Any], rng: np.random.Generator, base_path: Path,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Identity + mild jitter (shift ±4, scale 0.97-1.03, aspect 0.98-1.02)."""
    rows: list[dict[str, Any]] = []
    rep_examples: list[Any] = []
    for i, example in enumerate(train_examples):
        rows.append(_make_manifest_row(example, base_path, "identity_mild", None, i, 0))
        rep_examples.append(example)
        seed = int(rng.integers(0, 2**31 - 1))
        jitter = sample_jitter_params(
            np.random.default_rng(seed + i), shift_min_px=3, shift_max_px=4,
            scale_min=0.97, scale_max=1.03, aspect_min=0.98, aspect_max=1.02,
        )
        rows.append(_make_manifest_row(example, base_path, "identity_mild", jitter, i, seed))
        rep_examples.append((example, jitter))
    return rows, rep_examples


def strategy_C_identity_mild_medium(
    train_examples: list[Any], rng: np.random.Generator, base_path: Path,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Identity + mild + medium jitter."""
    rows: list[dict[str, Any]] = []
    rep_examples: list[Any] = []
    for i, example in enumerate(train_examples):
        rows.append(_make_manifest_row(example, base_path, "identity_mild_medium", None, i, 0))
        rep_examples.append(example)
        # mild
        seed1 = int(rng.integers(0, 2**31 - 1))
        j1 = sample_jitter_params(
            np.random.default_rng(seed1 + i), shift_min_px=3, shift_max_px=4,
            scale_min=0.97, scale_max=1.03, aspect_min=0.98, aspect_max=1.02,
        )
        rows.append(_make_manifest_row(example, base_path, "identity_mild_medium", j1, i, seed1))
        rep_examples.append((example, j1))
        # medium
        seed2 = int(rng.integers(0, 2**31 - 1))
        j2 = sample_jitter_params(
            np.random.default_rng(seed2 + i), shift_min_px=5, shift_max_px=8,
            scale_min=0.93, scale_max=1.08, aspect_min=0.95, aspect_max=1.05,
        )
        rows.append(_make_manifest_row(example, base_path, "identity_mild_medium", j2, i, seed2))
        rep_examples.append((example, j2))
    return rows, rep_examples


def strategy_D_stratified(
    train_examples: list[Any], rng: np.random.Generator, base_path: Path,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Balanced by temperature bin and source manifest. Identity + mild jitter per sample."""
    temp_bins: dict[str, list[Any]] = {}
    for ex in train_examples:
        bin_ = _temperature_bin(float(ex.temperature_c))
        temp_bins.setdefault(bin_, []).append(ex)

    source_groups: dict[str, list[Any]] = {}
    for ex in train_examples:
        src = str(ex.source_manifest).split("/")[-1] if ex.source_manifest else "unknown"
        source_groups.setdefault(src, []).append(ex)

    rows: list[dict[str, Any]] = []
    rep_examples: list[Any] = []
    for i, example in enumerate(train_examples):
        rows.append(_make_manifest_row(example, base_path, "stratified", None, i, 0))
        rep_examples.append(example)
        seed = int(rng.integers(0, 2**31 - 1))
        jitter = sample_jitter_params(
            np.random.default_rng(seed + i), shift_min_px=3, shift_max_px=4,
            scale_min=0.97, scale_max=1.03, aspect_min=0.98, aspect_max=1.02,
        )
        rows.append(_make_manifest_row(example, base_path, "stratified", jitter, i, seed))
        rep_examples.append((example, jitter))

    # Add extra jitter variants for underrepresented bins
    for bin_name, bin_examples in temp_bins.items():
        target_count = max(len(bin_examples), int(len(train_examples) / len(temp_bins)) * 2)
        if len(bin_examples) >= target_count:
            continue
        extra_needed = target_count - len(bin_examples)
        for j in range(extra_needed):
            ex = bin_examples[j % len(bin_examples)]
            idx = len(train_examples) + j
            seed = int(rng.integers(0, 2**31 - 1))
            jitter = sample_jitter_params(
                np.random.default_rng(seed + idx), shift_min_px=3, shift_max_px=4,
                scale_min=0.97, scale_max=1.03, aspect_min=0.98, aspect_max=1.02,
            )
            rows.append(_make_manifest_row(ex, base_path, "stratified", jitter, idx, seed))
            rep_examples.append((ex, jitter))

    return rows, rep_examples


def strategy_E_spread_boundary(
    train_examples: list[Any], rng: np.random.Generator, base_path: Path, model: keras.Model,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Identify train samples with tip spread near 45-65px, include identity + mild jitter."""
    print("  Computing Keras tip spread on train split...")
    boundary_candidates: list[Any] = []
    all_identity: list[Any] = []
    for example in train_examples:
        sample = load_heatmap_sample(
            example, base_path, input_size=INPUT_SIZE, heatmap_size=HEATMAP_SIZE, sigma_pixels=SIGMA_PIXELS, jitter=None,
        )
        inp = np.expand_dims(np.asarray(sample.crop_image, dtype=np.float32) * 2.0 - 1.0, axis=0)
        out = model(inp, training=False)
        if isinstance(out, dict):
            thm = out["tip_heatmap"][0, :, :, 0].numpy()
        elif isinstance(out, (list, tuple)):
            thm = out[1][0, :, :, 0].numpy()
        else:
            thm = out[0, :, :, 0].numpy()

        h, w = thm.shape
        coords = np.meshgrid(np.arange(w), np.arange(h))
        cy, cx = softargmax_2d(thm)
        dist_sq = (coords[1] - cy) ** 2 + (coords[0] - cx) ** 2
        thm_norm = thm / (np.sum(thm) + 1e-8)
        spread = float(np.sqrt(np.sum(thm_norm * dist_sq)))

        if 40 <= spread <= 70:
            boundary_candidates.append(example)
        all_identity.append(example)

    rows: list[dict[str, Any]] = []
    rep_examples: list[Any] = []

    for i, example in enumerate(all_identity):
        rows.append(_make_manifest_row(example, base_path, "spread_boundary", None, i, 0))
        rep_examples.append(example)

    b_rng = np.random.default_rng(42)
    for j, example in enumerate(boundary_candidates):
        idx = len(all_identity) + j
        seed = int(b_rng.integers(0, 2**31 - 1))
        jitter = sample_jitter_params(
            np.random.default_rng(seed + idx), shift_min_px=3, shift_max_px=4,
            scale_min=0.97, scale_max=1.03, aspect_min=0.98, aspect_max=1.02,
        )
        rows.append(_make_manifest_row(example, base_path, "spread_boundary", jitter, idx, seed))
        rep_examples.append((example, jitter))

    print(f"  Spread boundary samples (40-70px): {len(boundary_candidates)}")
    return rows, rep_examples


def strategy_F_combined(
    train_examples: list[Any], rng: np.random.Generator, base_path: Path, model: keras.Model,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Combination: stratified + medium jitter + spread-boundary oversampling."""
    rows, rep_examples = strategy_D_stratified(train_examples, rng, base_path)

    boundary_candidates: list[Any] = []
    for example in train_examples:
        sample = load_heatmap_sample(
            example, base_path, input_size=INPUT_SIZE, heatmap_size=HEATMAP_SIZE, sigma_pixels=SIGMA_PIXELS, jitter=None,
        )
        inp = np.expand_dims(np.asarray(sample.crop_image, dtype=np.float32) * 2.0 - 1.0, axis=0)
        out = model(inp, training=False)
        if isinstance(out, dict):
            thm = out["tip_heatmap"][0, :, :, 0].numpy()
        elif isinstance(out, (list, tuple)):
            thm = out[1][0, :, :, 0].numpy()
        else:
            thm = out[0, :, :, 0].numpy()

        h, w = thm.shape
        coords = np.meshgrid(np.arange(w), np.arange(h))
        cy, cx = softargmax_2d(thm)
        dist_sq = (coords[1] - cy) ** 2 + (coords[0] - cx) ** 2
        thm_norm = thm / (np.sum(thm) + 1e-8)
        spread = float(np.sqrt(np.sum(thm_norm * dist_sq)))
        if 40 <= spread <= 70:
            boundary_candidates.append(example)

    b_rng = np.random.default_rng(137)
    offset = len(rows)
    for j, example in enumerate(boundary_candidates):
        idx = offset + j
        seed = int(b_rng.integers(0, 2**31 - 1))
        jitter = sample_jitter_params(
            np.random.default_rng(seed + idx), shift_min_px=5, shift_max_px=8,
            scale_min=0.93, scale_max=1.08, aspect_min=0.95, aspect_max=1.05,
        )
        rows.append(_make_manifest_row(example, base_path, "combined", jitter, idx, seed))
        rep_examples.append((example, jitter))

    print(f"  Combined: stratified base ({len(train_examples)}) + {len(boundary_candidates)} spread-boundary jitter")
    return rows, rep_examples


def _representative_dataset(
    rep_examples: list[Any], base_path: Path,
) -> Iterable[list[np.ndarray]]:
    for item in rep_examples:
        if isinstance(item, tuple):
            example, jitter = item
        else:
            example = item
            jitter = None
        sample = load_heatmap_sample(
            example, base_path, input_size=INPUT_SIZE, heatmap_size=HEATMAP_SIZE, sigma_pixels=SIGMA_PIXELS, jitter=jitter,
        )
        yield [np.expand_dims(np.asarray(sample.crop_image, dtype=np.float32), axis=0)]


def _convert_int8(model: keras.Model, rep_examples: list[Any], base_path: Path) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(rep_examples, base_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep representative dataset strategies for v4 112 INT8 export")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["A", "B", "C", "D", "E", "F"],
                        help="Strategies to run (A B C D E F)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = _resolve_path(repo_root, MODEL_PATH)
    manifest_path = _resolve_path(repo_root, MANIFEST_PATH)
    decoder_path = _resolve_path(repo_root, DECODER_PATH)
    output_dir = _resolve_path(repo_root, OUTPUT_DIR)
    base_path = repo_root

    with decoder_path.open("r") as f:
        decoder_payload = json.load(f)
    decode_method = decoder_payload["decode_method"]
    window_size = decoder_payload["window_size"]
    assert decode_method == "softargmax" and window_size == 3

    print("[SWEEP] Loading model...")
    model = load_geometry_heatmap_keras_model(model_path)
    examples = load_clean_geometry_examples(manifest_path)
    train_examples = select_examples_from_split(examples, split="train")
    print(f"[SWEEP] {len(train_examples)} clean train rows")

    rng = np.random.default_rng(7)
    strategies = {
        "A": ("baseline_identity", strategy_A_baseline_identity),
        "B": ("identity_mild", strategy_B_identity_mild),
        "C": ("identity_mild_medium", strategy_C_identity_mild_medium),
        "D": ("stratified", strategy_D_stratified),
        "E": ("spread_boundary", strategy_E_spread_boundary),
        "F": ("combined", strategy_F_combined),
    }

    # Export FP32 once
    float32_path = output_dir / "model_v4_112_float32.tflite"
    if not float32_path.exists():
        print("[SWEEP] Exporting FP32 TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        float32_path.parent.mkdir(parents=True, exist_ok=True)
        float32_path.write_bytes(converter.convert())
        print(f"  -> {float32_path}")
    else:
        print("[SWEEP] FP32 already exists, skipping")

    # Sweep strategies
    for key in args.strategies:
        if key not in strategies:
            print(f"[SWEEP] Unknown strategy: {key}, skipping")
            continue

        name, generator = strategies[key]
        strategy_dir = output_dir / name
        int8_path = strategy_dir / "model_v4_112_int8.tflite"
        manifest_path_out = strategy_dir / "representative_dataset_manifest.csv"

        if int8_path.exists():
            print(f"[SWEEP] {name} already exported, skipping")
            continue

        print(f"[SWEEP] Generating representative dataset: {name}...")
        if key == "E":
            manifest_rows, rep_examples = generator(train_examples, rng, base_path, model)
        elif key == "F":
            manifest_rows, rep_examples = generator(train_examples, rng, base_path, model)
        else:
            manifest_rows, rep_examples = generator(train_examples, rng, base_path)

        strategy_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(manifest_rows, manifest_path_out)
        print(f"  Manifest: {len(manifest_rows)} rows -> {manifest_path_out}")

        print(f"  Converting INT8...")
        int8_tflite = _convert_int8(model, rep_examples, base_path)
        int8_path.write_bytes(int8_tflite)
        print(f"  INT8 model: {int8_path} ({int8_path.stat().st_size / 1024:.0f} KB)")

        config = {
            "strategy": name,
            "representative_dataset_count": len(manifest_rows),
            "source_model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "thresholds_path": str(THRESHOLDS_PATH),
            "decoder": {"decode_method": decode_method, "window_size": window_size},
            "input_size": INPUT_SIZE,
            "heatmap_size": HEATMAP_SIZE,
            "sigma_pixels": SIGMA_PIXELS,
            "preprocessing_mode": SELECTED_PREPROCESSING_MODE,
        }
        (strategy_dir / "export_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

        contract = summarize_tflite_contract(int8_path)
        contract["strategy"] = name
        contract["decoder"] = {"decode_method": decode_method, "window_size": window_size}
        contract["semantic_output_order_indices"] = [1, 0, 2]
        contract["semantic_output_names"] = ["center_heatmap", "tip_heatmap", "confidence"]
        (strategy_dir / "tflite_tensor_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

        print(f"  Done with {name}")

    print("[SWEEP] Complete")


if __name__ == "__main__":
    main()
