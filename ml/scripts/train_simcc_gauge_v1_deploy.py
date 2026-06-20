#!/usr/bin/env python3
"""Train the spatial SimCC deployable gauge geometry model with QAT from scratch.

Produces a TFLite-safe model and runs a Keras-vs-TFLite parity check.
"""

from __future__ import annotations

import argparse
import json
import math
import os as _os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_GPU_MEMORY_LIMIT_MB = int(_os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "3800"))
_SKIP_EXPLICIT_GPU_CONFIG = _os.environ.get("TF_SKIP_EXPLICIT_GPU_CONFIG", "0") == "1"
import tensorflow as tf

if not _SKIP_EXPLICIT_GPU_CONFIG:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=_GPU_MEMORY_LIMIT_MB)],
        )
del _os, _GPU_MEMORY_LIMIT_MB, _SKIP_EXPLICIT_GPU_CONFIG
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    load_clean_geometry_examples,
    load_heatmap_sample,
    sample_jitter_params,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.models_deploy import build_spatial_simcc_gauge_model
from embedded_gauge_reading_tinyml.geometry_heatmap_v3_quant_native_utils import (
    angle_degrees_from_center_to_tip_tf,
    circular_angle_loss_tf,
    normalized_temperature_huber_loss_tf,
    temperature_from_coords_tf,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MANIFEST = Path("ml/data/geometry_heatmap_v13_trusted_train_manifest.csv")
DEFAULT_CALIBRATION = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_OUTPUT = Path("ml/artifacts/training/simcc_gauge_v2_spatial_qat")
NUM_BINS = 112
SPATIAL_CHANNELS = 96
SIGMA_BINS = 3.0  # Gaussian sigma for SimCC soft targets
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 5e-4
TEMPERATURE_MIN_C = -30.0
TEMPERATURE_MAX_C = 50.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_simcc_target(
    coord_224: np.ndarray,
    *,
    num_bins: int = NUM_BINS,
    sigma_bins: float = SIGMA_BINS,
) -> np.ndarray:
    """Build 1D Gaussian soft target for a single coordinate axis.

    coord_224: (batch,) coordinate in 224-pixel space.
    Returns: (batch, num_bins) softmax-like target.
    """
    scale = (num_bins - 1) / 223.0
    bin_idx = coord_224 * scale  # (batch,)
    bins = np.arange(num_bins, dtype=np.float32)  # (num_bins,)
    sq_dist = (bins[np.newaxis, :] - bin_idx[:, np.newaxis]) ** 2
    neg_two_sigma2 = -2.0 * sigma_bins * sigma_bins
    targets = np.exp(sq_dist / neg_two_sigma2)
    targets_sum = targets.sum(axis=1, keepdims=True)
    targets = targets / np.maximum(targets_sum, 1e-8)
    return targets.astype(np.float32)


def _soft_argmax_1d(logits: tf.Tensor) -> tf.Tensor:
    """Soft-argmax over the last axis, returning a normalised [0,1] coordinate."""
    num_bins = tf.cast(tf.shape(logits)[-1], tf.float32)
    bins = tf.linspace(0.0, 1.0, tf.cast(num_bins, tf.int32))
    return tf.reduce_sum(logits * bins[tf.newaxis, :], axis=-1)


def _load_calibration(calib_path: Path) -> tuple[float, float, float]:
    """Load slope, intercept, cold_angle from calibration JSON."""
    with open(calib_path) as f:
        data = json.load(f)
    candidate = data["candidates"][data["selected_candidate_name"]]
    params = candidate["params"]
    return float(params["slope"]), float(params["intercept"]), float(params["cold_angle_degrees"])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SimCCDataGenerator(keras.utils.Sequence):
    """Deterministic dataset that yields (image, SimCC targets, metadata)."""

    def __init__(
        self,
        examples,
        *,
        base_path: Path,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
        sigma_pixels: float = 2.0,
        jitter: bool = False,
    ):
        super().__init__()
        self.examples = list(examples)
        self.base_path = base_path
        self.batch_size = batch_size
        self.seed = seed
        self.sigma_pixels = sigma_pixels
        self.jitter = jitter
        self.indices = np.arange(len(self.examples))
        self.epoch = 0

    def __len__(self):
        return int(math.ceil(len(self.examples) / float(self.batch_size)))

    def on_epoch_end(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(self.indices)
        self.epoch += 1

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.examples))
        batch_indices = self.indices[start:end]

        imgs, cx, cy, tx, ty, angles, temps = [], [], [], [], [], [], []
        for order_idx, example_idx in enumerate(batch_indices):
            ex = self.examples[int(example_idx)]
            jit = None
            if self.jitter:
                rng = np.random.default_rng(self.seed + self.epoch * 100_000 + int(example_idx) * 997 + order_idx)
                jit = sample_jitter_params(rng, shift_min_px=4, shift_max_px=8,
                                             scale_min=0.96, scale_max=1.05,
                                             aspect_min=0.98, aspect_max=1.02)
            sample = load_heatmap_sample(ex, self.base_path, input_size=224, heatmap_size=NUM_BINS,
                                         sigma_pixels=self.sigma_pixels, jitter=jit)
            imgs.append(sample.crop_image.astype(np.float32))
            cx.append(float(sample.metadata["center_x_224"]))
            cy.append(float(sample.metadata["center_y_224"]))
            tx.append(float(sample.metadata["tip_x_224"]))
            ty.append(float(sample.metadata["tip_y_224"]))
            angles.append(float(sample.metadata["angle_degrees"]))
            temps.append(float(sample.metadata["temperature_c"]))

        cx_arr = np.array(cx, dtype=np.float32)
        cy_arr = np.array(cy, dtype=np.float32)
        tx_arr = np.array(tx, dtype=np.float32)
        ty_arr = np.array(ty, dtype=np.float32)

        x = np.stack(imgs, axis=0)
        y = {
            "center_x_simcc": _make_simcc_target(cx_arr),
            "center_y_simcc": _make_simcc_target(cy_arr),
            "tip_x_simcc": _make_simcc_target(tx_arr),
            "tip_y_simcc": _make_simcc_target(ty_arr),
            "confidence": np.ones((len(imgs), 1), dtype=np.float32),
            "true_center_x_224": cx_arr[:, np.newaxis],
            "true_center_y_224": cy_arr[:, np.newaxis],
            "true_tip_x_224": tx_arr[:, np.newaxis],
            "true_tip_y_224": ty_arr[:, np.newaxis],
            "true_center_x_norm": (cx_arr / 223.0)[:, np.newaxis].astype(np.float32),
            "true_center_y_norm": (cy_arr / 223.0)[:, np.newaxis].astype(np.float32),
            "true_tip_x_norm": (tx_arr / 223.0)[:, np.newaxis].astype(np.float32),
            "true_tip_y_norm": (ty_arr / 223.0)[:, np.newaxis].astype(np.float32),
            "true_angle_degrees": np.array(angles, dtype=np.float32)[:, np.newaxis],
            "temperature_c": np.array(temps, dtype=np.float32)[:, np.newaxis],
        }
        return x, y


# ---------------------------------------------------------------------------
# Model wrapper with losses
# ---------------------------------------------------------------------------
class SimCCTrainer(keras.Model):
    """Training wrapper with SimCC cross-entropy + geometry supervision."""

    def __init__(
        self,
        base_model: keras.Model,
        *,
        angle_weight: float = 0.5,
        temperature_weight: float = 0.3,
        confidence_weight: float = 0.1,
        simcc_ce_weight: float = 1.0,
        slope: float = 0.3118859767261175,
        intercept: float = -33.14101213857672,
        cold_angle: float = 135.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.angle_weight = angle_weight
        self.temperature_weight = temperature_weight
        self.confidence_weight = confidence_weight
        self.simcc_ce_weight = simcc_ce_weight
        self._slope = slope
        self._intercept = intercept
        self._cold_angle = cold_angle

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            losses = self._compute_losses(x, y, pred)
        grads = tape.gradient(losses["total_loss"], self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        return {k: tf.cast(v, tf.float32) for k, v in losses.items()}

    def test_step(self, data):
        x, y = data
        pred = self(x, training=False)
        losses = self._compute_losses(x, y, pred)
        return {k: tf.cast(v, tf.float32) for k, v in losses.items()}

    def _compute_losses(self, x, y, pred):
        # SimCC cross-entropy
        ce_cx = tf.reduce_mean(keras.losses.categorical_crossentropy(y["center_x_simcc"], pred[0]))
        ce_cy = tf.reduce_mean(keras.losses.categorical_crossentropy(y["center_y_simcc"], pred[1]))
        ce_tx = tf.reduce_mean(keras.losses.categorical_crossentropy(y["tip_x_simcc"], pred[2]))
        ce_ty = tf.reduce_mean(keras.losses.categorical_crossentropy(y["tip_y_simcc"], pred[3]))
        simcc_loss = ce_cx + ce_cy + ce_tx + ce_ty

        # Confidence BCE
        conf_true = tf.cast(y["confidence"], tf.float32)
        conf_pred = tf.cast(pred[4], tf.float32)
        confidence_loss = tf.reduce_mean(keras.losses.binary_crossentropy(conf_true, conf_pred))

        # Decode coordinates via soft-argmax
        pred_cx_norm = _soft_argmax_1d(pred[0])
        pred_cy_norm = _soft_argmax_1d(pred[1])
        pred_tx_norm = _soft_argmax_1d(pred[2])
        pred_ty_norm = _soft_argmax_1d(pred[3])

        # Angle loss
        pred_cx_224 = pred_cx_norm * 223.0
        pred_cy_224 = pred_cy_norm * 223.0
        pred_tx_224 = pred_tx_norm * 223.0
        pred_ty_224 = pred_ty_norm * 223.0
        pred_angle = angle_degrees_from_center_to_tip_tf(
            pred_cx_224, pred_cy_224, pred_tx_224, pred_ty_224,
        )
        true_angle = tf.squeeze(tf.cast(y["true_angle_degrees"], tf.float32), axis=-1)
        angle_mask = tf.math.is_finite(true_angle)
        angle_loss = tf.cond(
            tf.reduce_any(angle_mask),
            lambda: circular_angle_loss_tf(
                tf.boolean_mask(pred_angle, angle_mask),
                tf.boolean_mask(true_angle, angle_mask),
            ),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

        # Temperature Huber loss
        pred_temp = temperature_from_coords_tf(
            pred_cx_224, pred_cy_224, pred_tx_224, pred_ty_224,
            slope=self._slope, intercept=self._intercept,
            cold_angle_degrees=self._cold_angle,
        )
        true_temp = tf.squeeze(tf.cast(y["temperature_c"], tf.float32), axis=-1)
        temp_mask = tf.math.is_finite(true_temp)
        temperature_loss = tf.cond(
            tf.reduce_any(temp_mask),
            lambda: normalized_temperature_huber_loss_tf(
                tf.boolean_mask(pred_temp, temp_mask),
                tf.boolean_mask(true_temp, temp_mask),
                minimum_celsius=TEMPERATURE_MIN_C,
                maximum_celsius=TEMPERATURE_MAX_C,
            ),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

        total_loss = (
            self.simcc_ce_weight * simcc_loss
            + self.confidence_weight * confidence_loss
            + self.angle_weight * angle_loss
            + self.temperature_weight * temperature_loss
        )

        return {
            "total_loss": total_loss,
            "simcc_loss": simcc_loss,
            "confidence_loss": confidence_loss,
            "angle_loss": angle_loss,
            "temperature_loss": temperature_loss,
        }


# ---------------------------------------------------------------------------
# Parity checker
# ---------------------------------------------------------------------------
def _predict_keras(model: keras.Model, inputs: np.ndarray) -> dict[str, np.ndarray]:
    out = model.predict(inputs, batch_size=4, verbose=0)
    return {"cx": out[0], "cy": out[1], "tx": out[2], "ty": out[3], "conf": out[4]}


def _dequantize_tflite_tensor(tensor: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    """Convert a TFLite tensor back to float32 when the tensor is quantized."""
    tensor = np.asarray(tensor)
    dtype = np.dtype(detail["dtype"])
    if dtype.kind == "f":
        return tensor.astype(np.float32, copy=False)

    scale, zero_point = detail.get("quantization", (0.0, 0))
    scale = float(scale)
    zero_point = float(zero_point)
    if scale == 0.0:
        return tensor.astype(np.float32, copy=False)
    return (tensor.astype(np.float32) - zero_point) * scale


def _detail_map(output_details: Any) -> dict[str, dict[str, Any]]:
    """Normalize TFLite output metadata into a name -> detail mapping."""
    if isinstance(output_details, dict):
        return {str(name): dict(detail) for name, detail in output_details.items()}
    return {str(detail["name"]): dict(detail) for detail in output_details}


def _match_output_name(names: list[str], token: str) -> str | None:
    """Find the most likely tensor name containing a semantic token."""
    token_lower = token.lower()
    exact = [name for name in names if name.lower() == token_lower]
    if exact:
        return exact[0]
    partial = [name for name in names if token_lower in name.lower()]
    if not partial:
        return None
    partial.sort(key=lambda name: (len(name), name))
    return partial[0]


def _predict_tflite(interpreter, inputs: np.ndarray) -> dict[str, np.ndarray]:
    """Run TFLite inference and return outputs keyed by semantic role.

    TFLite may rename or reorder outputs, so this helper first tries to use
    a signature runner and semantic output names. If that is unavailable, it
    falls back to the stable shape-based ordering used in the earlier fix.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    results: dict[str, list[np.ndarray]] = {"cx": [], "cy": [], "tx": [], "ty": [], "conf": []}

    signature_list: dict[str, Any] = {}
    if hasattr(interpreter, "get_signature_list"):
        try:
            signature_list = interpreter.get_signature_list() or {}
        except ValueError:
            signature_list = {}

    if signature_list:
        signature_key = next(iter(signature_list))
        runner = interpreter.get_signature_runner(signature_key)
        runner_input_details = runner.get_input_details()
        if isinstance(runner_input_details, dict):
            input_name = next(iter(runner_input_details))
        else:
            input_name = str(runner_input_details[0]["name"])

        runner_output_details = _detail_map(runner.get_output_details())
        runner_output_names = list(runner_output_details.keys())
        semantic_to_name = {
            "cx": _match_output_name(runner_output_names, "center_x_simcc"),
            "cy": _match_output_name(runner_output_names, "center_y_simcc"),
            "tx": _match_output_name(runner_output_names, "tip_x_simcc"),
            "ty": _match_output_name(runner_output_names, "tip_y_simcc"),
            "conf": _match_output_name(runner_output_names, "confidence"),
        }

        if all(name is not None for name in semantic_to_name.values()):
            for i in range(len(inputs)):
                raw_outputs = runner(**{input_name: inputs[i : i + 1]})
                for semantic, output_name in semantic_to_name.items():
                    assert output_name is not None
                    detail = runner_output_details[output_name]
                    results[semantic].append(_dequantize_tflite_tensor(raw_outputs[output_name], detail))
            return {key: np.concatenate(value, axis=0) for key, value in results.items()}

        # Signature outputs exist, but their names are not descriptive enough.
        # Fall back to the stable insertion order from the signature runner.
        ordered_output_details = list(runner_output_details.values())
        for i in range(len(inputs)):
            raw_outputs = runner(**{input_name: inputs[i : i + 1]})
            ordered_outputs = list(raw_outputs.values())
            for semantic, tensor, detail in zip(results.keys(), ordered_outputs, ordered_output_details):
                results[semantic].append(_dequantize_tflite_tensor(tensor, detail))
        return {key: np.concatenate(value, axis=0) for key, value in results.items()}

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output_name_map = _detail_map(output_details)
    output_names = list(output_name_map.keys())

    semantic_to_name = {
        "cx": _match_output_name(output_names, "center_x_simcc"),
        "cy": _match_output_name(output_names, "center_y_simcc"),
        "tx": _match_output_name(output_names, "tip_x_simcc"),
        "ty": _match_output_name(output_names, "tip_y_simcc"),
        "conf": _match_output_name(output_names, "confidence"),
    }
    if all(name is not None for name in semantic_to_name.values()):
        for i in range(len(inputs)):
            interpreter.set_tensor(input_details[0]["index"], inputs[i : i + 1])
            interpreter.invoke()
            for semantic, output_name in semantic_to_name.items():
                assert output_name is not None
                detail = output_name_map[output_name]
                results[semantic].append(_dequantize_tflite_tensor(interpreter.get_tensor(detail["index"]), detail))
        return {key: np.concatenate(value, axis=0) for key, value in results.items()}

    # Final fallback: sort by output shape and preserve the model output order.
    simcc_indices = [o["index"] for o in output_details if o["shape"][-1] == 112]
    conf_indices = [o["index"] for o in output_details if o["shape"][-1] == 1]
    key_map: dict[str, int] = {}
    if len(simcc_indices) >= 4:
        key_map = {
            "cx": simcc_indices[0],
            "cy": simcc_indices[1],
            "tx": simcc_indices[2],
            "ty": simcc_indices[3],
        }
    if conf_indices:
        key_map["conf"] = conf_indices[0]

    results = {k: [] for k in key_map}
    for i in range(len(inputs)):
        interpreter.set_tensor(input_details[0]["index"], inputs[i : i + 1])
        interpreter.invoke()
        for semantic, output_index in key_map.items():
            detail = next(detail for detail in output_details if detail["index"] == output_index)
            results[semantic].append(_dequantize_tflite_tensor(interpreter.get_tensor(output_index), detail))
    return {key: np.concatenate(value, axis=0) for key, value in results.items()}


def _decode_coords(pred: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Soft-argmax decode SimCC logits to normalized coords."""
    num_bins = pred["cx"].shape[-1]
    bins = np.linspace(0.0, 1.0, num_bins, dtype=np.float32)
    cx = (pred["cx"] * bins).sum(axis=-1)
    cy = (pred["cy"] * bins).sum(axis=-1)
    tx = (pred["tx"] * bins).sum(axis=-1)
    ty = (pred["ty"] * bins).sum(axis=-1)
    return cx, cy, tx, ty


def _angle_from_coords(cx, cy, tx, ty):
    dx = tx - cx
    dy = ty - cy
    return np.degrees(np.arctan2(-dy, dx)) % 360.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train spatial SimCC gauge model")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-bins", type=int, default=NUM_BINS)
    parser.add_argument("--backbone-alpha", type=float, default=0.35)
    parser.add_argument("--spatial-channels", type=int, default=SPATIAL_CHANNELS)
    parser.add_argument("--hard-case-manifest", type=Path, default=None)
    parser.add_argument("--hard-case-repeat", type=int, default=6)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    manifest_path = args.manifest_path if args.manifest_path.is_absolute() else repo_root / args.manifest_path
    calib_path = args.calibration_path if args.calibration_path.is_absolute() else repo_root / args.calibration_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    slope, intercept, cold_angle = _load_calibration(calib_path)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("[SIMCC] Loading data...", flush=True)
    examples = load_clean_geometry_examples(manifest_path)
    train_examples = select_examples_from_split(examples, split="train")
    val_examples = select_examples_from_split(examples, split="val")

    # Hard-case upweighting
    if args.hard_case_manifest is not None:
        hc_path = args.hard_case_manifest if args.hard_case_manifest.is_absolute() else repo_root / args.hard_case_manifest
        hc_names = set()
        import csv
        with open(hc_path) as f:
            for row in csv.DictReader(f):
                hc_names.add(Path(row["image_path"]).name)
        hc_examples = [ex for ex in examples if Path(ex.image_path).name in hc_names]
        hard_base = len(hc_examples)
        train_examples = train_examples + hc_examples * max(int(args.hard_case_repeat), 0)
        print(f"  Train: {len(train_examples)} (base={len(examples)} train-only={len(select_examples_from_split(examples, split='train'))}, hard-case base={hard_base} repeat={args.hard_case_repeat})", flush=True)
    else:
        print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}", flush=True)

    train_gen = SimCCDataGenerator(train_examples, base_path=repo_root,
                                   batch_size=args.batch_size, jitter=True)
    val_gen = SimCCDataGenerator(val_examples, base_path=repo_root,
                                 batch_size=args.batch_size, jitter=False)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print("[SIMCC] Building model...", flush=True)
    model = build_spatial_simcc_gauge_model(
        alpha=args.backbone_alpha,
        num_bins=args.num_bins,
        spatial_channels=args.spatial_channels,
        pretrained=True,
    )
    trainer = SimCCTrainer(
        model,
        slope=slope, intercept=intercept, cold_angle=cold_angle,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"[SIMCC] Training {args.epochs} epochs (float32, PTQ for int8)...", flush=True)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.epochs * len(train_gen),
        alpha=0.01,
    )
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0))
    history = trainer.fit(train_gen, validation_data=val_gen, epochs=args.epochs, verbose=1)

    # ------------------------------------------------------------------
    # Export to TFLite
    # ------------------------------------------------------------------
    print("[SIMCC] Exporting TFLite float32...", flush=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    f32_tflite = converter.convert()
    f32_path = output_dir / "model_float32.tflite"
    f32_path.write_bytes(f32_tflite)
    print(f"  Float32: {len(f32_tflite)/1024:.0f} KB", flush=True)

    print("[SIMCC] Exporting TFLite int8 (PTQ)...", flush=True)

    def representative_dataset():
        for ex in examples[:50]:
            sample = load_heatmap_sample(
                ex,
                repo_root,
                input_size=224,
                heatmap_size=NUM_BINS,
                sigma_pixels=2.0,
                jitter=None,
            )
            yield [np.expand_dims(sample.crop_image.astype(np.float32), axis=0)]

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = representative_dataset
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_tflite = int8_converter.convert()
    int8_path = output_dir / "model_int8.tflite"
    int8_path.write_bytes(int8_tflite)
    print(f"  Int8: {len(int8_tflite)/1024:.0f} KB", flush=True)

    # ------------------------------------------------------------------
    # Parity check
    # ------------------------------------------------------------------
    print("[SIMCC] Running Keras-vs-TFLite parity checks...", flush=True)
    val_x = np.zeros((len(val_examples), 224, 224, 3), dtype=np.float32)
    for i, ex in enumerate(val_examples):
        sample = load_heatmap_sample(ex, repo_root, input_size=224, heatmap_size=112, sigma_pixels=2.0, jitter=None)
        val_x[i] = sample.crop_image.astype(np.float32)

    keras_pred = _predict_keras(model, val_x)
    keras_cx, keras_cy, keras_tx, keras_ty = _decode_coords(keras_pred)

    f32_interpreter = tf.lite.Interpreter(model_path=str(f32_path))
    f32_interpreter.allocate_tensors()
    f32_pred = _predict_tflite(f32_interpreter, val_x)
    f32_cx, f32_cy, f32_tx, f32_ty = _decode_coords(f32_pred)

    int8_interpreter = tf.lite.Interpreter(model_path=str(int8_path))
    int8_interpreter.allocate_tensors()
    int8_pred = _predict_tflite(int8_interpreter, val_x)
    int8_cx, int8_cy, int8_tx, int8_ty = _decode_coords(int8_pred)

    def _parity_metrics(ref_cx, ref_cy, ref_tx, ref_ty, test_cx, test_cy, test_tx, test_ty, num_outputs):
        cx_delta = np.abs(ref_cx - test_cx) * 223.0
        cy_delta = np.abs(ref_cy - test_cy) * 223.0
        tx_delta = np.abs(ref_tx - test_tx) * 223.0
        ty_delta = np.abs(ref_ty - test_ty) * 223.0
        return {
            "center_delta_px_224": {
                "mean": float(cx_delta.mean() + cy_delta.mean()) / 2.0,
                "max": float(max(cx_delta.max(), cy_delta.max())),
            },
            "tip_delta_px_224": {
                "mean": float(tx_delta.mean() + ty_delta.mean()) / 2.0,
                "max": float(max(tx_delta.max(), ty_delta.max())),
            },
            "max_any_delta_px": float(max(cx_delta.max(), cy_delta.max(), tx_delta.max(), ty_delta.max())),
            "num_outputs": int(num_outputs),
        }

    parity_float32 = _parity_metrics(
        keras_cx, keras_cy, keras_tx, keras_ty,
        f32_cx, f32_cy, f32_tx, f32_ty,
        len(f32_interpreter.get_output_details()),
    )
    parity_int8 = _parity_metrics(
        keras_cx, keras_cy, keras_tx, keras_ty,
        int8_cx, int8_cy, int8_tx, int8_ty,
        len(int8_interpreter.get_output_details()),
    )
    print(
        "  Float32 parity: center_delta="
        f"{parity_float32['center_delta_px_224']['mean']:.4f} px, "
        f"tip_delta={parity_float32['tip_delta_px_224']['mean']:.4f} px, "
        f"max_any={parity_float32['max_any_delta_px']:.4f} px",
        flush=True,
    )
    print(
        "  Int8 parity: center_delta="
        f"{parity_int8['center_delta_px_224']['mean']:.4f} px, "
        f"tip_delta={parity_int8['tip_delta_px_224']['mean']:.4f} px, "
        f"max_any={parity_int8['max_any_delta_px']:.4f} px",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------------------
    print("[SIMCC] Computing validation metrics...", flush=True)
    # Use Keras predictions
    pred_angles = _angle_from_coords(keras_cx, keras_cy, keras_tx, keras_ty)

    # Compute ground-truth angles from validation coordinates (no-jitter crop)
    from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
        create_jittered_crop,
        JitterParams,
    )
    _jit = JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)
    gt_angles = np.array([
        _angle_from_coords(
            create_jittered_crop(ex, jitter=_jit).center_x_224,
            create_jittered_crop(ex, jitter=_jit).center_y_224,
            create_jittered_crop(ex, jitter=_jit).tip_x_224,
            create_jittered_crop(ex, jitter=_jit).tip_y_224,
        )
        for ex in val_examples
    ])
    true_temps = np.array([float(ex.temperature_c) for ex in val_examples])

    # Unwrap angles: sort by GT temperature, unwrap sequentially
    _order = np.argsort(true_temps)
    _gt_unwrapped = gt_angles[_order].copy()
    _pred_unwrapped = pred_angles[_order].copy()
    for i in range(1, len(_gt_unwrapped)):
        while _gt_unwrapped[i] > _gt_unwrapped[i - 1] + 180:
            _gt_unwrapped[i] -= 360
        while _gt_unwrapped[i] < _gt_unwrapped[i - 1] - 180:
            _gt_unwrapped[i] += 360
        while _pred_unwrapped[i] > _pred_unwrapped[i - 1] + 180:
            _pred_unwrapped[i] -= 360
        while _pred_unwrapped[i] < _pred_unwrapped[i - 1] - 180:
            _pred_unwrapped[i] += 360

    # Fit angle→temp mapping on unwrapped GT data
    from numpy.polynomial import polynomial as P
    _c = P.polyfit(_gt_unwrapped, true_temps[_order], 1)
    temps = _c[0] + _c[1] * _pred_unwrapped
    temps = np.clip(temps, TEMPERATURE_MIN_C, TEMPERATURE_MAX_C)
    errors = np.abs(temps - true_temps[_order])
    valid = np.isfinite(true_temps[_order]) & np.isfinite(temps)
    errors = errors[valid]

    metrics = {
        "accepted_mae_c": float(errors.mean()) if len(errors) > 0 else float("nan"),
        "raw_mae_c": float(errors.mean()) if len(errors) > 0 else float("nan"),
        "raw_rmse_c": float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else float("nan"),
        "raw_worst_error_c": float(errors.max()) if len(errors) > 0 else float("nan"),
        "raw_gt20_failures": int((errors > 20).sum()),
        "percentage_under_2c": float((errors < 2.0).mean() * 100),
        "percentage_under_5c": float((errors < 5.0).mean() * 100),
        "percentage_under_10c": float((errors < 10.0).mean() * 100),
        "val_count": int(len(val_examples)),
        "params": int(sum(np.prod(v.shape) for v in model.trainable_variables)),
        "tflite_f32_size_kb": int(len(f32_tflite) / 1024),
        "tflite_int8_size_kb": int(len(int8_tflite) / 1024),
    }
    print(f"  Raw MAE: {metrics['raw_mae_c']:.4f} C, RMSE: {metrics['raw_rmse_c']:.4f} C, worst: {metrics['raw_worst_error_c']:.4f} C", flush=True)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    model.save(output_dir / "model.keras")
    with open(output_dir / "parity.json", "w") as f:
        json.dump({"float32": parity_float32, "int8": parity_int8}, f, indent=2)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Summary
    summary = {
        "model_name": "simcc_gauge_v2_spatial_qat",
        "architecture": "MobileNetV2_alpha0.35 + 14x14 spatial SimCC + confidence (float32+PTQ)",
        "inputs": [224, 224, 3],
        "outputs": ["center_x_simcc", "center_y_simcc", "tip_x_simcc", "tip_y_simcc", "confidence"],
        "spatial_channels": args.spatial_channels,
        **metrics,
        "parity": {"float32": parity_float32, "int8": parity_int8},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SIMCC] Done (float32 train + PTQ int8). Artifacts saved to {output_dir}", flush=True)
    print(
        f"  Model: {metrics['params']:,} params, "
        f"{metrics['tflite_f32_size_kb']} KB float32 / {metrics['tflite_int8_size_kb']} KB int8 TFLite",
        flush=True,
    )
    print(f"  Keras MAE: {metrics['raw_mae_c']:.4f} C", flush=True)
    print(f"  Float32 parity max delta: {parity_float32['max_any_delta_px']:.4f} px", flush=True)
    print(f"  Int8 parity max delta: {parity_int8['max_any_delta_px']:.4f} px", flush=True)


if __name__ == "__main__":
    main()
