#!/usr/bin/env python3
"""Fine-tune geometry_heatmap_v2 with quantization-noise robustness.

Standard TFMOT QAT is intentionally not forced here because the active Poetry
environment does not ship tensorflow_model_optimization.  Instead, this script
keeps the corrected decoder and export contract intact while adding fake int8
output round-trips plus light Gaussian output noise during training.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_qat_utils import fake_quantize_01_tensor
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (
    HeatmapSample,
    decode_prediction_row,
    load_clean_geometry_examples,
    load_heatmap_sample,
    load_heatmap_samples,
    load_selected_calibration_candidate,
    sample_jitter_params,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import load_geometry_heatmap_keras_model
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import GeometryGuardrailThresholds, apply_geometry_guardrails, decode_heatmap_geometry_prediction
from embedded_gauge_reading_tinyml.heatmap_losses import (
    mean_predicted_heatmap_peak,
    softargmax_coordinate_mae,
    weighted_center_heatmap_loss,
    weighted_tip_heatmap_loss,
)
from embedded_gauge_reading_tinyml.models_geometry import build_mobilenetv2_geometry_heatmap_v1


TRAIN_JITTER_SHIFT_MIN_PX: int = 4
TRAIN_JITTER_SHIFT_MAX_PX: int = 8
TRAIN_JITTER_SCALE_MIN: float = 0.96
TRAIN_JITTER_SCALE_MAX: float = 1.05
TRAIN_JITTER_ASPECT_MIN: float = 0.98
TRAIN_JITTER_ASPECT_MAX: float = 1.02

DEFAULT_OUTPUT_DIR = Path("ml/artifacts/training/geometry_heatmap_v2_qat")
DEFAULT_MODEL_PATH = Path("ml/artifacts/training/geometry_heatmap_v2/model.keras")
DEFAULT_MANIFEST_PATH = Path("ml/data/geometry_reader_manifest_v2_clean.csv")
DEFAULT_CALIBRATION_PATH = Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json")
DEFAULT_THRESHOLDS_PATH = Path("ml/artifacts/training/geometry_heatmap_v2_board_replay/selected_board_guardrail_thresholds.json")
DEFAULT_DECODER_PATH = Path("ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json")

LOSS_WEIGHTS = {
    "center_heatmap": 2.0,
    "tip_heatmap": 1.0,
    "confidence": 0.1,
}


@dataclass(frozen=True)
class SplitBatch:
    """A deterministic validation batch for board-replay-style selection."""

    samples: list[HeatmapSample]
    x: np.ndarray
    y: dict[str, np.ndarray]


def _resolve_path(repo_root: Path, path: Path) -> Path:
    """Resolve a repository-relative path."""

    return path if path.is_absolute() else repo_root / path


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write a JSON file with stable formatting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a deterministic CSV table."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen_fieldnames: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen_fieldnames:
                continue
            seen_fieldnames.add(key)
            fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_history(history: dict[str, list[float]], output_path: Path) -> None:
    """Persist a callback-friendly history dictionary to CSV."""

    metric_names = list(history.keys())
    epochs = len(history.get("loss", []))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("epoch," + ",".join(metric_names) + "\n")
        for index in range(epochs):
            values = [str(history[name][index]) for name in metric_names]
            handle.write(f"{index + 1}," + ",".join(values) + "\n")


def _get_backbone(model: keras.Model) -> keras.Model:
    """Find the nested MobileNetV2 backbone inside the heatmap model."""

    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower():
            return layer
    raise RuntimeError("Could not find the nested MobileNetV2 backbone.")


def _set_backbone_trainability(model: keras.Model, *, trainable_last_block: bool) -> None:
    """Freeze the backbone or unfreeze only its last MobileNetV2 block."""

    backbone = _get_backbone(model)
    if not trainable_last_block:
        backbone.trainable = False
        for layer in backbone.layers:
            layer.trainable = False
        return

    backbone.trainable = True
    for layer in backbone.layers:
        is_last_block = layer.name.startswith("block_16") or layer.name in {"Conv_1", "Conv_1_bn", "out_relu"}
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = is_last_block


def _as_output_dict(outputs: Any) -> dict[str, tf.Tensor]:
    """Normalize model outputs to the semantic heatmap dictionary."""

    if isinstance(outputs, dict):
        return {
            "center_heatmap": tf.cast(outputs["center_heatmap"], tf.float32),
            "tip_heatmap": tf.cast(outputs["tip_heatmap"], tf.float32),
            "confidence": tf.cast(outputs["confidence"], tf.float32),
        }
    center_heatmap, tip_heatmap, confidence = outputs
    return {
        "center_heatmap": tf.cast(center_heatmap, tf.float32),
        "tip_heatmap": tf.cast(tip_heatmap, tf.float32),
        "confidence": tf.cast(confidence, tf.float32),
    }


def _quantize_output(tensor: tf.Tensor, *, noise_stddev: float) -> tf.Tensor:
    """Inject small output noise and then fake-quantize a sigmoid-bounded tensor."""

    values = tf.cast(tensor, tf.float32)
    if noise_stddev > 0.0:
        values = values + tf.random.normal(tf.shape(values), stddev=noise_stddev, dtype=values.dtype)
    values = tf.clip_by_value(values, 0.0, 1.0)
    return fake_quantize_01_tensor(values)


class GeometryHeatmapSequence(keras.utils.Sequence):
    """Jittered training batches generated on the fly from clean train rows."""

    def __init__(
        self,
        examples: list[Any],
        *,
        base_path: Path,
        batch_size: int,
        heatmap_size: int,
        sigma_pixels: float,
        seed: int,
    ) -> None:
        self.examples = list(examples)
        self.base_path = base_path
        self.batch_size = int(batch_size)
        self.heatmap_size = int(heatmap_size)
        self.sigma_pixels = float(sigma_pixels)
        self.seed = int(seed)
        self.indices = np.arange(len(self.examples))
        self.epoch = 0

    def __len__(self) -> int:
        return int(math.ceil(len(self.examples) / float(self.batch_size)))

    def on_epoch_end(self) -> None:
        """Shuffle the sample order between epochs."""

        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(self.indices)
        self.epoch += 1

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Build one jittered mini-batch."""

        batch_start = index * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.examples))
        batch_indices = self.indices[batch_start:batch_end]

        batch_x: list[np.ndarray] = []
        batch_center: list[np.ndarray] = []
        batch_tip: list[np.ndarray] = []

        for order_index, example_index in enumerate(batch_indices):
            example = self.examples[int(example_index)]
            jitter_rng = np.random.default_rng(self.seed + self.epoch * 100_000 + int(example_index) * 997 + order_index)
            jitter = sample_jitter_params(
                jitter_rng,
                shift_min_px=TRAIN_JITTER_SHIFT_MIN_PX,
                shift_max_px=TRAIN_JITTER_SHIFT_MAX_PX,
                scale_min=TRAIN_JITTER_SCALE_MIN,
                scale_max=TRAIN_JITTER_SCALE_MAX,
                aspect_min=TRAIN_JITTER_ASPECT_MIN,
                aspect_max=TRAIN_JITTER_ASPECT_MAX,
            )
            sample = load_heatmap_sample(
                example,
                self.base_path,
                heatmap_size=self.heatmap_size,
                sigma_pixels=self.sigma_pixels,
                jitter=jitter,
            )
            batch_x.append(sample.crop_image.astype(np.float32))
            batch_center.append(sample.center_heatmap.astype(np.float32)[..., np.newaxis])
            batch_tip.append(sample.tip_heatmap.astype(np.float32)[..., np.newaxis])

        x = np.stack(batch_x, axis=0)
        y = {
            "center_heatmap": np.stack(batch_center, axis=0),
            "tip_heatmap": np.stack(batch_tip, axis=0),
            "confidence": np.ones((len(batch_x), 1), dtype=np.float32),
        }
        return x, y


class ReplayBoardMetricCallback(keras.callbacks.Callback):
    """Evaluate one split with the corrected board-replay decoder after each epoch."""

    def __init__(
        self,
        *,
        metric_prefix: str,
        samples: list[HeatmapSample],
        inputs: np.ndarray,
        calibration_candidate: Any,
        thresholds: GeometryGuardrailThresholds,
        decode_method: str,
        window_size: int,
        best_model_path: Path,
    ) -> None:
        super().__init__()
        self.metric_prefix = metric_prefix
        self.samples = list(samples)
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.calibration_candidate = calibration_candidate
        self.thresholds = thresholds
        self.decode_method = decode_method
        self.window_size = int(window_size)
        self.best_model_path = best_model_path
        self.best_score = math.inf
        self.best_summary: dict[str, float] | None = None

    def _evaluate_model(self) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Decode the current model on the stored split."""

        predictions = self.model.predict(self.inputs, verbose=0)
        outputs = _as_output_dict(predictions)

        rows: list[dict[str, Any]] = []
        for index, sample in enumerate(self.samples):
            confidence = float(np.ravel(outputs["confidence"][index])[0])
            decoded = decode_heatmap_geometry_prediction(
                sample,
                outputs["center_heatmap"][index],
                outputs["tip_heatmap"][index],
                confidence,
                self.calibration_candidate,
                decode_method=self.decode_method,
                window_size=self.window_size,
            )
            guarded = apply_geometry_guardrails(decoded, self.thresholds)
            row = decode_prediction_row(
                sample,
                outputs["center_heatmap"][index],
                outputs["tip_heatmap"][index],
                confidence,
                calibration_candidate=self.calibration_candidate,
            )
            row["guardrail_status"] = guarded.status
            row["guarded_temperature_c"] = float(guarded.temperature_c)
            row["rejection_reasons"] = ";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none"
            rows.append(row)

        accepted_errors = np.asarray(
            [
                abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"]))
                for row in rows
                if str(row["guardrail_status"]) != "rejected"
            ],
            dtype=np.float64,
        )
        guarded_statuses = [str(row["guardrail_status"]) for row in rows]
        calibrated_errors = np.asarray(
            [
                abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"]))
                for row in rows
                if str(row["guardrail_status"]) != "rejected"
            ],
            dtype=np.float64,
        )
        summary = {
            "count": float(len(rows)),
            "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
            "acceptance_rate": float(np.mean([status != "rejected" for status in guarded_statuses])),
            "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
            "accepted_gt20_failures": float(
                sum(
                    1
                    for row in rows
                    if str(row["guardrail_status"]) != "rejected"
                    and abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0
                )
            ),
            "percentage_under_2c": float(np.mean(calibrated_errors < 2.0) * 100.0) if calibrated_errors.size else math.nan,
            "percentage_under_5c": float(np.mean(calibrated_errors < 5.0) * 100.0) if calibrated_errors.size else math.nan,
            "percentage_under_10c": float(np.mean(calibrated_errors < 10.0) * 100.0) if calibrated_errors.size else math.nan,
            "center_mae_px_224": float(np.mean([float(row["center_px_mae_224"]) for row in rows])),
            "tip_mae_px_224": float(np.mean([float(row["tip_px_mae_224"]) for row in rows])),
            "angle_mae_degrees": float(np.mean([float(row["angle_mae_degrees"]) for row in rows])),
            "center_heatmap_peak_mean": float(np.mean([float(row["center_heatmap_peak_value"]) for row in rows])),
            "tip_heatmap_peak_mean": float(np.mean([float(row["tip_heatmap_peak_value"]) for row in rows])),
            "confidence_mean": float(np.mean([float(row["confidence"]) for row in rows])),
            "top_rejection_reasons": ";".join(
                f"{reason}:{count}" for reason, count in self._top_rejection_reasons(rows)
            )
            if any(status == "rejected" for status in guarded_statuses)
            else "none",
        }
        return rows, summary

    def _top_rejection_reasons(self, rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
        """Count the most common rejection reasons."""

        reason_counts: dict[str, int] = {}
        for row in rows:
            if str(row["guardrail_status"]) != "rejected":
                continue
            reasons = str(row["rejection_reasons"]).split(";")
            for reason in reasons:
                if not reason or reason == "none":
                    continue
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:5]

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Record the replay-style validation metric and save the best weights."""

        logs = {} if logs is None else logs
        _, summary = self._evaluate_model()
        self.best_summary = summary if summary["accepted_mae_c"] < self.best_score else self.best_summary
        logs[f"val_{self.metric_prefix}_accepted_mae"] = summary["accepted_mae_c"]
        logs[f"val_{self.metric_prefix}_acceptance_rate"] = summary["acceptance_rate"]
        logs[f"val_{self.metric_prefix}_worst_accepted_error"] = summary["worst_accepted_error_c"]
        logs[f"val_{self.metric_prefix}_accepted_gt20_failures"] = summary["accepted_gt20_failures"]
        logs[f"val_{self.metric_prefix}_center_mae_px"] = summary["center_mae_px_224"]
        logs[f"val_{self.metric_prefix}_tip_mae_px"] = summary["tip_mae_px_224"]
        logs[f"val_{self.metric_prefix}_angle_mae_degrees"] = summary["angle_mae_degrees"]
        logs[f"val_{self.metric_prefix}_center_heatmap_peak_mean"] = summary["center_heatmap_peak_mean"]
        logs[f"val_{self.metric_prefix}_tip_heatmap_peak_mean"] = summary["tip_heatmap_peak_mean"]
        logs[f"val_{self.metric_prefix}_confidence_mean"] = summary["confidence_mean"]

        if summary["accepted_mae_c"] < self.best_score:
            self.best_score = summary["accepted_mae_c"]
            self.model.base_model.save(self.best_model_path)


class GeometryQATTrainingModel(keras.Model):
    """Wrap the base heatmap model with fake quantization noise for training."""

    def __init__(
        self,
        *,
        base_model: keras.Model,
        reference_model: keras.Model,
        output_noise_stddev: float,
        distillation_weight: float,
        heatmap_distillation_weight: float,
        confidence_distillation_weight: float,
    ) -> None:
        super().__init__(name="geometry_heatmap_v2_qat_training_model")
        self.base_model = base_model
        self.reference_model = reference_model
        self.output_noise_stddev = float(output_noise_stddev)
        self.distillation_weight = float(distillation_weight)
        self.heatmap_distillation_weight = float(heatmap_distillation_weight)
        self.confidence_distillation_weight = float(confidence_distillation_weight)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> dict[str, tf.Tensor]:
        """Return the semantic outputs, adding quantization noise only during training."""

        outputs = _as_output_dict(self.base_model(inputs, training=training))
        if training:
            outputs = {
                "center_heatmap": _quantize_output(outputs["center_heatmap"], noise_stddev=self.output_noise_stddev),
                "tip_heatmap": _quantize_output(outputs["tip_heatmap"], noise_stddev=self.output_noise_stddev),
                "confidence": _quantize_output(outputs["confidence"], noise_stddev=self.output_noise_stddev),
            }
        return outputs

    def _distillation_loss(
        self,
        pred: dict[str, tf.Tensor],
        reference: dict[str, tf.Tensor],
    ) -> tf.Tensor:
        """Match the current outputs against the frozen base model outputs."""

        center_loss = tf.reduce_mean(tf.square(pred["center_heatmap"] - reference["center_heatmap"]))
        tip_loss = tf.reduce_mean(tf.square(pred["tip_heatmap"] - reference["tip_heatmap"]))
        confidence_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(reference["confidence"], pred["confidence"])
        )
        return (
            self.heatmap_distillation_weight * (center_loss + tip_loss)
            + self.confidence_distillation_weight * confidence_loss
        )

    def train_step(self, data: Any) -> dict[str, float]:
        """Run one optimized training step with optional distillation."""

        x, y = data
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            main_loss = self.compiled_loss(y, pred, regularization_losses=self.losses)
            reference = _as_output_dict(self.reference_model(x, training=False))
            distillation_loss = self._distillation_loss(pred, reference)
            total_loss = main_loss + self.distillation_weight * distillation_loss
        gradients = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
        return {
            "loss": tf.cast(total_loss, tf.float32),
            "main_loss": tf.cast(main_loss, tf.float32),
            "distillation_loss": tf.cast(distillation_loss, tf.float32),
        }

    def test_step(self, data: Any) -> dict[str, float]:
        """Run one validation step without gradient updates."""

        x, y = data
        pred = self(x, training=False)
        main_loss = self.compiled_loss(y, pred, regularization_losses=self.losses)
        return {"loss": tf.cast(main_loss, tf.float32), "main_loss": tf.cast(main_loss, tf.float32)}


def _compile_model(model: keras.Model, *, learning_rate: float) -> None:
    """Compile the wrapped model with the heatmap and confidence objectives."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "center_heatmap": weighted_center_heatmap_loss,
            "tip_heatmap": weighted_tip_heatmap_loss,
            "confidence": keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "center_heatmap": LOSS_WEIGHTS["center_heatmap"],
            "tip_heatmap": LOSS_WEIGHTS["tip_heatmap"],
            "confidence": LOSS_WEIGHTS["confidence"],
        },
    )


def _evaluate_replay_split(
    model: keras.Model,
    samples: list[HeatmapSample],
    *,
    calibration_candidate: Any,
    thresholds: GeometryGuardrailThresholds,
    decode_method: str,
    window_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Decode one split and summarize the board-replay-style metrics."""

    x = np.stack([sample.crop_image for sample in samples], axis=0).astype(np.float32)
    predictions = model.predict(x, verbose=0)
    outputs = _as_output_dict(predictions)

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        confidence = float(np.ravel(outputs["confidence"][index])[0])
        decoded = decode_heatmap_geometry_prediction(
            sample,
            outputs["center_heatmap"][index],
            outputs["tip_heatmap"][index],
            confidence,
            calibration_candidate,
            decode_method=decode_method,
            window_size=window_size,
        )
        guarded = apply_geometry_guardrails(decoded, thresholds)
        row = decode_prediction_row(
            sample,
            outputs["center_heatmap"][index],
            outputs["tip_heatmap"][index],
            confidence,
            calibration_candidate=calibration_candidate,
        )
        row["guardrail_status"] = guarded.status
        row["guarded_temperature_c"] = float(guarded.temperature_c)
        row["rejection_reasons"] = ";".join(guarded.rejection_reasons) if guarded.rejection_reasons else "none"
        rows.append(row)

    accepted_errors = np.asarray(
        [
            abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"]))
            for row in rows
            if str(row["guardrail_status"]) != "rejected"
        ],
        dtype=np.float64,
    )
    calibrated_errors = np.asarray(
        [
            abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"]))
            for row in rows
            if str(row["guardrail_status"]) != "rejected"
        ],
        dtype=np.float64,
    )
    summary: dict[str, float] = {
        "count": float(len(rows)),
        "accepted_mae_c": float(np.mean(accepted_errors)) if accepted_errors.size else math.nan,
        "acceptance_rate": float(np.mean([str(row["guardrail_status"]) != "rejected" for row in rows])),
        "worst_accepted_error_c": float(np.max(accepted_errors)) if accepted_errors.size else math.nan,
        "accepted_gt20_failures": float(
            sum(
                1
                for row in rows
                if str(row["guardrail_status"]) != "rejected"
                and abs(float(row["guarded_temperature_c"]) - float(row["true_temperature_c"])) > 20.0
            )
        ),
        "percentage_under_2c": float(np.mean(calibrated_errors < 2.0) * 100.0) if calibrated_errors.size else math.nan,
        "percentage_under_5c": float(np.mean(calibrated_errors < 5.0) * 100.0) if calibrated_errors.size else math.nan,
        "percentage_under_10c": float(np.mean(calibrated_errors < 10.0) * 100.0) if calibrated_errors.size else math.nan,
        "center_mae_px_224": float(np.mean([float(row["center_px_mae_224"]) for row in rows])),
        "tip_mae_px_224": float(np.mean([float(row["tip_px_mae_224"]) for row in rows])),
        "angle_mae_degrees": float(np.mean([float(row["angle_mae_degrees"]) for row in rows])),
        "center_heatmap_peak_mean": float(np.mean([float(row["center_heatmap_peak_value"]) for row in rows])),
        "tip_heatmap_peak_mean": float(np.mean([float(row["tip_heatmap_peak_value"]) for row in rows])),
        "confidence_mean": float(np.mean([float(row["confidence"]) for row in rows])),
        "top_rejection_reasons": _top_rejection_reason_string(rows),
        "rejection_count": float(sum(1 for row in rows if str(row["guardrail_status"]) == "rejected")),
        "accepted_count": float(sum(1 for row in rows if str(row["guardrail_status"]) != "rejected")),
    }
    return rows, summary


def _top_rejection_reason_string(rows: list[dict[str, Any]]) -> str:
    """Format the most common rejection reasons for reporting."""

    counts: dict[str, int] = {}
    for row in rows:
        if str(row["guardrail_status"]) != "rejected":
            continue
        reasons = str(row["rejection_reasons"]).split(";")
        for reason in reasons:
            if not reason or reason == "none":
                continue
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "none"
    return ";".join(f"{reason}:{count}" for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5])


def _write_markdown_report(
    *,
    output_path: Path,
    feasibility: dict[str, Any],
    frozen_summary: dict[str, float],
    final_summary: dict[str, float],
    selected_stage: str,
    decode_method: str,
    window_size: int,
    selected_candidate_name: str,
    calibration_json: dict[str, Any],
) -> None:
    """Write a concise training report for the QAT run."""

    lines = [
        "# Geometry Heatmap v2 QAT Fine-Tuning",
        "",
        f"- Training strategy: {feasibility['preferred_training_strategy']}",
        f"- Decoder: {decode_method} w{window_size}",
        f"- Calibration candidate: {selected_candidate_name}",
        f"- Selected stage: {selected_stage}",
        "",
        "## Frozen Stage Val Replay",
        f"- Accepted MAE: {frozen_summary['accepted_mae_c']:.4f} C",
        f"- Acceptance rate: {frozen_summary['acceptance_rate']:.4f}",
        f"- Worst accepted error: {frozen_summary['worst_accepted_error_c']:.4f} C",
        f"- Accepted >20 C failures: {int(frozen_summary['accepted_gt20_failures'])}",
        f"- Under 2/5/10 C: {frozen_summary['percentage_under_2c']:.2f}% / {frozen_summary['percentage_under_5c']:.2f}% / {frozen_summary['percentage_under_10c']:.2f}%",
        f"- Center MAE px: {frozen_summary['center_mae_px_224']:.4f}",
        f"- Tip MAE px: {frozen_summary['tip_mae_px_224']:.4f}",
        f"- Angle MAE deg: {frozen_summary['angle_mae_degrees']:.4f}",
        "",
        "## Final Val Replay",
        f"- Accepted MAE: {final_summary['accepted_mae_c']:.4f} C",
        f"- Acceptance rate: {final_summary['acceptance_rate']:.4f}",
        f"- Worst accepted error: {final_summary['worst_accepted_error_c']:.4f} C",
        f"- Accepted >20 C failures: {int(final_summary['accepted_gt20_failures'])}",
        f"- Under 2/5/10 C: {final_summary['percentage_under_2c']:.2f}% / {final_summary['percentage_under_5c']:.2f}% / {final_summary['percentage_under_10c']:.2f}%",
        f"- Center MAE px: {final_summary['center_mae_px_224']:.4f}",
        f"- Tip MAE px: {final_summary['tip_mae_px_224']:.4f}",
        f"- Angle MAE deg: {final_summary['angle_mae_degrees']:.4f}",
        "",
        "## Feasibility",
        f"- Standard QAT feasible: {feasibility['standard_qat_feasible']}",
        f"- Preferred strategy: {feasibility['preferred_training_strategy']}",
        f"- tensorflow_model_optimization available: {feasibility['tfmot_available']}",
        "",
        "## Calibration provenance",
        f"- Selected candidate: {selected_candidate_name}",
        f"- Calibration selection basis: {calibration_json.get('selection_basis', 'unknown')}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _copy_if_exists(source_path: Path, destination_path: Path) -> None:
    """Copy an existing artifact to the requested destination."""

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source artifact: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def main() -> None:
    """Fine-tune geometry_heatmap_v2 with quantization robustness."""

    parser = argparse.ArgumentParser(description="Train geometry_heatmap_v2 QAT fallback")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--calibration-json-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--thresholds-path", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--decoder-path", type=Path, default=DEFAULT_DECODER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frozen-epochs", type=int, default=40)
    parser.add_argument("--unfrozen-epochs", type=int, default=30)
    parser.add_argument("--frozen-learning-rate", type=float, default=3e-5)
    parser.add_argument("--unfrozen-learning-rate", type=float, default=1e-5)
    parser.add_argument("--heatmap-size", type=int, default=56)
    parser.add_argument("--sigma-pixels", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-noise-stddev", type=float, default=0.008)
    parser.add_argument("--distillation-weight", type=float, default=0.15)
    parser.add_argument("--heatmap-distillation-weight", type=float, default=0.25)
    parser.add_argument("--confidence-distillation-weight", type=float, default=0.05)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = _resolve_path(repo_root, args.model_path)
    manifest_path = _resolve_path(repo_root, args.manifest_path)
    calibration_json_path = _resolve_path(repo_root, args.calibration_json_path)
    thresholds_path = _resolve_path(repo_root, args.thresholds_path)
    decoder_path = _resolve_path(repo_root, args.decoder_path)
    output_dir = _resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with decoder_path.open("r", encoding="utf-8") as handle:
        decoder_payload = json.load(handle)
    decode_method = str(decoder_payload["decode_method"])
    window_size = int(decoder_payload["window_size"])
    if decode_method != "softargmax" or window_size != 3:
        raise RuntimeError(f"Expected corrected decoder softargmax w3, found {decode_method} w{window_size}.")

    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_json_path)
    with thresholds_path.open("r", encoding="utf-8") as handle:
        thresholds_payload = json.load(handle)
    selected_thresholds = thresholds_payload["selected_thresholds"]
    thresholds = GeometryGuardrailThresholds(
        center_peak_min=float(selected_thresholds["center_peak_min"]),
        tip_peak_min=float(selected_thresholds["tip_peak_min"]),
        confidence_min=float(selected_thresholds["confidence_min"]),
        max_heatmap_entropy=float(selected_thresholds["max_heatmap_entropy"]),
        max_heatmap_spread_px=float(selected_thresholds["max_heatmap_spread_px"]),
        center_tip_distance_ratio_min=float(selected_thresholds["center_tip_distance_ratio_min"]),
        center_tip_distance_ratio_max=float(selected_thresholds["center_tip_distance_ratio_max"]),
        edge_margin_px=float(selected_thresholds["edge_margin_px"]),
        temperature_physical_margin_c=float(selected_thresholds["temperature_physical_range_margin_c"]),
        clamp_temperature_to_physical_range=bool(selected_thresholds["clamp_temperature_to_physical_range"]),
        minimum_celsius=float(selected_thresholds["minimum_celsius"]),
        maximum_celsius=float(selected_thresholds["maximum_celsius"]),
    )

    examples = load_clean_geometry_examples(manifest_path)
    train_examples = select_examples_from_split(examples, split="train")
    val_examples = select_examples_from_split(examples, split="val")
    train_sequence = GeometryHeatmapSequence(
        train_examples,
        base_path=repo_root,
        batch_size=args.batch_size,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        seed=args.seed,
    )
    val_samples = load_heatmap_samples(
        val_examples,
        repo_root,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
    )
    val_x = np.stack([sample.crop_image for sample in val_samples], axis=0).astype(np.float32)

    base_model = load_geometry_heatmap_keras_model(model_path)
    reference_model = load_geometry_heatmap_keras_model(model_path)
    reference_model.trainable = False

    qat_model = GeometryQATTrainingModel(
        base_model=base_model,
        reference_model=reference_model,
        output_noise_stddev=args.output_noise_stddev,
        distillation_weight=args.distillation_weight,
        heatmap_distillation_weight=args.heatmap_distillation_weight,
        confidence_distillation_weight=args.confidence_distillation_weight,
    )
    _set_backbone_trainability(base_model, trainable_last_block=False)
    _compile_model(qat_model, learning_rate=args.frozen_learning_rate)

    frozen_best_path = output_dir / "model_qat_frozen_best.keras"
    frozen_callback = ReplayBoardMetricCallback(
        metric_prefix="board",
        samples=val_samples,
        inputs=val_x,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        decode_method=decode_method,
        window_size=window_size,
        best_model_path=frozen_best_path,
    )
    frozen_history = qat_model.fit(
        train_sequence,
        validation_data=(val_x, {
            "center_heatmap": np.stack([sample.center_heatmap[..., np.newaxis] for sample in val_samples], axis=0),
            "tip_heatmap": np.stack([sample.tip_heatmap[..., np.newaxis] for sample in val_samples], axis=0),
            "confidence": np.ones((len(val_samples), 1), dtype=np.float32),
        }),
        epochs=args.frozen_epochs,
        callbacks=[
            frozen_callback,
            keras.callbacks.EarlyStopping(
                monitor="val_board_accepted_mae",
                mode="min",
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_board_accepted_mae",
                mode="min",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
            keras.callbacks.TerminateOnNaN(),
        ],
        verbose=2,
    )
    frozen_rows, frozen_summary = _evaluate_replay_split(
        qat_model,
        val_samples,
        calibration_candidate=calibration_candidate,
        thresholds=thresholds,
        decode_method=decode_method,
        window_size=window_size,
    )
    frozen_weights = base_model.get_weights()

    _write_csv(frozen_rows, output_dir / "val_predictions_frozen.csv")

    needs_unfreeze = (
        frozen_summary["accepted_mae_c"] > 4.5
        or frozen_summary["tip_mae_px_224"] > 18.0
        or frozen_summary["center_mae_px_224"] > 8.0
    )

    final_stage = "frozen"
    final_history = frozen_history.history
    final_rows = frozen_rows
    final_summary = frozen_summary
    selected_best_path = frozen_best_path
    if needs_unfreeze:
        _set_backbone_trainability(base_model, trainable_last_block=True)
        _compile_model(qat_model, learning_rate=args.unfrozen_learning_rate)
        unfrozen_best_path = output_dir / "model_qat_unfrozen_best.keras"
        unfrozen_callback = ReplayBoardMetricCallback(
            metric_prefix="board",
            samples=val_samples,
            inputs=val_x,
            calibration_candidate=calibration_candidate,
            thresholds=thresholds,
            decode_method=decode_method,
            window_size=window_size,
            best_model_path=unfrozen_best_path,
        )
        unfrozen_history = qat_model.fit(
            train_sequence,
            validation_data=(val_x, {
                "center_heatmap": np.stack([sample.center_heatmap[..., np.newaxis] for sample in val_samples], axis=0),
                "tip_heatmap": np.stack([sample.tip_heatmap[..., np.newaxis] for sample in val_samples], axis=0),
                "confidence": np.ones((len(val_samples), 1), dtype=np.float32),
            }),
            epochs=args.unfrozen_epochs,
            callbacks=[
                unfrozen_callback,
                keras.callbacks.EarlyStopping(
                    monitor="val_board_accepted_mae",
                    mode="min",
                    patience=10,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_board_accepted_mae",
                    mode="min",
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                ),
                keras.callbacks.TerminateOnNaN(),
            ],
            verbose=2,
        )
        unfrozen_rows, unfrozen_summary = _evaluate_replay_split(
            qat_model,
            val_samples,
            calibration_candidate=calibration_candidate,
            thresholds=thresholds,
            decode_method=decode_method,
            window_size=window_size,
        )
        if unfrozen_summary["accepted_mae_c"] <= frozen_summary["accepted_mae_c"]:
            final_stage = "unfrozen"
            final_history = unfrozen_history.history
            final_rows = unfrozen_rows
            final_summary = unfrozen_summary
            selected_best_path = unfrozen_best_path
        else:
            base_model.set_weights(frozen_weights)
            final_stage = "frozen"
            final_history = frozen_history.history
            final_rows = frozen_rows
            final_summary = frozen_summary
            selected_best_path = frozen_best_path

    output_model_path = output_dir / "model_qat.keras"
    selected_model_path = output_dir / "model.keras"
    selected_best_model_path = output_dir / "best_model.keras"
    base_model.save(output_model_path)
    shutil.copy2(output_model_path, selected_model_path)
    _copy_if_exists(selected_best_path, selected_best_model_path)
    _write_csv(final_rows, output_dir / "val_predictions.csv")
    _write_history(final_history, output_dir / "history.csv")
    summary_payload = {
        "selected_stage": final_stage,
        "selected_model_path": str(selected_model_path),
        "selected_best_model_path": str(selected_best_model_path),
        "decoder_method": decode_method,
        "window_size": window_size,
        "loss_weights": LOSS_WEIGHTS,
        "output_noise_stddev": args.output_noise_stddev,
        "distillation_weight": args.distillation_weight,
        "heatmap_distillation_weight": args.heatmap_distillation_weight,
        "confidence_distillation_weight": args.confidence_distillation_weight,
        "feasibility": {
            "tfmot_available": importlib.util.find_spec("tensorflow_model_optimization") is not None,
            "preferred_training_strategy": "quantization_noise_fine_tuning",
        },
        "selected_calibration_candidate": calibration_candidate.name,
        "selected_calibration_json": calibration_json,
        "frozen_val_summary": frozen_summary,
        "final_val_summary": final_summary,
    }
    _write_json(summary_payload, output_dir / "summary.json")
    _write_json(
        {
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "calibration_json_path": str(calibration_json_path),
            "thresholds_path": str(thresholds_path),
            "decoder_path": str(decoder_path),
            "output_model_path": str(output_model_path),
            "selected_model_path": str(selected_model_path),
            "selected_best_model_path": str(selected_best_model_path),
            "selected_stage": final_stage,
            "decoder_method": decode_method,
            "window_size": window_size,
            "loss_weights": LOSS_WEIGHTS,
            "output_noise_stddev": args.output_noise_stddev,
            "distillation_weight": args.distillation_weight,
            "heatmap_distillation_weight": args.heatmap_distillation_weight,
            "confidence_distillation_weight": args.confidence_distillation_weight,
        },
        output_dir / "config.json",
    )
    _write_markdown_report(
        output_path=output_dir / "replay_report.md",
        feasibility={
            "preferred_training_strategy": "quantization_noise_fine_tuning",
            "standard_qat_feasible": importlib.util.find_spec("tensorflow_model_optimization") is not None,
            "tfmot_available": importlib.util.find_spec("tensorflow_model_optimization") is not None,
        },
        frozen_summary=frozen_summary,
        final_summary=final_summary,
        selected_stage=final_stage,
        decode_method=decode_method,
        window_size=window_size,
        selected_candidate_name=calibration_candidate.name,
        calibration_json=calibration_json,
    )

    print(f"[QAT] Output model: {output_model_path}", flush=True)
    print(f"[QAT] Selected stage: {final_stage}", flush=True)
    print(f"[QAT] Val accepted MAE: {final_summary['accepted_mae_c']:.4f} C", flush=True)
    print(f"[QAT] Val acceptance rate: {final_summary['acceptance_rate']:.4f}", flush=True)
    print(f"[QAT] Val worst accepted error: {final_summary['worst_accepted_error_c']:.4f} C", flush=True)


if __name__ == "__main__":
    main()
