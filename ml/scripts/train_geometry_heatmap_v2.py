#!/usr/bin/env python3
"""Train the full geometry heatmap v2 model on clean train rows only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
from embedded_gauge_reading_tinyml.heatmap_losses import (
    mean_predicted_heatmap_peak,
    softargmax_coordinate_mae,
    weighted_center_heatmap_loss,
    weighted_tip_heatmap_loss,
)
from embedded_gauge_reading_tinyml.models_geometry import build_mobilenetv2_geometry_heatmap_v1


TRAIN_JITTER_SHIFT_MIN_PX: int = 8
TRAIN_JITTER_SHIFT_MAX_PX: int = 12
TRAIN_JITTER_SCALE_MIN: float = 0.90
TRAIN_JITTER_SCALE_MAX: float = 1.15
TRAIN_JITTER_ASPECT_MIN: float = 0.95
TRAIN_JITTER_ASPECT_MAX: float = 1.05

WEIGHTS = {
    "center_heatmap_loss_weight": 2.0,
    "tip_heatmap_loss_weight": 1.0,
    "confidence_loss_weight": 0.1,
}


@dataclass(frozen=True)
class SplitBatch:
    """A preloaded deterministic batch for evaluation."""

    samples: list[HeatmapSample]
    x: np.ndarray
    y: dict[str, np.ndarray]


class GeometryHeatmapSequence(keras.utils.Sequence):
    """On-the-fly jittered training sequence for the heatmap model."""

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
        self.batch_size = batch_size
        self.heatmap_size = heatmap_size
        self.sigma_pixels = sigma_pixels
        self.seed = seed
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
        """Build one jittered batch."""

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


def _write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write a JSON file with stable formatting."""

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write prediction rows to CSV, dropping large array fields."""

    if not rows:
        raise ValueError("Cannot write an empty prediction table.")
    skip_keys = {"pred_center_heatmap_array", "pred_tip_heatmap_array"}
    fieldnames = [key for key in rows[0].keys() if key not in skip_keys]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def _write_history(history: keras.callbacks.History, output_path: Path) -> None:
    """Persist a Keras history object as CSV."""

    history_dict = history.history
    metric_names = list(history_dict.keys())
    epochs = len(history_dict.get("loss", []))
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("epoch," + ",".join(metric_names) + "\n")
        for index in range(epochs):
            values = [str(history_dict[name][index]) for name in metric_names]
            handle.write(f"{index + 1}," + ",".join(values) + "\n")


def _get_backbone(model: keras.Model) -> keras.Model:
    """Find the nested MobileNetV2 backbone inside the heatmap model."""

    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower():
            return layer
    raise RuntimeError("Could not find the nested MobileNetV2 backbone.")


def _set_backbone_trainability(model: keras.Model, *, trainable_last_block: bool) -> None:
    """Freeze the full backbone or unfreeze only the last MobileNetV2 block."""

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


def _compile_model(model: keras.Model, *, learning_rate: float, weights: dict[str, float]) -> None:
    """Compile the model with the v2 weighted heatmap objective."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "center_heatmap": weighted_center_heatmap_loss,
            "tip_heatmap": weighted_tip_heatmap_loss,
            "confidence": keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "center_heatmap": weights["center_heatmap_loss_weight"],
            "tip_heatmap": weights["tip_heatmap_loss_weight"],
            "confidence": weights["confidence_loss_weight"],
        },
        metrics={
            "center_heatmap": [softargmax_coordinate_mae, mean_predicted_heatmap_peak],
            "tip_heatmap": [softargmax_coordinate_mae, mean_predicted_heatmap_peak],
        },
    )


def _make_model(*, learning_rate: float, backbone_frozen: bool) -> keras.Model:
    """Build the heatmap model used for both training stages."""

    model = build_mobilenetv2_geometry_heatmap_v1(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=backbone_frozen,
        heatmap_size=56,
        learning_rate=learning_rate,
    )
    model._name = "mobilenetv2_geometry_heatmap_v2"  # keep the saved artifact obvious
    return model


def _load_split_batch(
    examples: list[Any],
    base_path: Path,
    *,
    heatmap_size: int,
    sigma_pixels: float,
) -> SplitBatch:
    """Load a deterministic identity-crop batch for evaluation."""

    samples = load_heatmap_samples(
        examples,
        base_path,
        heatmap_size=heatmap_size,
        sigma_pixels=sigma_pixels,
        jitter=None,
    )
    x = np.stack([sample.crop_image for sample in samples], axis=0).astype(np.float32)
    y = {
        "center_heatmap": np.stack([sample.center_heatmap for sample in samples], axis=0).astype(np.float32)[..., np.newaxis],
        "tip_heatmap": np.stack([sample.tip_heatmap for sample in samples], axis=0).astype(np.float32)[..., np.newaxis],
        "confidence": np.ones((len(samples), 1), dtype=np.float32),
    }
    return SplitBatch(samples=samples, x=x, y=y)


def _load_training_batch(
    examples: list[Any],
    base_path: Path,
    *,
    heatmap_size: int,
    sigma_pixels: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build one deterministic jittered training batch in memory."""

    batch_x: list[np.ndarray] = []
    batch_center: list[np.ndarray] = []
    batch_tip: list[np.ndarray] = []

    for index, example in enumerate(examples):
        jitter_rng = np.random.default_rng(seed + index * 997)
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
            base_path,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
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


def _evaluate_model(
    model: keras.Model,
    samples: list[HeatmapSample],
    *,
    calibration_candidate: Any,
    heatmap_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Decode predictions and summarize the resulting metrics."""

    x = np.stack([sample.crop_image for sample in samples], axis=0).astype(np.float32)
    predictions = model.predict(x, verbose=0)
    center_batch = np.asarray(predictions[0], dtype=np.float32)
    tip_batch = np.asarray(predictions[1], dtype=np.float32)
    confidence_batch = np.asarray(predictions[2], dtype=np.float32)

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        confidence = float(np.ravel(confidence_batch[index])[0])
        row = decode_prediction_row(
            sample,
            center_batch[index],
            tip_batch[index],
            confidence,
            calibration_candidate=calibration_candidate,
        )
        rows.append(row)

    summary = _summarize_predictions(rows)
    return rows, summary


def _summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize one split worth of decoded predictions."""

    center_errors = np.asarray([float(row["center_px_mae_224"]) for row in rows], dtype=np.float64)
    tip_errors = np.asarray([float(row["tip_px_mae_224"]) for row in rows], dtype=np.float64)
    angle_errors = np.asarray([float(row["angle_mae_degrees"]) for row in rows], dtype=np.float64)
    current_temp_errors = np.asarray(
        [float(row["absolute_error_c_current_mapping"]) for row in rows],
        dtype=np.float64,
    )
    calibrated_temp_errors = np.asarray(
        [float(row["absolute_error_c_calibrated"]) for row in rows],
        dtype=np.float64,
    )
    center_peaks = np.asarray([float(row["center_heatmap_peak_value"]) for row in rows], dtype=np.float64)
    tip_peaks = np.asarray([float(row["tip_heatmap_peak_value"]) for row in rows], dtype=np.float64)
    confidences = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)

    return {
        "count": float(len(rows)),
        "center_px_mae_224": float(np.mean(center_errors)),
        "tip_px_mae_224": float(np.mean(tip_errors)),
        "angle_mae_degrees": float(np.mean(angle_errors)),
        "temperature_mae_c_current_mapping": float(np.mean(current_temp_errors)),
        "temperature_mae_c_calibrated": float(np.mean(calibrated_temp_errors)),
        "temperature_rmse_c_calibrated": float(np.sqrt(np.mean(np.square(calibrated_temp_errors)))),
        "percentage_under_2c_calibrated": float(np.mean(calibrated_temp_errors < 2.0) * 100.0),
        "percentage_under_5c_calibrated": float(np.mean(calibrated_temp_errors < 5.0) * 100.0),
        "percentage_under_10c_calibrated": float(np.mean(calibrated_temp_errors < 10.0) * 100.0),
        "center_heatmap_peak_mean": float(np.mean(center_peaks)),
        "center_heatmap_peak_median": float(np.median(center_peaks)),
        "tip_heatmap_peak_mean": float(np.mean(tip_peaks)),
        "tip_heatmap_peak_median": float(np.median(tip_peaks)),
        "confidence_mean": float(np.mean(confidences)),
        "confidence_median": float(np.median(confidences)),
        "worst_calibrated_error_c": float(np.max(calibrated_temp_errors)),
    }


def _copy_selected_artifacts(source_dir: Path, output_dir: Path) -> None:
    """Copy the selected variant artifacts to the top level output directory."""

    for name in ["model.keras", "history.csv", "config.json", "val_predictions.csv", "summary.json"]:
        source = source_dir / name
        target = output_dir / name
        if source.exists():
            target.write_bytes(source.read_bytes())


def _save_variant_artifacts(
    *,
    variant_dir: Path,
    model: keras.Model,
    history: keras.callbacks.History,
    val_rows: list[dict[str, Any]],
    summary: dict[str, float],
    config: dict[str, Any],
) -> None:
    """Persist the selected or intermediate variant artifacts."""

    variant_dir.mkdir(parents=True, exist_ok=True)
    model.save(variant_dir / "model.keras")
    _write_history(history, variant_dir / "history.csv")
    _write_csv(val_rows, variant_dir / "val_predictions.csv")
    _write_json({**config, "summary": summary}, variant_dir / "summary.json")
    _write_json(config, variant_dir / "config.json")


def _load_variant_config(args: argparse.Namespace, *, selected_candidate_name: str) -> dict[str, Any]:
    """Build the configuration payload saved with the artifacts."""

    return {
        "manifest_path": str(args.manifest_path),
        "calibration_json_path": str(args.calibration_json_path),
        "input_size": 224,
        "heatmap_size": args.heatmap_size,
        "sigma_pixels": args.sigma_pixels,
        "batch_size": args.batch_size,
        "frozen_learning_rate": args.learning_rate,
        "unfreeze_learning_rate": args.unfreeze_learning_rate,
        "frozen_epochs": args.frozen_epochs,
        "unfrozen_epochs": args.unfrozen_epochs,
        "augmentation": {
            "shift_min_px": TRAIN_JITTER_SHIFT_MIN_PX,
            "shift_max_px": TRAIN_JITTER_SHIFT_MAX_PX,
            "scale_min": TRAIN_JITTER_SCALE_MIN,
            "scale_max": TRAIN_JITTER_SCALE_MAX,
            "aspect_min": TRAIN_JITTER_ASPECT_MIN,
            "aspect_max": TRAIN_JITTER_ASPECT_MAX,
        },
        "loss_weights": WEIGHTS,
        "selected_calibration_candidate": selected_candidate_name,
    }


def main() -> None:
    """Train the full heatmap v2 model and keep the best stage."""

    parser = argparse.ArgumentParser(description="Train geometry heatmap v2")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Clean manifest path.",
    )
    parser.add_argument(
        "--calibration-json-path",
        type=Path,
        default=Path("ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json"),
        help="Phase 4.7 calibration artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2"),
        help="Output directory for the selected model artifacts.",
    )
    parser.add_argument(
        "--frozen-output-subdir",
        type=str,
        default="variant_a_frozen",
        help="Subdirectory name for the frozen-backbone stage artifacts.",
    )
    parser.add_argument(
        "--unfrozen-output-subdir",
        type=str,
        default="variant_b_unfrozen",
        help="Subdirectory name for the optional unfrozen stage artifacts.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--frozen-epochs", type=int, default=120, help="Maximum frozen-stage epochs.")
    parser.add_argument("--unfrozen-epochs", type=int, default=80, help="Maximum unfrozen-stage epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Frozen-stage learning rate.")
    parser.add_argument("--unfreeze-learning-rate", type=float, default=1e-5, help="Unfrozen-stage learning rate.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap size.")
    parser.add_argument("--sigma-pixels", type=float, default=5.0, help="Gaussian sigma at heatmap scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    calibration_json_path = (
        base_path / args.calibration_json_path if not args.calibration_json_path.is_absolute() else args.calibration_json_path
    )
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_candidate, calibration_json = load_selected_calibration_candidate(calibration_json_path)
    config = _load_variant_config(args, selected_candidate_name=calibration_candidate.name)

    examples = load_clean_geometry_examples(manifest_path)
    train_examples = select_examples_from_split(examples, split="train")
    val_examples = select_examples_from_split(examples, split="val")

    train_x, train_y = _load_training_batch(
        train_examples,
        base_path,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        seed=args.seed,
    )
    val_batch = _load_split_batch(
        val_examples,
        base_path,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
    )

    # Stage A: frozen backbone.
    frozen_model = _make_model(learning_rate=args.learning_rate, backbone_frozen=True)
    _compile_model(frozen_model, learning_rate=args.learning_rate, weights=WEIGHTS)
    frozen_history = frozen_model.fit(
        train_x,
        train_y,
        validation_data=(val_batch.x, val_batch.y),
        epochs=args.frozen_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
            keras.callbacks.TerminateOnNaN(),
        ],
        verbose=2,
    )
    frozen_val_rows, frozen_val_summary = _evaluate_model(
        frozen_model,
        val_batch.samples,
        calibration_candidate=calibration_candidate,
        heatmap_size=args.heatmap_size,
    )
    frozen_weights = frozen_model.get_weights()

    frozen_dir = output_dir / args.frozen_output_subdir
    frozen_config = {
        **config,
        "stage": "frozen",
        "learning_rate": args.learning_rate,
        "selected_variant": "frozen",
    }
    _save_variant_artifacts(
        variant_dir=frozen_dir,
        model=frozen_model,
        history=frozen_history,
        val_rows=frozen_val_rows,
        summary=frozen_val_summary,
        config=frozen_config,
    )

    # Decide whether to unfreeze the last block based on the calibrated validation gap.
    needs_unfreeze = (
        frozen_val_summary["temperature_mae_c_calibrated"] > 7.91
        or frozen_val_summary["tip_px_mae_224"] > 21.82
        or frozen_val_summary["center_px_mae_224"] > 11.30
    )

    final_model = frozen_model
    final_history = frozen_history
    final_rows = frozen_val_rows
    final_summary = frozen_val_summary
    final_stage = "frozen"

    unfrozen_summary: dict[str, float] | None = None

    if needs_unfreeze:
        _set_backbone_trainability(frozen_model, trainable_last_block=True)
        _compile_model(frozen_model, learning_rate=args.unfreeze_learning_rate, weights=WEIGHTS)
        unfrozen_history = frozen_model.fit(
            train_x,
            train_y,
            validation_data=(val_batch.x, val_batch.y),
            epochs=args.unfrozen_epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
                keras.callbacks.TerminateOnNaN(),
            ],
            verbose=2,
        )
        unfrozen_val_rows, unfrozen_val_summary = _evaluate_model(
            frozen_model,
            val_batch.samples,
            calibration_candidate=calibration_candidate,
            heatmap_size=args.heatmap_size,
        )
        unfrozen_summary = unfrozen_val_summary

        unfrozen_dir = output_dir / args.unfrozen_output_subdir
        unfrozen_config = {
            **config,
            "stage": "unfrozen",
            "learning_rate": args.unfreeze_learning_rate,
            "selected_variant": "unfrozen",
        }
        _save_variant_artifacts(
            variant_dir=unfrozen_dir,
            model=frozen_model,
            history=unfrozen_history,
            val_rows=unfrozen_val_rows,
            summary=unfrozen_val_summary,
            config=unfrozen_config,
        )

        if unfrozen_val_summary["temperature_mae_c_calibrated"] <= frozen_val_summary["temperature_mae_c_calibrated"]:
            final_model = frozen_model
            final_history = unfrozen_history
            final_rows = unfrozen_val_rows
            final_summary = unfrozen_val_summary
            final_stage = "unfrozen"
        else:
            frozen_model.set_weights(frozen_weights)
            final_model = frozen_model
            final_history = frozen_history
            final_rows = frozen_val_rows
            final_summary = frozen_val_summary
            final_stage = "frozen"

    # Final selected artifacts.
    selected_config = {
        **config,
        "selected_stage": final_stage,
        "frozen_val_summary": frozen_val_summary,
        "unfrozen_val_summary": unfrozen_summary,
        "selection_rule": "lowest calibrated validation temperature MAE",
        "calibration_json": calibration_json,
    }
    _write_json(selected_config, output_dir / "config.json")
    _write_history(final_history, output_dir / "history.csv")
    _write_csv(final_rows, output_dir / "val_predictions.csv")
    final_model.save(output_dir / "model.keras")
    _write_json(final_summary, output_dir / "summary.json")

    print(f"Selected stage: {final_stage}")
    print(f"Selected model: {output_dir / 'model.keras'}")
    print(f"Selected history: {output_dir / 'history.csv'}")
    print(f"Selected config: {output_dir / 'config.json'}")


if __name__ == "__main__":
    main()
