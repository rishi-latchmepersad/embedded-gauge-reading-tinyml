#!/usr/bin/env python3
"""Tiny overfit sanity check for the geometry heatmap pipeline."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)
from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    JitterParams,
    SourceGeometryExample,
    create_jittered_crop,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import (
    heatmap_index_to_crop_pixel,
    load_clean_geometry_examples,
    make_target_heatmaps,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.heatmap_losses import (
    combined_heatmap_loss,
    mean_predicted_heatmap_peak,
    softargmax_coordinate_mae,
)
from embedded_gauge_reading_tinyml.models_geometry import build_mobilenetv2_geometry_heatmap_v1
from embedded_gauge_reading_tinyml.heatmap_utils import argmax_2d, softargmax_2d


@dataclass(frozen=True)
class TinySample:
    """One overfit sample with loaded crop and labels."""

    example: SourceGeometryExample
    crop_image: np.ndarray
    metadata: dict[str, float]
    center_heatmap: np.ndarray
    tip_heatmap: np.ndarray


class StopWhenFitEnough(keras.callbacks.Callback):
    """Stop training once the tiny set is clearly overfit."""

    def __init__(self, *, center_mae_px: float, tip_mae_px: float, peak_threshold: float) -> None:
        """Store the target thresholds for the overfit gate."""

        super().__init__()
        self.center_mae_px = center_mae_px
        self.tip_mae_px = tip_mae_px
        self.peak_threshold = peak_threshold

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Stop as soon as the model is both accurate and peaky enough."""

        if logs is None:
            return

        center_mae = logs.get("center_heatmap_softargmax_coordinate_mae")
        tip_mae = logs.get("tip_heatmap_softargmax_coordinate_mae")
        center_peak = logs.get("center_heatmap_mean_predicted_heatmap_peak")
        tip_peak = logs.get("tip_heatmap_mean_predicted_heatmap_peak")

        if (
            center_mae is not None
            and tip_mae is not None
            and center_peak is not None
            and tip_peak is not None
            and center_mae <= self.center_mae_px
            and tip_mae <= self.tip_mae_px
            and center_peak >= self.peak_threshold
            and tip_peak >= self.peak_threshold
        ):
            print(
                f"Stopping early at epoch {epoch + 1}: "
                f"center_mae={center_mae:.3f}, tip_mae={tip_mae:.3f}, "
                f"center_peak={center_peak:.3f}, tip_peak={tip_peak:.3f}"
            )
            self.model.stop_training = True


def _make_jitter(rng: np.random.Generator, *, shift_range: int, scale_range: tuple[float, float], aspect_range: tuple[float, float]) -> JitterParams:
    """Create a mild jitter configuration for tiny overfit samples."""

    if shift_range == 0 and scale_range == (1.0, 1.0) and aspect_range == (1.0, 1.0):
        return JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)

    return JitterParams(
        shift_x=int(rng.integers(-shift_range, shift_range + 1)),
        shift_y=int(rng.integers(-shift_range, shift_range + 1)),
        scale=float(rng.uniform(scale_range[0], scale_range[1])),
        aspect=float(rng.uniform(aspect_range[0], aspect_range[1])),
    )


def _load_tiny_samples(
    *,
    examples: list[SourceGeometryExample],
    base_path: Path,
    heatmap_size: int,
    sigma_pixels: float,
    shift_range: int,
    scale_range: tuple[float, float],
    aspect_range: tuple[float, float],
    seed: int,
) -> list[TinySample]:
    """Load the tiny overfit dataset from a deterministic subset of examples."""

    rng = np.random.default_rng(seed)
    samples: list[TinySample] = []

    for example in examples:
        jitter = _make_jitter(
            rng,
            shift_range=shift_range,
            scale_range=scale_range,
            aspect_range=aspect_range,
        )
        crop = create_jittered_crop(example, jitter)
        if not crop.accepted:
            continue

        image_path = base_path / crop.source_image_path
        with Image.open(image_path) as source_image:
            source_image = source_image.convert("RGB")
            crop_box = (crop.crop_x1, crop.crop_y1, crop.crop_x2, crop.crop_y2)
            crop_image = np.asarray(
                source_image.crop(crop_box).resize((224, 224), Image.LANCZOS),
                dtype=np.float32,
            ) / 255.0
        metadata = {
            "center_x_224": crop.center_x_224,
            "center_y_224": crop.center_y_224,
            "tip_x_224": crop.tip_x_224,
            "tip_y_224": crop.tip_y_224,
            "center_x_norm": crop.center_x_normalized,
            "center_y_norm": crop.center_y_normalized,
            "tip_x_norm": crop.tip_x_normalized,
            "tip_y_norm": crop.tip_y_normalized,
        }
        center_heatmap, tip_heatmap = make_target_heatmaps(
            center_x_norm=metadata["center_x_norm"],
            center_y_norm=metadata["center_y_norm"],
            tip_x_norm=metadata["tip_x_norm"],
            tip_y_norm=metadata["tip_y_norm"],
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
        )
        samples.append(
            TinySample(
                example=example,
                crop_image=crop_image,
                metadata=metadata,
                center_heatmap=center_heatmap,
                tip_heatmap=tip_heatmap,
            )
        )

    return samples


def _make_training_arrays(samples: list[TinySample]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert tiny samples into model-ready NumPy arrays."""

    x = np.stack([sample.crop_image for sample in samples], axis=0).astype(np.float32)
    y = {
        "center_heatmap": np.stack([sample.center_heatmap for sample in samples], axis=0).astype(np.float32)[..., np.newaxis],
        "tip_heatmap": np.stack([sample.tip_heatmap for sample in samples], axis=0).astype(np.float32)[..., np.newaxis],
        "confidence": np.ones((len(samples), 1), dtype=np.float32),
    }
    return x, y


def _decode_sample(
    *,
    sample: TinySample,
    prediction: list[np.ndarray],
    heatmap_size: int,
) -> dict[str, Any]:
    """Decode one model prediction into evaluation metrics."""

    center_heatmap = np.asarray(prediction[0][0, ..., 0], dtype=np.float32)
    tip_heatmap = np.asarray(prediction[1][0, ..., 0], dtype=np.float32)
    confidence = float(prediction[2][0, 0])

    center_argmax_row, center_argmax_col = argmax_2d(center_heatmap)
    tip_argmax_row, tip_argmax_col = argmax_2d(tip_heatmap)
    center_soft_row, center_soft_col = softargmax_2d(center_heatmap)
    tip_soft_row, tip_soft_col = softargmax_2d(tip_heatmap)

    center_argmax_x = heatmap_index_to_crop_pixel(center_argmax_col, heatmap_size=heatmap_size)
    center_argmax_y = heatmap_index_to_crop_pixel(center_argmax_row, heatmap_size=heatmap_size)
    tip_argmax_x = heatmap_index_to_crop_pixel(tip_argmax_col, heatmap_size=heatmap_size)
    tip_argmax_y = heatmap_index_to_crop_pixel(tip_argmax_row, heatmap_size=heatmap_size)

    center_soft_x = heatmap_index_to_crop_pixel(center_soft_col, heatmap_size=heatmap_size)
    center_soft_y = heatmap_index_to_crop_pixel(center_soft_row, heatmap_size=heatmap_size)
    tip_soft_x = heatmap_index_to_crop_pixel(tip_soft_col, heatmap_size=heatmap_size)
    tip_soft_y = heatmap_index_to_crop_pixel(tip_soft_row, heatmap_size=heatmap_size)

    true_center_x = sample.metadata["center_x_224"]
    true_center_y = sample.metadata["center_y_224"]
    true_tip_x = sample.metadata["tip_x_224"]
    true_tip_y = sample.metadata["tip_y_224"]

    predicted_angle = angle_degrees_from_center_to_tip(
        center_soft_x,
        center_soft_y,
        tip_soft_x,
        tip_soft_y,
    )
    true_angle = angle_degrees_from_center_to_tip(true_center_x, true_center_y, true_tip_x, true_tip_y)
    predicted_temp = celsius_from_inner_dial_angle_degrees(predicted_angle)

    center_argmax_error = math.hypot(center_argmax_x - true_center_x, center_argmax_y - true_center_y)
    tip_argmax_error = math.hypot(tip_argmax_x - true_tip_x, tip_argmax_y - true_tip_y)
    center_soft_error = math.hypot(center_soft_x - true_center_x, center_soft_y - true_center_y)
    tip_soft_error = math.hypot(tip_soft_x - true_tip_x, tip_soft_y - true_tip_y)

    return {
        "image_path": sample.example.image_path,
        "split": sample.example.split,
        "temperature_c": sample.example.temperature_c,
        "predicted_temperature_c": predicted_temp,
        "absolute_error_c": abs(predicted_temp - sample.example.temperature_c),
        "true_angle_degrees": true_angle,
        "predicted_angle_degrees": predicted_angle,
        "angle_error_degrees": abs(circular_angle_error_degrees(predicted_angle, true_angle)),
        "confidence": confidence,
        "center_heatmap_max": float(np.max(center_heatmap)),
        "tip_heatmap_max": float(np.max(tip_heatmap)),
        "center_heatmap_std": float(np.std(center_heatmap)),
        "tip_heatmap_std": float(np.std(tip_heatmap)),
        "center_argmax_error": center_argmax_error,
        "tip_argmax_error": tip_argmax_error,
        "center_softargmax_error": center_soft_error,
        "tip_softargmax_error": tip_soft_error,
        "predicted_center_x_224_softargmax": center_soft_x,
        "predicted_center_y_224_softargmax": center_soft_y,
        "predicted_tip_x_224_softargmax": tip_soft_x,
        "predicted_tip_y_224_softargmax": tip_soft_y,
    }


def _write_predictions_csv(predictions: list[dict[str, Any]], output_path: Path) -> None:
    """Persist predictions for later inspection."""

    fieldnames = list(predictions[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


def _write_history_csv(history: keras.callbacks.History, output_path: Path) -> None:
    """Persist the Keras history in a compact CSV form."""

    history_dict = history.history
    epochs = len(history_dict.get("loss", []))
    metric_names = list(history_dict.keys())
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("epoch," + ",".join(metric_names) + "\n")
        for index in range(epochs):
            values = [str(history_dict[name][index]) for name in metric_names]
            handle.write(f"{index + 1}," + ",".join(values) + "\n")


def _write_report(
    *,
    samples: list[TinySample],
    predictions: list[dict[str, Any]],
    output_path: Path,
    training_summary: dict[str, float],
    sigma_pixels: float,
) -> None:
    """Write the tiny overfit report with success criteria."""

    center_mae = float(np.mean([row["center_softargmax_error"] for row in predictions]))
    tip_mae = float(np.mean([row["tip_softargmax_error"] for row in predictions]))
    temp_mae = float(np.mean([row["absolute_error_c"] for row in predictions]))
    angle_mae = float(np.mean([row["angle_error_degrees"] for row in predictions]))
    center_peak = float(np.mean([row["center_heatmap_max"] for row in predictions]))
    tip_peak = float(np.mean([row["tip_heatmap_max"] for row in predictions]))
    under_2c = float(np.mean([row["absolute_error_c"] < 2.0 for row in predictions]) * 100.0)
    under_5c = float(np.mean([row["absolute_error_c"] < 5.0 for row in predictions]) * 100.0)
    under_10c = float(np.mean([row["absolute_error_c"] < 10.0 for row in predictions]) * 100.0)

    passed = center_mae < 3.0 and tip_mae < 5.0 and temp_mae < 3.0 and center_peak > 0.5 and tip_peak > 0.5

    worst_3 = sorted(predictions, key=lambda row: row["absolute_error_c"], reverse=True)[:3]

    lines = [
        "# Geometry Heatmap Tiny Overfit v1",
        "",
        "## Run Summary",
        "",
        f"- Samples: {len(samples)}",
        f"- Sigma: {sigma_pixels:.2f} pixels",
        f"- Epochs trained: {int(training_summary['epochs'])}",
        f"- Final loss: {training_summary['loss']:.6f}",
        f"- Final center softargmax MAE: {training_summary['center_softargmax_mae']:.6f}",
        f"- Final tip softargmax MAE: {training_summary['tip_softargmax_mae']:.6f}",
        f"- Final center peak: {training_summary['center_peak']:.6f}",
        f"- Final tip peak: {training_summary['tip_peak']:.6f}",
        "",
        "## Holdout-On-Same-8 Metrics",
        "",
        f"- Center MAE: {center_mae:.3f} px",
        f"- Tip MAE: {tip_mae:.3f} px",
        f"- Temperature MAE: {temp_mae:.3f} C",
        f"- Angle MAE: {angle_mae:.3f} deg",
        f"- Under 2 C: {under_2c:.1f}%",
        f"- Under 5 C: {under_5c:.1f}%",
        f"- Under 10 C: {under_10c:.1f}%",
        f"- Mean center peak: {center_peak:.4f}",
        f"- Mean tip peak: {tip_peak:.4f}",
        "",
        "## Success Criteria",
        "",
        f"- Center MAE < 3 px: {'yes' if center_mae < 3.0 else 'no'}",
        f"- Tip MAE < 5 px: {'yes' if tip_mae < 5.0 else 'no'}",
        f"- Temperature MAE < 3 C: {'yes' if temp_mae < 3.0 else 'no'}",
        f"- Mean center peak > 0.5: {'yes' if center_peak > 0.5 else 'no'}",
        f"- Mean tip peak > 0.5: {'yes' if tip_peak > 0.5 else 'no'}",
        f"- Overall result: {'PASS' if passed else 'FAIL'}",
        "",
        "## Worst 3 Samples",
        "",
        "| Image | Abs Error (C) | Center Err (px) | Tip Err (px) | Peak Center | Peak Tip |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in worst_3:
        lines.append(
            f"| {Path(row['image_path']).name} | {row['absolute_error_c']:.3f} | "
            f"{row['center_softargmax_error']:.3f} | {row['tip_softargmax_error']:.3f} | "
            f"{row['center_heatmap_max']:.4f} | {row['tip_heatmap_max']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If this pass/fail gate fails, the heatmap loss is still not giving the model enough signal to fit even eight examples.",
            "- If it passes, the next step is a controlled v2 training run with the same weighted heatmap objective and a small auxiliary coordinate penalty.",
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Train on eight samples and verify the model can overfit them."""

    parser = argparse.ArgumentParser(description="Tiny overfit sanity check for geometry heatmaps")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Clean geometry manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_tiny_overfit_v1"),
        help="Training artifact directory.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_tiny_overfit_v1.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Number of training samples to use.")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum epochs to train.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the tiny overfit.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap size.")
    parser.add_argument("--sigma-pixels", type=float, default=2.5, help="Gaussian sigma in heatmap pixels.")
    parser.add_argument("--jitter-shift", type=int, default=0, help="Max jitter shift in pixels.")
    parser.add_argument("--jitter-scale-min", type=float, default=1.0, help="Minimum jitter scale.")
    parser.add_argument("--jitter-scale-max", type=float, default=1.0, help="Maximum jitter scale.")
    parser.add_argument("--jitter-aspect-min", type=float, default=1.0, help="Minimum jitter aspect ratio.")
    parser.add_argument("--jitter-aspect-max", type=float, default=1.0, help="Maximum jitter aspect ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_clean_geometry_examples(manifest_path)
    train_examples = select_examples_from_split(examples, split="train", limit=args.limit * 4)
    if len(train_examples) < args.limit:
        raise RuntimeError(f"Only found {len(train_examples)} clean train rows; need {args.limit}.")

    selected_examples = train_examples[: args.limit]
    samples = _load_tiny_samples(
        examples=selected_examples,
        base_path=base_path,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        shift_range=args.jitter_shift,
        scale_range=(args.jitter_scale_min, args.jitter_scale_max),
        aspect_range=(args.jitter_aspect_min, args.jitter_aspect_max),
        seed=args.seed,
    )

    if len(samples) < args.limit:
        raise RuntimeError(f"Only built {len(samples)} training samples; expected {args.limit}.")

    x_train, y_train = _make_training_arrays(samples)

    model = build_mobilenetv2_geometry_heatmap_v1(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=True,
        heatmap_size=args.heatmap_size,
        learning_rate=args.learning_rate,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            "center_heatmap": combined_heatmap_loss,
            "tip_heatmap": combined_heatmap_loss,
            "confidence": keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "center_heatmap": 1.0,
            "tip_heatmap": 1.0,
            "confidence": 0.05,
        },
        metrics={
            "center_heatmap": [softargmax_coordinate_mae, mean_predicted_heatmap_peak],
            "tip_heatmap": [softargmax_coordinate_mae, mean_predicted_heatmap_peak],
        },
    )

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.CSVLogger(str(output_dir / "fit_log.csv")),
        StopWhenFitEnough(center_mae_px=0.75, tip_mae_px=1.25, peak_threshold=0.5),
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(output_dir / "model.keras")
    _write_history_csv(history, output_dir / "history.csv")

    predictions: list[dict[str, Any]] = []
    for sample in samples:
        prediction = model.predict(sample.crop_image[np.newaxis, ...], verbose=0)
        predictions.append(_decode_sample(sample=sample, prediction=prediction, heatmap_size=args.heatmap_size))

    _write_predictions_csv(predictions, output_dir / "predictions.csv")

    training_summary = {
        "epochs": float(len(history.history.get("loss", []))),
        "loss": float(history.history.get("loss", [math.nan])[-1]),
        "center_softargmax_mae": float(history.history.get("center_heatmap_softargmax_coordinate_mae", [math.nan])[-1]),
        "tip_softargmax_mae": float(history.history.get("tip_heatmap_softargmax_coordinate_mae", [math.nan])[-1]),
        "center_peak": float(history.history.get("center_heatmap_mean_predicted_heatmap_peak", [math.nan])[-1]),
        "tip_peak": float(history.history.get("tip_heatmap_mean_predicted_heatmap_peak", [math.nan])[-1]),
    }

    _write_report(
        samples=samples,
        predictions=predictions,
        output_path=report_path,
        training_summary=training_summary,
        sigma_pixels=args.sigma_pixels,
    )

    print(f"Artifacts written to {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
