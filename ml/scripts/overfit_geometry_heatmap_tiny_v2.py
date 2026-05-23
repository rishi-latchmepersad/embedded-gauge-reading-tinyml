#!/usr/bin/env python3
"""Tiny overfit gate for the geometry heatmap pipeline v2."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import (
    heatmap_index_to_crop_pixel,
    load_clean_geometry_examples,
    load_identity_crop,
    make_target_heatmaps,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.heatmap_losses import (
    center_priority_heatmap_loss,
    mean_predicted_heatmap_peak,
    softargmax_coordinate_mae,
    tip_priority_heatmap_loss,
)
from embedded_gauge_reading_tinyml.heatmap_utils import argmax_2d, softargmax_2d
from embedded_gauge_reading_tinyml.models_geometry import build_mobilenetv2_geometry_heatmap_v1


@dataclass(frozen=True)
class TinySample:
    """One clean training example used for the overfit gate."""

    image_path: str
    split: str
    temperature_c: float
    crop_image: np.ndarray
    metadata: dict[str, float]
    center_heatmap: np.ndarray
    tip_heatmap: np.ndarray


def _load_tiny_samples(
    *,
    manifest_path: Path,
    base_path: Path,
    heatmap_size: int,
    sigma_pixels: float,
    limit: int,
) -> list[TinySample]:
    """Load the deterministic 8-image training subset with no jitter."""

    examples = load_clean_geometry_examples(manifest_path)
    selected_examples = select_examples_from_split(examples, split="train", limit=limit)
    if len(selected_examples) < limit:
        raise RuntimeError(f"Only found {len(selected_examples)} clean train rows; need {limit}.")

    samples: list[TinySample] = []
    for example in selected_examples:
        crop_image, metadata, _ = load_identity_crop(example, base_path)
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
                image_path=example.image_path,
                split=example.split,
                temperature_c=example.temperature_c,
                crop_image=crop_image,
                metadata=metadata,
                center_heatmap=center_heatmap,
                tip_heatmap=tip_heatmap,
            )
        )

    return samples


def _make_training_arrays(samples: list[TinySample]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert the sample list into model-ready NumPy arrays."""

    x = np.stack([sample.crop_image for sample in samples], axis=0).astype(np.float32)
    y = {
        "center_heatmap": np.stack([sample.center_heatmap for sample in samples], axis=0).astype(np.float32)[..., np.newaxis],
        "tip_heatmap": np.stack([sample.tip_heatmap for sample in samples], axis=0).astype(np.float32)[..., np.newaxis],
        "confidence": np.ones((len(samples), 1), dtype=np.float32),
    }
    return x, y


def _get_backbone(model: keras.Model) -> keras.Model:
    """Find the nested MobileNetV2 backbone inside the heatmap model."""

    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name:
            return layer
    raise RuntimeError("Could not find the nested MobileNetV2 backbone.")


def _set_backbone_trainability(model: keras.Model, *, trainable_last_block: bool) -> None:
    """Freeze the full backbone or unfreeze just the last MobileNetV2 block."""

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


def _compile_model(
    model: keras.Model,
    *,
    learning_rate: float,
    center_heatmap_loss_weight: float,
    tip_heatmap_loss_weight: float,
    confidence_loss_weight: float,
) -> None:
    """Compile the model with the stronger center weighting used for v2."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "center_heatmap": center_priority_heatmap_loss,
            "tip_heatmap": tip_priority_heatmap_loss,
            "confidence": keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "center_heatmap": center_heatmap_loss_weight,
            "tip_heatmap": tip_heatmap_loss_weight,
            "confidence": confidence_loss_weight,
        },
        metrics={
            "center_heatmap": [softargmax_coordinate_mae, mean_predicted_heatmap_peak],
            "tip_heatmap": [softargmax_coordinate_mae, mean_predicted_heatmap_peak],
        },
    )


def _heatmap_point(heatmap: np.ndarray, *, heatmap_size: int, method: str) -> tuple[float, float]:
    """Decode a heatmap to crop-space coordinates."""

    if method == "argmax":
        row, col = argmax_2d(heatmap)
    elif method == "softargmax":
        row, col = softargmax_2d(heatmap)
    else:
        raise ValueError(f"Unknown heatmap decode method: {method}")

    return (
        heatmap_index_to_crop_pixel(col, heatmap_size=heatmap_size),
        heatmap_index_to_crop_pixel(row, heatmap_size=heatmap_size),
    )


def _decode_predictions(
    samples: list[TinySample],
    predictions: list[np.ndarray],
    *,
    heatmap_size: int,
) -> list[dict[str, Any]]:
    """Decode model outputs into a rich per-sample metrics table."""

    decoded: list[dict[str, Any]] = []
    center_batch = np.asarray(predictions[0], dtype=np.float32)
    tip_batch = np.asarray(predictions[1], dtype=np.float32)
    confidence_batch = np.asarray(predictions[2], dtype=np.float32)

    for index, sample in enumerate(samples):
        center_heatmap = np.squeeze(np.asarray(center_batch[index], dtype=np.float32))
        tip_heatmap = np.squeeze(np.asarray(tip_batch[index], dtype=np.float32))
        confidence = float(np.ravel(confidence_batch[index])[0])

        center_argmax_x, center_argmax_y = _heatmap_point(center_heatmap, heatmap_size=heatmap_size, method="argmax")
        tip_argmax_x, tip_argmax_y = _heatmap_point(tip_heatmap, heatmap_size=heatmap_size, method="argmax")
        center_soft_x, center_soft_y = _heatmap_point(center_heatmap, heatmap_size=heatmap_size, method="softargmax")
        tip_soft_x, tip_soft_y = _heatmap_point(tip_heatmap, heatmap_size=heatmap_size, method="softargmax")

        true_center_x = sample.metadata["center_x_224"]
        true_center_y = sample.metadata["center_y_224"]
        true_tip_x = sample.metadata["tip_x_224"]
        true_tip_y = sample.metadata["tip_y_224"]

        predicted_angle_soft = angle_degrees_from_center_to_tip(
            center_soft_x,
            center_soft_y,
            tip_soft_x,
            tip_soft_y,
        )
        predicted_angle_argmax = angle_degrees_from_center_to_tip(
            center_argmax_x,
            center_argmax_y,
            tip_argmax_x,
            tip_argmax_y,
        )
        true_angle = angle_degrees_from_center_to_tip(true_center_x, true_center_y, true_tip_x, true_tip_y)

        predicted_temp_soft = celsius_from_inner_dial_angle_degrees(predicted_angle_soft)
        predicted_temp_argmax = celsius_from_inner_dial_angle_degrees(predicted_angle_argmax)

        decoded.append(
            {
                "image_path": sample.image_path,
                "split": sample.split,
                "temperature_c": sample.temperature_c,
                "predicted_temperature_c": predicted_temp_soft,
                "predicted_temperature_c_argmax": predicted_temp_argmax,
                "absolute_error_c": abs(predicted_temp_soft - sample.temperature_c),
                "absolute_error_c_argmax": abs(predicted_temp_argmax - sample.temperature_c),
                "true_angle_degrees": true_angle,
                "predicted_angle_degrees": predicted_angle_soft,
                "predicted_angle_degrees_argmax": predicted_angle_argmax,
                "angle_error_degrees": abs(circular_angle_error_degrees(predicted_angle_soft, true_angle)),
                "angle_error_degrees_argmax": abs(circular_angle_error_degrees(predicted_angle_argmax, true_angle)),
                "confidence": confidence,
                "center_heatmap_max": float(np.max(center_heatmap)),
                "tip_heatmap_max": float(np.max(tip_heatmap)),
                "center_heatmap_std": float(np.std(center_heatmap)),
                "tip_heatmap_std": float(np.std(tip_heatmap)),
                "center_argmax_error": math.hypot(center_argmax_x - true_center_x, center_argmax_y - true_center_y),
                "tip_argmax_error": math.hypot(tip_argmax_x - true_tip_x, tip_argmax_y - true_tip_y),
                "center_softargmax_error": math.hypot(center_soft_x - true_center_x, center_soft_y - true_center_y),
                "tip_softargmax_error": math.hypot(tip_soft_x - true_tip_x, tip_soft_y - true_tip_y),
                "pred_center_x_224_argmax": center_argmax_x,
                "pred_center_y_224_argmax": center_argmax_y,
                "pred_tip_x_224_argmax": tip_argmax_x,
                "pred_tip_y_224_argmax": tip_argmax_y,
                "pred_center_x_224_softargmax": center_soft_x,
                "pred_center_y_224_softargmax": center_soft_y,
                "pred_tip_x_224_softargmax": tip_soft_x,
                "pred_tip_y_224_softargmax": tip_soft_y,
                "true_center_x_224": true_center_x,
                "true_center_y_224": true_center_y,
                "true_tip_x_224": true_tip_x,
                "true_tip_y_224": true_tip_y,
                "center_heatmap": center_heatmap,
                "tip_heatmap": tip_heatmap,
                "crop_image": sample.crop_image,
            }
        )

    return decoded


def _summarize_predictions(decoded_samples: list[dict[str, Any]]) -> dict[str, float]:
    """Reduce the per-sample metrics to a compact summary table."""

    center_mae = float(np.mean([sample["center_softargmax_error"] for sample in decoded_samples]))
    tip_mae = float(np.mean([sample["tip_softargmax_error"] for sample in decoded_samples]))
    center_argmax_mae = float(np.mean([sample["center_argmax_error"] for sample in decoded_samples]))
    tip_argmax_mae = float(np.mean([sample["tip_argmax_error"] for sample in decoded_samples]))
    temp_mae = float(np.mean([sample["absolute_error_c"] for sample in decoded_samples]))
    temp_mae_argmax = float(np.mean([sample["absolute_error_c_argmax"] for sample in decoded_samples]))
    angle_mae = float(np.mean([sample["angle_error_degrees"] for sample in decoded_samples]))
    angle_mae_argmax = float(np.mean([sample["angle_error_degrees_argmax"] for sample in decoded_samples]))
    center_peak = float(np.mean([sample["center_heatmap_max"] for sample in decoded_samples]))
    tip_peak = float(np.mean([sample["tip_heatmap_max"] for sample in decoded_samples]))
    center_peak_std = float(np.std([sample["center_heatmap_max"] for sample in decoded_samples]))
    tip_peak_std = float(np.std([sample["tip_heatmap_max"] for sample in decoded_samples]))

    return {
        "center_mae_px": center_mae,
        "tip_mae_px": tip_mae,
        "center_argmax_mae_px": center_argmax_mae,
        "tip_argmax_mae_px": tip_argmax_mae,
        "temperature_mae_c": temp_mae,
        "temperature_mae_c_argmax": temp_mae_argmax,
        "angle_mae_deg": angle_mae,
        "angle_mae_deg_argmax": angle_mae_argmax,
        "center_peak": center_peak,
        "tip_peak": tip_peak,
        "center_peak_std": center_peak_std,
        "tip_peak_std": tip_peak_std,
        "under_2c_pct": float(np.mean([sample["absolute_error_c"] < 2.0 for sample in decoded_samples]) * 100.0),
        "under_5c_pct": float(np.mean([sample["absolute_error_c"] < 5.0 for sample in decoded_samples]) * 100.0),
        "under_10c_pct": float(np.mean([sample["absolute_error_c"] < 10.0 for sample in decoded_samples]) * 100.0),
        "under_2c_pct_argmax": float(np.mean([sample["absolute_error_c_argmax"] < 2.0 for sample in decoded_samples]) * 100.0),
        "under_5c_pct_argmax": float(np.mean([sample["absolute_error_c_argmax"] < 5.0 for sample in decoded_samples]) * 100.0),
        "under_10c_pct_argmax": float(np.mean([sample["absolute_error_c_argmax"] < 10.0 for sample in decoded_samples]) * 100.0),
    }


def _center_prior_ablation(
    samples: list[TinySample],
    decoded_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare several center priors while keeping the model's tip prediction."""

    average_center_x = float(np.mean([sample.metadata["center_x_224"] for sample in samples]))
    average_center_y = float(np.mean([sample.metadata["center_y_224"] for sample in samples]))
    crop_center_x = 112.0
    crop_center_y = 112.0

    rows: list[dict[str, float]] = []
    for sample, decoded in zip(samples, decoded_samples):
        predicted_tip_x = decoded["pred_tip_x_224_softargmax"]
        predicted_tip_y = decoded["pred_tip_y_224_softargmax"]
        true_temp = sample.temperature_c

        mode_points = {
            "A": (decoded["pred_center_x_224_softargmax"], decoded["pred_center_y_224_softargmax"]),
            "B": (sample.metadata["center_x_224"], sample.metadata["center_y_224"]),
            "C": (average_center_x, average_center_y),
            "D": (crop_center_x, crop_center_y),
        }

        row: dict[str, float] = {"true_temp_c": true_temp}
        for mode, (center_x, center_y) in mode_points.items():
            temp_c = celsius_from_inner_dial_angle_degrees(
                angle_degrees_from_center_to_tip(center_x, center_y, predicted_tip_x, predicted_tip_y)
            )
            row[f"temp_{mode}_c"] = temp_c
            row[f"abs_error_{mode}_c"] = abs(temp_c - true_temp)

        rows.append(row)

    summary = {
        "average_center_x_224": average_center_x,
        "average_center_y_224": average_center_y,
        "crop_center_x_224": crop_center_x,
        "crop_center_y_224": crop_center_y,
        "mode_a_temp_mae_c": float(np.mean([row["abs_error_A_c"] for row in rows])),
        "mode_b_temp_mae_c": float(np.mean([row["abs_error_B_c"] for row in rows])),
        "mode_c_temp_mae_c": float(np.mean([row["abs_error_C_c"] for row in rows])),
        "mode_d_temp_mae_c": float(np.mean([row["abs_error_D_c"] for row in rows])),
        "rows": rows,
    }
    return summary


def _plot_heatmap_panel(
    ax: plt.Axes,
    heatmap: np.ndarray,
    *,
    expected_x: float,
    expected_y: float,
    title: str,
    heatmap_size: int,
) -> None:
    """Render a heatmap with expected, argmax, and softargmax markers."""

    row_argmax, col_argmax = argmax_2d(heatmap)
    row_soft, col_soft = softargmax_2d(heatmap)
    ax.imshow(heatmap, cmap="magma", origin="upper")
    ax.scatter([expected_x], [expected_y], c="white", s=45, marker="o", edgecolors="black")
    ax.scatter([col_argmax], [row_argmax], c="cyan", s=55, marker="x", linewidths=2.0)
    ax.scatter([col_soft], [row_soft], c="yellow", s=55, marker="d", edgecolors="black")
    ax.set_xlim(-0.5, heatmap_size - 0.5)
    ax.set_ylim(heatmap_size - 0.5, -0.5)
    ax.set_xlabel("x / col")
    ax.set_ylabel("y / row")
    ax.set_title(f"{title}\nmax={float(np.max(heatmap)):.4f}")


def _save_debug_overlay(
    sample: TinySample,
    decoded: dict[str, Any],
    *,
    output_path: Path,
    heatmap_size: int,
) -> None:
    """Save a compact per-sample debug figure."""

    fig = plt.figure(figsize=(17, 10), dpi=150)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 1.0), height_ratios=(1.0, 1.0))

    ax_crop = fig.add_subplot(grid[:, 0])
    ax_center_target = fig.add_subplot(grid[0, 1])
    ax_center_pred = fig.add_subplot(grid[0, 2])
    ax_tip_target = fig.add_subplot(grid[1, 1])
    ax_tip_pred = fig.add_subplot(grid[1, 2])

    ax_crop.imshow(sample.crop_image)
    ax_crop.scatter(
        [sample.metadata["center_x_224"], decoded["pred_center_x_224_softargmax"]],
        [sample.metadata["center_y_224"], decoded["pred_center_y_224_softargmax"]],
        c=["lime", "cyan"],
        s=70,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        label="center",
    )
    ax_crop.scatter(
        [sample.metadata["tip_x_224"], decoded["pred_tip_x_224_softargmax"]],
        [sample.metadata["tip_y_224"], decoded["pred_tip_y_224_softargmax"]],
        c=["red", "yellow"],
        s=70,
        marker="x",
        linewidths=2.0,
        label="tip",
    )
    ax_crop.plot(
        [sample.metadata["center_x_224"], sample.metadata["tip_x_224"]],
        [sample.metadata["center_y_224"], sample.metadata["tip_y_224"]],
        color="white",
        linewidth=2.0,
        alpha=0.8,
        label="true needle",
    )
    ax_crop.plot(
        [decoded["pred_center_x_224_softargmax"], decoded["pred_tip_x_224_softargmax"]],
        [decoded["pred_center_y_224_softargmax"], decoded["pred_tip_y_224_softargmax"]],
        color="deepskyblue",
        linewidth=2.0,
        alpha=0.8,
        label="pred needle",
    )
    ax_crop.set_title("Crop overlay")
    ax_crop.set_axis_off()
    ax_crop.legend(loc="lower right", fontsize=8, framealpha=0.9)

    _plot_heatmap_panel(
        ax_center_target,
        sample.center_heatmap,
        expected_x=sample.metadata["center_x_norm"] * (heatmap_size - 1),
        expected_y=sample.metadata["center_y_norm"] * (heatmap_size - 1),
        title="Target center heatmap",
        heatmap_size=heatmap_size,
    )
    _plot_heatmap_panel(
        ax_center_pred,
        decoded["center_heatmap"],
        expected_x=sample.metadata["center_x_norm"] * (heatmap_size - 1),
        expected_y=sample.metadata["center_y_norm"] * (heatmap_size - 1),
        title="Predicted center heatmap",
        heatmap_size=heatmap_size,
    )
    _plot_heatmap_panel(
        ax_tip_target,
        sample.tip_heatmap,
        expected_x=sample.metadata["tip_x_norm"] * (heatmap_size - 1),
        expected_y=sample.metadata["tip_y_norm"] * (heatmap_size - 1),
        title="Target tip heatmap",
        heatmap_size=heatmap_size,
    )
    _plot_heatmap_panel(
        ax_tip_pred,
        decoded["tip_heatmap"],
        expected_x=sample.metadata["tip_x_norm"] * (heatmap_size - 1),
        expected_y=sample.metadata["tip_y_norm"] * (heatmap_size - 1),
        title="Predicted tip heatmap",
        heatmap_size=heatmap_size,
    )

    summary = [
        f"true temp: {sample.temperature_c:.2f} C",
        f"pred temp: {decoded['predicted_temperature_c']:.2f} C",
        f"abs temp err: {decoded['absolute_error_c']:.2f} C",
        f"center err: {decoded['center_argmax_error']:.2f} px (argmax) / {decoded['center_softargmax_error']:.2f} px (softargmax)",
        f"tip err: {decoded['tip_argmax_error']:.2f} px (argmax) / {decoded['tip_softargmax_error']:.2f} px (softargmax)",
        f"center peak: {decoded['center_heatmap_max']:.4f}",
        f"tip peak: {decoded['tip_heatmap_max']:.4f}",
    ]
    fig.text(0.02, 0.01, "\n".join(summary), family="monospace", fontsize=10, va="bottom")
    fig.suptitle(Path(sample.image_path).name, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_predictions_csv(decoded_samples: list[dict[str, Any]], output_path: Path) -> None:
    """Persist the per-sample predictions in a compact CSV file."""

    fieldnames = [
        key
        for key in decoded_samples[0].keys()
        if key not in {"center_heatmap", "tip_heatmap", "crop_image"}
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in decoded_samples:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def _write_history_csv(history: keras.callbacks.History, output_path: Path) -> None:
    """Persist a Keras history object as CSV."""

    history_dict = history.history
    metric_names = list(history_dict.keys())
    epochs = len(history_dict.get("loss", []))
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("epoch," + ",".join(metric_names) + "\n")
        for index in range(epochs):
            values = [str(history_dict[name][index]) for name in metric_names]
            handle.write(f"{index + 1}," + ",".join(values) + "\n")


class TinyGateCallback(keras.callbacks.Callback):
    """Evaluate the training set each epoch and stop once the tiny gate passes."""

    def __init__(
        self,
        *,
        samples: list[TinySample],
        x_eval: np.ndarray,
        heatmap_size: int,
    ) -> None:
        """Store the fixed tiny-overfit evaluation set."""

        super().__init__()
        self.samples = samples
        self.x_eval = x_eval
        self.heatmap_size = heatmap_size
        self.evaluations: list[dict[str, float]] = []
        self.best_snapshot: dict[str, float] | None = None
        self.best_weights: list[np.ndarray] | None = None
        self.passed_gate = False
        self.passed_epoch: int | None = None

    def _evaluate_current_model(self) -> dict[str, float]:
        """Run a full decode pass on the tiny set."""

        predictions = self.model.predict(self.x_eval, verbose=0)
        decoded = _decode_predictions(self.samples, predictions, heatmap_size=self.heatmap_size)
        summary = _summarize_predictions(decoded)
        summary["predictions"] = decoded
        return summary

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Track the best tiny-set fit and stop when the explicit gate passes."""

        del logs
        summary = self._evaluate_current_model()
        summary["epoch"] = float(epoch + 1)
        self.evaluations.append(summary)

        score = (
            summary["center_mae_px"]
            + summary["tip_mae_px"]
            + summary["temperature_mae_c"]
            + 0.1 * summary["angle_mae_deg"]
        )
        if self.best_snapshot is None or score < self.best_snapshot["score"]:
            self.best_snapshot = {
                "epoch": float(epoch + 1),
                "score": score,
                **summary,
            }
            self.best_weights = self.model.get_weights()

        passed = (
            summary["center_mae_px"] < 3.0
            and summary["tip_mae_px"] < 5.0
            and summary["temperature_mae_c"] < 3.0
            and summary["center_peak"] > 0.5
            and summary["tip_peak"] > 0.5
        )
        print(
            "[GATE] epoch="
            f"{epoch + 1} center_mae={summary['center_mae_px']:.3f} tip_mae={summary['tip_mae_px']:.3f} "
            f"temp_mae={summary['temperature_mae_c']:.3f} angle_mae={summary['angle_mae_deg']:.3f} "
            f"center_peak={summary['center_peak']:.4f} tip_peak={summary['tip_peak']:.4f} "
            f"passed={'yes' if passed else 'no'}"
        )
        if passed:
            self.passed_gate = True
            self.passed_epoch = epoch + 1
            self.model.stop_training = True

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        """Restore the best seen weights so the report reflects the best fit."""

        del logs
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def _train_mode(
    *,
    samples: list[TinySample],
    x_train: np.ndarray,
    y_train: dict[str, np.ndarray],
    heatmap_size: int,
    learning_rate: float,
    center_heatmap_loss_weight: float,
    tip_heatmap_loss_weight: float,
    confidence_loss_weight: float,
    backbone_frozen: bool,
    epochs: int,
    batch_size: int,
    output_dir: Path,
    mode_name: str,
) -> tuple[keras.Model, keras.callbacks.History, TinyGateCallback, list[dict[str, Any]]]:
    """Train one mode and return the fitted model plus its diagnostics."""

    model = build_mobilenetv2_geometry_heatmap_v1(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=backbone_frozen,
        heatmap_size=heatmap_size,
        learning_rate=learning_rate,
    )
    _compile_model(
        model,
        learning_rate=learning_rate,
        center_heatmap_loss_weight=center_heatmap_loss_weight,
        tip_heatmap_loss_weight=tip_heatmap_loss_weight,
        confidence_loss_weight=confidence_loss_weight,
    )

    gate_callback = TinyGateCallback(samples=samples, x_eval=x_train, heatmap_size=heatmap_size)
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.CSVLogger(str(output_dir / f"{mode_name}_fit_log.csv")),
        gate_callback,
        keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    return model, history, gate_callback, gate_callback.evaluations


def _save_mode_artifacts(
    *,
    output_dir: Path,
    mode_name: str,
    model: keras.Model,
    history: keras.callbacks.History,
    evaluations: list[dict[str, float]],
    decoded_samples: list[dict[str, Any]],
) -> None:
    """Persist the artifacts for one training mode."""

    mode_dir = output_dir / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    model.save(mode_dir / "model.keras")
    _write_history_csv(history, mode_dir / "history.csv")
    _write_predictions_csv(decoded_samples, mode_dir / "predictions.csv")

    with open(mode_dir / "evaluations.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "center_mae_px",
                "tip_mae_px",
                "temperature_mae_c",
                "angle_mae_deg",
                "center_peak",
                "tip_peak",
                "passed_gate",
            ]
        )
        for row in evaluations:
            writer.writerow(
                [
                    int(row["epoch"]),
                    row["center_mae_px"],
                    row["tip_mae_px"],
                    row["temperature_mae_c"],
                    row["angle_mae_deg"],
                    row["center_peak"],
                    row["tip_peak"],
                    bool(
                        row["center_mae_px"] < 3.0
                        and row["tip_mae_px"] < 5.0
                        and row["temperature_mae_c"] < 3.0
                        and row["center_peak"] > 0.5
                        and row["tip_peak"] > 0.5
                    ),
                ]
            )


def _copy_final_artifacts(
    *,
    output_dir: Path,
    model: keras.Model,
    history: keras.callbacks.History,
    decoded_samples: list[dict[str, Any]],
) -> None:
    """Write the final selected mode artifacts to the top level."""

    model.save(output_dir / "model.keras")
    _write_history_csv(history, output_dir / "history.csv")
    _write_predictions_csv(decoded_samples, output_dir / "predictions.csv")


def _save_debug_overlays(
    *,
    debug_dir: Path,
    samples: list[TinySample],
    decoded_samples: list[dict[str, Any]],
    heatmap_size: int,
) -> None:
    """Write one overlay per sample for the final mode."""

    debug_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample, decoded) in enumerate(zip(samples, decoded_samples)):
        overlay_path = debug_dir / f"{index:03d}_{Path(sample.image_path).stem}.png"
        _save_debug_overlay(sample, decoded, output_path=overlay_path, heatmap_size=heatmap_size)


def _write_report(
    *,
    report_path: Path,
    samples: list[TinySample],
    frozen_summary: dict[str, float],
    frozen_passed: bool,
    frozen_epoch: int | None,
    unfrozen_summary: dict[str, float] | None,
    unfrozen_passed: bool | None,
    unfrozen_epoch: int | None,
    ablation: dict[str, Any],
    final_mode_name: str,
    final_summary: dict[str, float],
    decoded_samples: list[dict[str, Any]],
    center_heatmap_loss_weight: float,
    tip_heatmap_loss_weight: float,
    confidence_loss_weight: float,
    sigma_pixels: float,
    backbone_frozen: bool,
) -> None:
    """Write the final tiny-overfit report with the ablation results."""

    unfreeze_used = unfrozen_summary is not None
    under_2c = final_summary["under_2c_pct"]
    under_5c = final_summary["under_5c_pct"]
    under_10c = final_summary["under_10c_pct"]

    worst_3 = sorted(decoded_samples, key=lambda row: row["absolute_error_c"], reverse=True)[:3]

    lines = [
        "# Geometry Heatmap Tiny Overfit v2",
        "",
        "## Run Summary",
        "",
        f"- Samples: {len(samples)}",
        f"- Sigma: {sigma_pixels:.2f} pixels at 56x56",
        f"- Center heatmap loss weight: {center_heatmap_loss_weight:.1f}",
        f"- Tip heatmap loss weight: {tip_heatmap_loss_weight:.1f}",
        f"- Confidence loss weight: {confidence_loss_weight:.1f}",
        f"- Final mode: {final_mode_name}",
        f"- Frozen backbone passed: {'yes' if frozen_passed else 'no'}",
        f"- Unfreezing needed: {'yes' if unfreeze_used else 'no'}",
        "",
        "## Final Mode Metrics",
        "",
        f"- Center MAE: {final_summary['center_mae_px']:.3f} px",
        f"- Tip MAE: {final_summary['tip_mae_px']:.3f} px",
        f"- Angle MAE: {final_summary['angle_mae_deg']:.3f} deg",
        f"- Temperature MAE: {final_summary['temperature_mae_c']:.3f} C",
        f"- Mean center peak: {final_summary['center_peak']:.4f}",
        f"- Mean tip peak: {final_summary['tip_peak']:.4f}",
        f"- Under 2 C: {under_2c:.1f}%",
        f"- Under 5 C: {under_5c:.1f}%",
        f"- Under 10 C: {under_10c:.1f}%",
        f"- Argmax center MAE: {final_summary['center_argmax_mae_px']:.3f} px",
        f"- Argmax tip MAE: {final_summary['tip_argmax_mae_px']:.3f} px",
        f"- Argmax temperature MAE: {final_summary['temperature_mae_c_argmax']:.3f} C",
        f"- Argmax angle MAE: {final_summary['angle_mae_deg_argmax']:.3f} deg",
        "",
        "## Frozen Backbone Check",
        "",
        f"- Frozen pass: {'yes' if frozen_passed else 'no'}",
        f"- Frozen epoch: {frozen_epoch if frozen_epoch is not None else 'n/a'}",
        f"- Frozen center MAE: {frozen_summary['center_mae_px']:.3f} px",
        f"- Frozen tip MAE: {frozen_summary['tip_mae_px']:.3f} px",
        f"- Frozen temperature MAE: {frozen_summary['temperature_mae_c']:.3f} C",
        f"- Frozen mean center peak: {frozen_summary['center_peak']:.4f}",
        f"- Frozen mean tip peak: {frozen_summary['tip_peak']:.4f}",
        "",
    ]

    if unfrozen_summary is not None:
        lines.extend(
            [
                "## Unfrozen Fine-Tune Check",
                "",
                f"- Unfrozen pass: {'yes' if unfrozen_passed else 'no'}",
                f"- Unfrozen epoch: {unfrozen_epoch if unfrozen_epoch is not None else 'n/a'}",
                f"- Unfrozen center MAE: {unfrozen_summary['center_mae_px']:.3f} px",
                f"- Unfrozen tip MAE: {unfrozen_summary['tip_mae_px']:.3f} px",
                f"- Unfrozen temperature MAE: {unfrozen_summary['temperature_mae_c']:.3f} C",
                f"- Unfrozen mean center peak: {unfrozen_summary['center_peak']:.4f}",
                f"- Unfrozen mean tip peak: {unfrozen_summary['tip_peak']:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Center-Prior Ablation",
            "",
            "| Mode | Center Source | Tip Source | Temp MAE (C) |",
            "| --- | --- | --- | ---: |",
            f"| A | model-predicted center | model-predicted tip | {ablation['mode_a_temp_mae_c']:.3f} |",
            f"| B | true / manifest center | model-predicted tip | {ablation['mode_b_temp_mae_c']:.3f} |",
            f"| C | average train-set center | model-predicted tip | {ablation['mode_c_temp_mae_c']:.3f} |",
            f"| D | loose-crop geometric center | model-predicted tip | {ablation['mode_d_temp_mae_c']:.3f} |",
            "",
            f"- Average train-set center: ({ablation['average_center_x_224']:.2f}, {ablation['average_center_y_224']:.2f})",
            f"- Loose-crop geometric center: ({ablation['crop_center_x_224']:.2f}, {ablation['crop_center_y_224']:.2f})",
            "",
            "## Worst 3 Predictions",
            "",
            "| Image | Abs Error (C) | Center Err (px) | Tip Err (px) | Peak Center | Peak Tip |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in worst_3:
        lines.append(
            f"| {Path(row['image_path']).name} | {row['absolute_error_c']:.3f} | "
            f"{row['center_softargmax_error']:.3f} | {row['tip_softargmax_error']:.3f} | "
            f"{row['center_heatmap_max']:.4f} | {row['tip_heatmap_max']:.4f} |"
        )

    passed_final = (
        final_summary["center_mae_px"] < 3.0
        and final_summary["tip_mae_px"] < 5.0
        and final_summary["temperature_mae_c"] < 3.0
        and final_summary["center_peak"] > 0.5
        and final_summary["tip_peak"] > 0.5
    )

    recommendation = (
        "Proceed to a full heatmap_v2 training run with the same weighted objective and the final mode settings."
        if passed_final
        else "Do not proceed to full heatmap_v2 yet; the tiny-overfit gate is still not clean enough."
    )
    if ablation["mode_b_temp_mae_c"] < ablation["mode_a_temp_mae_c"] or ablation["mode_c_temp_mae_c"] < ablation["mode_a_temp_mae_c"] or ablation["mode_d_temp_mae_c"] < ablation["mode_a_temp_mae_c"]:
        recommendation = (
            "The center-prior ablation is better than the full center+tip prediction, so a simpler tip-only or fixed-center architecture is worth testing before scaling up."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Tiny-overfit gate passed: {'yes' if passed_final else 'no'}",
            f"- Backbone unfreezing used: {'yes' if unfreeze_used else 'no'}",
            f"- Recommendation: {recommendation}",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Train the tiny overfit gate and stop before any full v2 run."""

    parser = argparse.ArgumentParser(description="Tiny overfit gate for geometry heatmaps")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Clean geometry manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_tiny_overfit_v2"),
        help="Training artifact directory.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("ml/debug/geometry_heatmap_tiny_overfit_v2/overfit"),
        help="Debug overlay directory.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_tiny_overfit_v2.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Number of training samples to use.")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum epochs per stage.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the tiny overfit.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Frozen-stage learning rate.")
    parser.add_argument("--unfreeze-learning-rate", type=float, default=1e-5, help="Fine-tuning learning rate.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap size.")
    parser.add_argument("--sigma-pixels", type=float, default=5.0, help="Gaussian sigma in heatmap pixels.")
    parser.add_argument("--center-heatmap-loss-weight", type=float, default=3.0, help="Center heatmap loss weight.")
    parser.add_argument("--tip-heatmap-loss-weight", type=float, default=1.0, help="Tip heatmap loss weight.")
    parser.add_argument("--confidence-loss-weight", type=float, default=0.0, help="Confidence loss weight.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    debug_dir = base_path / args.debug_dir if not args.debug_dir.is_absolute() else args.debug_dir
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    samples = _load_tiny_samples(
        manifest_path=manifest_path,
        base_path=base_path,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        limit=args.limit,
    )
    x_train, y_train = _make_training_arrays(samples)

    # Stage 1: frozen backbone.
    frozen_model = build_mobilenetv2_geometry_heatmap_v1(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=True,
        heatmap_size=args.heatmap_size,
        learning_rate=args.learning_rate,
    )
    _compile_model(
        frozen_model,
        learning_rate=args.learning_rate,
        center_heatmap_loss_weight=args.center_heatmap_loss_weight,
        tip_heatmap_loss_weight=args.tip_heatmap_loss_weight,
        confidence_loss_weight=args.confidence_loss_weight,
    )
    frozen_gate = TinyGateCallback(samples=samples, x_eval=x_train, heatmap_size=args.heatmap_size)
    frozen_history = frozen_model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[
            keras.callbacks.CSVLogger(str(output_dir / "frozen_fit_log.csv")),
            frozen_gate,
            keras.callbacks.TerminateOnNaN(),
        ],
        verbose=2,
    )
    frozen_decoded = _decode_predictions(samples, frozen_model.predict(x_train, verbose=0), heatmap_size=args.heatmap_size)
    frozen_summary = _summarize_predictions(frozen_decoded)
    _save_mode_artifacts(
        output_dir=output_dir,
        mode_name="frozen",
        model=frozen_model,
        history=frozen_history,
        evaluations=frozen_gate.evaluations,
        decoded_samples=frozen_decoded,
    )

    unfrozen_summary: dict[str, float] | None = None
    unfrozen_gate_passed: bool | None = None
    unfrozen_gate_epoch: int | None = None
    final_model = frozen_model
    final_history = frozen_history
    final_decoded = frozen_decoded
    final_mode_name = "frozen"

    if not frozen_gate.passed_gate:
        # Stage 2: unfreeze the last MobileNetV2 block only, then fine-tune with a tiny LR.
        _set_backbone_trainability(frozen_model, trainable_last_block=True)
        _compile_model(
            frozen_model,
            learning_rate=args.unfreeze_learning_rate,
            center_heatmap_loss_weight=args.center_heatmap_loss_weight,
            tip_heatmap_loss_weight=args.tip_heatmap_loss_weight,
            confidence_loss_weight=args.confidence_loss_weight,
        )
        unfrozen_gate = TinyGateCallback(samples=samples, x_eval=x_train, heatmap_size=args.heatmap_size)
        unfrozen_history = frozen_model.fit(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=[
                keras.callbacks.CSVLogger(str(output_dir / "unfrozen_fit_log.csv")),
                unfrozen_gate,
                keras.callbacks.TerminateOnNaN(),
            ],
            verbose=2,
        )
        unfrozen_decoded = _decode_predictions(samples, frozen_model.predict(x_train, verbose=0), heatmap_size=args.heatmap_size)
        unfrozen_summary = _summarize_predictions(unfrozen_decoded)
        _save_mode_artifacts(
            output_dir=output_dir,
            mode_name="unfrozen",
            model=frozen_model,
            history=unfrozen_history,
            evaluations=unfrozen_gate.evaluations,
            decoded_samples=unfrozen_decoded,
        )

        final_model = frozen_model
        final_history = unfrozen_history
        final_decoded = unfrozen_decoded
        final_mode_name = "unfrozen"
        unfrozen_gate_passed = unfrozen_gate.passed_gate
        unfrozen_gate_epoch = unfrozen_gate.passed_epoch

    _copy_final_artifacts(
        output_dir=output_dir,
        model=final_model,
        history=final_history,
        decoded_samples=final_decoded,
    )

    _save_debug_overlays(
        debug_dir=debug_dir,
        samples=samples,
        decoded_samples=final_decoded,
        heatmap_size=args.heatmap_size,
    )

    ablation = _center_prior_ablation(samples, final_decoded)
    final_summary = _summarize_predictions(final_decoded)
    _write_report(
        report_path=report_path,
        samples=samples,
        frozen_summary=frozen_summary,
        frozen_passed=frozen_gate.passed_gate,
        frozen_epoch=frozen_gate.passed_epoch,
        unfrozen_summary=unfrozen_summary,
        unfrozen_passed=unfrozen_gate_passed,
        unfrozen_epoch=unfrozen_gate_epoch,
        ablation=ablation,
        final_mode_name=final_mode_name,
        final_summary=final_summary,
        decoded_samples=final_decoded,
        center_heatmap_loss_weight=args.center_heatmap_loss_weight,
        tip_heatmap_loss_weight=args.tip_heatmap_loss_weight,
        confidence_loss_weight=args.confidence_loss_weight,
        sigma_pixels=args.sigma_pixels,
        backbone_frozen=True,
    )

    print(f"Artifacts written to {output_dir}")
    print(f"Debug overlays written to {debug_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
