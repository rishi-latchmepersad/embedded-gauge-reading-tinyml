"""Fit and save a calibration layer using the training crop domain plus board crops."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import keras
import numpy as np
import tensorflow as tf

# Make the package importable when this script is run from the ml/ directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.presets import DEFAULT_GAUGE_ID
from embedded_gauge_reading_tinyml.training import (
    TrainConfig,
    _build_tf_dataset,
    _build_training_examples,
)


@dataclass(frozen=True)
class EvalItem:
    """One image path and its scalar label."""

    image_path: Path
    value: float


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the crop-domain calibration job."""
    parser = argparse.ArgumentParser(
        description="Calibrate a scalar gauge regressor using crop-space predictions."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model.keras.")
    parser.add_argument(
        "--labelled-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "labelled",
        help="Directory that contains the CVAT zip exports.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory that contains the raw labelled images.",
    )
    parser.add_argument(
        "--board-manifest",
        type=Path,
        required=True,
        help="CSV manifest of cropped board captures used as calibration anchors.",
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default=DEFAULT_GAUGE_ID,
        help="Gauge identifier used to derive training labels and crop boxes.",
    )
    parser.add_argument(
        "--crop-pad-ratio",
        type=float,
        default=0.10,
        help="Padding ratio applied around the labelled dial ellipse.",
    )
    parser.add_argument(
        "--strict-labels",
        action="store_true",
        help="Reject labels slightly outside the configured sweep.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=224,
        help="Input image height expected by the source model.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=224,
        help="Input image width expected by the source model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used while collecting crop-domain predictions.",
    )
    parser.add_argument(
        "--legacy-mobilenetv2-preprocess",
        action="store_true",
        help="Load older MobileNetV2 models that used a non-serializable preprocess Lambda.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the calibrated model and metrics.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["affine", "spline"],
        default="spline",
        help="Calibration family to fit on top of the base scalar regressor.",
    )
    parser.add_argument(
        "--spline-knot-mode",
        type=str,
        choices=["all", "interior", "quantile"],
        default="quantile",
        help="How to choose knot locations for piecewise calibration.",
    )
    parser.add_argument(
        "--spline-knot-count",
        type=int,
        default=8,
        help="How many quantile knots to use when --spline-knot-mode quantile is selected.",
    )
    return parser.parse_args()


def _load_model(model_path: Path, *, legacy_preprocess: bool) -> keras.Model:
    """Load a saved Keras model with optional legacy MobileNetV2 support."""
    print(f"[CAL] Loading model from {model_path}...", flush=True)
    custom_objects: dict[str, Any] = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
    }
    if legacy_preprocess:
        print("[CAL] Legacy MobileNetV2 preprocess support enabled.", flush=True)
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )
    print(f"[CAL] Model loaded: {model.name}", flush=True)
    return model


def _load_board_manifest(manifest_path: Path) -> list[EvalItem]:
    """Load cropped board images and their labels from a CSV manifest."""
    items: list[EvalItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        required_columns = {"image_path", "value"}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError("Manifest must include image_path and value columns.")

        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = REPO_ROOT / image_path
            items.append(EvalItem(image_path=image_path, value=float(row["value"])))

    print(f"[CAL] Loaded {len(items)} board items from {manifest_path}.", flush=True)
    return items


def _infer_image_size(model: keras.Model) -> tuple[int, int]:
    """Infer the model input size from the saved Keras graph."""
    input_shape = model.input_shape
    if not isinstance(input_shape, tuple) or len(input_shape) < 3:
        raise ValueError(f"Unexpected model input shape: {input_shape!r}")

    height = int(input_shape[1]) if input_shape[1] is not None else 224
    width = int(input_shape[2]) if input_shape[2] is not None else 224
    return height, width


def _predict_training_examples(
    model: keras.Model,
    examples: list,
    *,
    image_height: int,
    image_width: int,
    batch_size: int,
) -> np.ndarray:
    """Predict values for the training examples using the same crop pipeline as training."""
    if not examples:
        return np.asarray([], dtype=np.float32)

    config = TrainConfig(
        image_height=image_height,
        image_width=image_width,
        batch_size=batch_size,
        augment_training=False,
        edge_focus_strength=0.0,
    )
    dataset = _build_tf_dataset(examples, config, training=False)
    image_dataset = dataset.map(lambda image, target: image, num_parallel_calls=tf.data.AUTOTUNE)
    predictions = model.predict(image_dataset, verbose=0).reshape(-1)
    return np.asarray(predictions, dtype=np.float32)


def _predict_board_items(
    model: keras.Model,
    items: list[EvalItem],
    *,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Predict values for board crops that are already centered and resized externally."""
    predictions: list[float] = []
    for item in items:
        image = tf.keras.utils.load_img(
            item.image_path,
            target_size=(image_height, image_width),
        )
        image_array = tf.keras.utils.img_to_array(image).astype(np.float32)
        batch = tf.convert_to_tensor(image_array[None, ...] / 255.0, dtype=tf.float32)
        pred_value = float(model.predict(batch, verbose=0)[0][0])
        predictions.append(pred_value)
        print(
            f"[CAL] BOARD {item.image_path.name}: true={item.value:.4f} pred={pred_value:.4f}",
            flush=True,
        )

    return np.asarray(predictions, dtype=np.float32)


def _fit_affine_calibration(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Fit label ~= scale * prediction + bias with least squares."""
    if predictions.ndim != 1 or labels.ndim != 1:
        raise ValueError("Calibration inputs must be one-dimensional.")
    if predictions.size != labels.size:
        raise ValueError("Prediction and label counts must match.")

    design = np.vstack([predictions, np.ones_like(predictions)]).T
    scale, bias = np.linalg.lstsq(design, labels, rcond=None)[0]
    return float(scale), float(bias)


def _fit_piecewise_calibration(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    knot_mode: str,
    knot_count: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a fixed piecewise-linear calibration model on the scalar output."""
    if predictions.ndim != 1 or labels.ndim != 1:
        raise ValueError("Calibration inputs must be one-dimensional.")
    if predictions.size != labels.size:
        raise ValueError("Prediction and label counts must match.")

    sorted_predictions = np.sort(np.unique(predictions))
    if sorted_predictions.size < 2:
        raise ValueError("Need at least two unique predictions for spline calibration.")
    if knot_mode == "interior":
        knots = sorted_predictions[1:-1]
    elif knot_mode == "quantile":
        if knot_count < 2:
            raise ValueError("Need at least two quantile knots for spline calibration.")
        if sorted_predictions.size <= knot_count:
            knots = sorted_predictions
        else:
            quantiles = np.linspace(0.0, 1.0, knot_count + 2, dtype=np.float32)[1:-1]
            knots = np.quantile(predictions, quantiles).astype(np.float32)
            knots = np.unique(knots)
            knots = knots[(knots > sorted_predictions[0]) & (knots < sorted_predictions[-1])]
            if knots.size == 0:
                knots = sorted_predictions[1:-1]
    else:
        knots = sorted_predictions

    features = [predictions]
    for knot in knots:
        features.append(np.maximum(predictions - knot, 0.0))
    design = np.vstack(features).T
    augmented_design = np.column_stack([design, np.ones_like(predictions)])
    solution = np.linalg.lstsq(augmented_design, labels, rcond=None)[0]
    weights = solution[:-1]
    bias = solution[-1]
    return float(bias), np.asarray(weights, dtype=np.float32), np.asarray(knots, dtype=np.float32)


def _build_calibrated_model(
    model: keras.Model,
    *,
    scale: float,
    bias: float,
) -> keras.Model:
    """Wrap the scalar regressor with a fixed affine output calibration layer."""
    calibration_layer = keras.layers.Dense(
        1,
        use_bias=True,
        name="value_calibration",
        kernel_initializer=keras.initializers.Constant([[scale]]),
        bias_initializer=keras.initializers.Constant(bias),
        trainable=False,
    )
    calibrated_output = calibration_layer(model.output)
    return keras.Model(
        inputs=model.input,
        outputs=calibrated_output,
        name=f"{model.name}_calibrated",
    )


def _build_piecewise_calibrated_model(
    model: keras.Model,
    *,
    bias: float,
    weights: np.ndarray,
    knots: np.ndarray,
) -> keras.Model:
    """Wrap the scalar regressor with a fixed piecewise-linear calibration head."""
    if weights.ndim != 1:
        raise ValueError("Piecewise weights must be one-dimensional.")
    if knots.ndim != 1:
        raise ValueError("Spline knots must be one-dimensional.")
    if weights.size != knots.size + 1:
        raise ValueError("Expected one linear weight plus one weight per knot.")

    base_output = model.output
    basis_terms: list[keras.layers.Layer] = [base_output]
    for knot_index, knot in enumerate(knots):
        shifted = keras.layers.Dense(
            1,
            use_bias=True,
            kernel_initializer=keras.initializers.Constant([[1.0]]),
            bias_initializer=keras.initializers.Constant(float(-knot)),
            trainable=False,
            name=f"value_calibration_shift_{knot_index}",
        )(base_output)
        basis_terms.append(
            keras.layers.ReLU(name=f"value_calibration_relu_{knot_index}")(shifted)
        )

    features = keras.layers.Concatenate(name="value_calibration_features")(basis_terms)
    calibration_layer = keras.layers.Dense(
        1,
        use_bias=True,
        name="value_calibration",
        kernel_initializer=keras.initializers.Constant(weights.reshape((-1, 1))),
        bias_initializer=keras.initializers.Constant(float(bias)),
        trainable=False,
    )
    calibrated_output = calibration_layer(features)
    return keras.Model(
        inputs=model.input,
        outputs=calibrated_output,
        name=f"{model.name}_piecewise_calibrated",
    )


def _summarize_errors(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
) -> dict[str, float]:
    """Compute compact error metrics for one prediction set."""
    abs_errors = np.abs(predictions - labels)
    mean_abs_err = float(np.mean(abs_errors)) if abs_errors.size else 0.0
    max_abs_err = float(np.max(abs_errors)) if abs_errors.size else 0.0
    cases_over_5c = float(np.sum(abs_errors > 5.0))
    print(
        f"[CAL] {title}: mean_abs_err={mean_abs_err:.4f} "
        f"max_abs_err={max_abs_err:.4f} cases_over_5c={int(cases_over_5c)}",
        flush=True,
    )
    return {
        f"{title}_mean_abs_err": mean_abs_err,
        f"{title}_max_abs_err": max_abs_err,
        f"{title}_cases_over_5c": cases_over_5c,
    }


def main() -> None:
    """Fit the calibration layer and save the calibrated model artifact."""
    args = _parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.model.parent.with_name(f"{args.model.parent.name}_calibrated_crop")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.model, legacy_preprocess=args.legacy_mobilenetv2_preprocess)
    image_height, image_width = _infer_image_size(model)
    print(f"[CAL] Model input size: {image_height}x{image_width}", flush=True)

    gauge_specs = load_gauge_specs()
    if args.gauge_id not in gauge_specs:
        raise ValueError(f"Unknown gauge id: {args.gauge_id}")
    spec = gauge_specs[args.gauge_id]

    labelled_dir = args.labelled_dir if args.labelled_dir.is_absolute() else REPO_ROOT / args.labelled_dir
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir
    samples = load_dataset(labelled_dir=labelled_dir, raw_dir=raw_dir)
    training_examples, dropped_out_of_sweep = _build_training_examples(
        samples,
        spec,
        strict_labels=args.strict_labels,
        crop_pad_ratio=args.crop_pad_ratio,
    )
    print(
        f"[CAL] Training examples: {len(training_examples)} dropped={dropped_out_of_sweep}",
        flush=True,
    )

    board_items = _load_board_manifest(args.board_manifest)
    train_predictions = _predict_training_examples(
        model,
        training_examples,
        image_height=image_height,
        image_width=image_width,
        batch_size=args.batch_size,
    )
    train_labels = np.asarray([example.value for example in training_examples], dtype=np.float32)
    board_predictions = _predict_board_items(
        model,
        board_items,
        image_height=image_height,
        image_width=image_width,
    )
    board_labels = np.asarray([item.value for item in board_items], dtype=np.float32)

    raw_all_predictions = np.concatenate([train_predictions, board_predictions], axis=0)
    all_labels = np.concatenate([train_labels, board_labels], axis=0)

    raw_metrics = _summarize_errors(raw_all_predictions, all_labels, title="raw")
    _summarize_errors(train_predictions, train_labels, title="train_raw")
    _summarize_errors(board_predictions, board_labels, title="board_raw")

    metrics: dict[str, object] = {}
    if args.mode == "affine":
        scale, bias = _fit_affine_calibration(raw_all_predictions, all_labels)
        calibrated_all = scale * raw_all_predictions + bias
        calibrated_train = scale * train_predictions + bias
        calibrated_board = scale * board_predictions + bias
        print(
            f"[CAL] Affine calibration fitted: scale={scale:.6f} bias={bias:.6f}",
            flush=True,
        )
        calibrated_model = _build_calibrated_model(model, scale=scale, bias=bias)
        metrics.update({"mode": "affine", "scale": scale, "bias": bias})
    else:
        bias, weights, knots = _fit_piecewise_calibration(
            raw_all_predictions,
            all_labels,
            knot_mode=args.spline_knot_mode,
            knot_count=args.spline_knot_count,
        )
        design_all = np.column_stack(
            [raw_all_predictions]
            + [np.maximum(raw_all_predictions - knot, 0.0) for knot in knots]
        )
        design_train = np.column_stack(
            [train_predictions] + [np.maximum(train_predictions - knot, 0.0) for knot in knots]
        )
        design_board = np.column_stack(
            [board_predictions] + [np.maximum(board_predictions - knot, 0.0) for knot in knots]
        )
        calibrated_all = design_all @ weights + bias
        calibrated_train = design_train @ weights + bias
        calibrated_board = design_board @ weights + bias
        print(
            f"[CAL] Piecewise calibration fitted: knot_mode={args.spline_knot_mode} "
            f"bias={bias:.6f} knots={knots.size} weights={weights.size}",
            flush=True,
        )
        calibrated_model = _build_piecewise_calibrated_model(
            model,
            bias=bias,
            weights=weights,
            knots=knots,
        )
        metrics.update(
            {
                "mode": "spline",
                "spline_knot_mode": args.spline_knot_mode,
                "spline_knot_count": args.spline_knot_count,
                "bias": bias,
                "weights": weights.tolist(),
                "knots": knots.tolist(),
            }
        )

    _summarize_errors(calibrated_all, all_labels, title="calibrated")
    _summarize_errors(calibrated_train, train_labels, title="train_calibrated")
    _summarize_errors(calibrated_board, board_labels, title="board_calibrated")

    model_path = output_dir / "model.keras"
    calibrated_model.save(model_path)
    print(f"[CAL] Saved calibrated model to {model_path}", flush=True)

    metrics.update(
        {
            "image_height": image_height,
            "image_width": image_width,
            "num_training_examples": int(train_labels.size),
            "num_board_examples": int(board_labels.size),
            "raw_combined_mean_abs_err": raw_metrics["raw_mean_abs_err"],
            "raw_combined_max_abs_err": raw_metrics["raw_max_abs_err"],
            "manifest": str(args.board_manifest),
            "source_model": str(args.model),
        }
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[CAL] Wrote metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
