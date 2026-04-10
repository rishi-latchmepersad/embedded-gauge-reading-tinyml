"""Fit and save a calibration layer on top of a scalar gauge regressor."""

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


@dataclass(frozen=True)
class EvalItem:
    """One image and its scalar gauge value label."""

    image_path: Path
    value: float


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the calibration job."""
    parser = argparse.ArgumentParser(
        description="Calibrate a scalar gauge regressor with an affine output layer."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model.keras.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV manifest with image_path,value rows used for calibration.",
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
        default="affine",
        help="Calibration family to fit on top of the base scalar regressor.",
    )
    parser.add_argument(
        "--spline-knot-mode",
        type=str,
        choices=["all", "interior", "quantile"],
        default="all",
        help="Which prediction points to use as spline knots when --mode spline is selected.",
    )
    parser.add_argument(
        "--spline-knot-count",
        type=int,
        default=8,
        help="How many quantile knots to use when --spline-knot-mode quantile is selected.",
    )
    return parser.parse_args()


def _load_model(model_path: Path, *, legacy_preprocess: bool) -> keras.Model:
    """Load a saved Keras model, optionally providing the legacy preprocess symbol."""
    print(f"[CAL] Loading model from {model_path}...", flush=True)
    # Keep the legacy preprocess symbol available even when newer models no longer need it.
    custom_objects: dict[str, Any] = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
    }
    if legacy_preprocess:
        print("[CAL] Legacy MobileNetV2 preprocess support enabled.", flush=True)
    # Calibration only needs the forward graph, so skip compile-time deserialization.
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )
    print(f"[CAL] Model loaded: {model.name}", flush=True)
    return model


def _load_manifest(manifest_path: Path) -> list[EvalItem]:
    """Load image/value pairs from a CSV manifest."""
    items: list[EvalItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = REPO_ROOT / image_path
            items.append(EvalItem(image_path=image_path, value=float(row["value"])))
    print(f"[CAL] Loaded {len(items)} calibration items from {manifest_path}.", flush=True)
    return items


def _infer_image_size(model: keras.Model) -> tuple[int, int]:
    """Infer the expected input image size from the model."""
    input_shape = model.input_shape
    if not isinstance(input_shape, tuple) or len(input_shape) < 3:
        raise ValueError(f"Unexpected model input shape: {input_shape!r}")

    height = int(input_shape[1]) if input_shape[1] is not None else 224
    width = int(input_shape[2]) if input_shape[2] is not None else 224
    return height, width


def _predict_values(
    model: keras.Model,
    items: list[EvalItem],
    *,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on each calibration image and collect predictions and labels."""
    predictions: list[float] = []
    labels: list[float] = []

    for item in items:
        image = tf.keras.utils.load_img(item.image_path, target_size=(image_height, image_width))
        image_array = tf.keras.utils.img_to_array(image).astype(np.float32)
        batch = tf.convert_to_tensor(image_array[None, ...] / 255.0, dtype=tf.float32)
        pred_value = float(model.predict(batch, verbose=0)[0][0])
        predictions.append(pred_value)
        labels.append(float(item.value))
        print(
            f"[CAL] {item.image_path.name}: true={item.value:.4f} pred={pred_value:.4f}",
            flush=True,
        )

    return np.asarray(predictions, dtype=np.float32), np.asarray(labels, dtype=np.float32)


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
    calibrated_model = keras.Model(
        inputs=model.input,
        outputs=calibrated_output,
        name=f"{model.name}_calibrated",
    )
    return calibrated_model


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
        basis_terms.append(keras.layers.ReLU(name=f"value_calibration_relu_{knot_index}")(shifted))

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


def main() -> None:
    """Fit the calibration layer and save the calibrated model artifact."""
    args = _parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.model.parent.with_name(f"{args.model.parent.name}_calibrated")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.model, legacy_preprocess=args.legacy_mobilenetv2_preprocess)
    image_height, image_width = _infer_image_size(model)
    print(f"[CAL] Model input size: {image_height}x{image_width}", flush=True)

    items = _load_manifest(args.manifest)
    predictions, labels = _predict_values(
        model,
        items,
        image_height=image_height,
        image_width=image_width,
    )

    raw_mae = float(np.mean(np.abs(predictions - labels)))
    metrics: dict[str, object] = {}
    if args.mode == "affine":
        scale, bias = _fit_affine_calibration(predictions, labels)
        calibrated_predictions = scale * predictions + bias
        calibrated_mae = float(np.mean(np.abs(calibrated_predictions - labels)))
        print(
            f"[CAL] Affine calibration fitted: scale={scale:.6f} bias={bias:.6f}",
            flush=True,
        )
        calibrated_model = _build_calibrated_model(model, scale=scale, bias=bias)
        metrics.update(
            {
                "mode": "affine",
                "scale": scale,
                "bias": bias,
            }
        )
    else:
        bias, weights, knots = _fit_piecewise_calibration(
            predictions,
            labels,
            knot_mode=args.spline_knot_mode,
            knot_count=args.spline_knot_count,
        )
        design = np.column_stack(
            [predictions] + [np.maximum(predictions - knot, 0.0) for knot in knots]
        )
        calibrated_predictions = design @ weights + bias
        calibrated_mae = float(np.mean(np.abs(calibrated_predictions - labels)))
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

    print(f"[CAL] Raw MAE: {raw_mae:.4f}", flush=True)
    print(f"[CAL] Calibrated MAE: {calibrated_mae:.4f}", flush=True)

    model_path = output_dir / "model.keras"
    calibrated_model.save(model_path)
    print(f"[CAL] Saved calibrated model to {model_path}", flush=True)

    metrics.update(
        {
        "image_height": image_height,
        "image_width": image_width,
        "num_items": int(labels.size),
        "raw_mae": raw_mae,
        "calibrated_mae": calibrated_mae,
        "manifest": str(args.manifest),
        "source_model": str(args.model),
        }
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[CAL] Wrote metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
