"""Fit deploy-time calibration for the OBB + scalar gauge cascade."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal, cast

import numpy as np
import tensorflow as tf

# Make the package importable when this script is run from the ``ml/`` directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import (  # noqa: E402
    load_rgb_image,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.geometry_cascade import (  # noqa: E402
    source_xy_from_resized_xy,
)

ObbModelKind = Literal["auto", "keras", "tflite"]


@dataclass(slots=True)
class ObbSession:
    """Hold the loaded OBB backend and its tensor details."""

    kind: Literal["keras", "tflite"]
    model: tf.keras.Model | tf.lite.Interpreter
    input_details: dict[str, Any] | None = None
    output_details: dict[str, Any] | None = None


@dataclass(frozen=True)
class ManifestItem:
    """One manifest row with an image path and scalar label."""

    image_path: Path
    value: float


@dataclass(frozen=True)
class PredictionSet:
    """Raw scalar predictions for one manifest."""

    name: str
    raw_predictions: np.ndarray
    labels: np.ndarray


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the calibration fit."""
    parser = argparse.ArgumentParser(
        description="Fit firmware calibration for the OBB + scalar cascade."
    )
    parser.add_argument(
        "--obb-model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "deployment" / "prod_model_v0.3_obb_int8" / "model_int8.tflite",
        help="Path to the OBB localizer model.",
    )
    parser.add_argument(
        "--obb-model-kind",
        choices=["auto", "keras", "tflite"],
        default="auto",
        help="OBB backend loader to use.",
    )
    parser.add_argument(
        "--scalar-model",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "deployment"
        / "scalar_full_finetune_from_best_piecewise_calibrated_int8"
        / "model_int8.tflite",
        help="Path to the quantized scalar reader.",
    )
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        action="append",
        required=True,
        help="CSV manifest used to fit the new calibration. May be passed multiple times.",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        action="append",
        required=True,
        help="CSV manifest used to validate the calibration. May be passed multiple times.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square input size for both the OBB localizer and the scalar reader.",
    )
    parser.add_argument(
        "--obb-crop-scale",
        type=float,
        default=1.20,
        help="Scale factor applied to the OBB-derived crop before scalar inference.",
    )
    parser.add_argument(
        "--min-crop-size",
        type=float,
        default=48.0,
        help="Minimum edge length for the scalar crop after OBB expansion.",
    )
    parser.add_argument(
        "--spline-knot-mode",
        choices=["interior", "quantile", "all"],
        default="quantile",
        help="How to choose knot locations for the piecewise calibration fit.",
    )
    parser.add_argument(
        "--spline-knot-count",
        type=int,
        default=6,
        help="How many quantile knots to use when fitting the piecewise model.",
    )
    parser.add_argument(
        "--reference-scale",
        type=float,
        default=1.1953182220458984,
        help="Current firmware affine calibration scale used as the reference baseline.",
    )
    parser.add_argument(
        "--reference-bias",
        type=float,
        default=-1.0408254861831665,
        help="Current firmware affine calibration bias used as the reference baseline.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "calibration" / "prodv0_3_obb_scalar_calibration.json",
        help="Where to write the fitted calibration payload.",
    )
    return parser.parse_args()


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest image path relative to the repo root when needed."""
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return REPO_ROOT / image_path


def _load_manifest(manifest_path: Path) -> list[ManifestItem]:
    """Load labeled image paths and scalar values from a CSV manifest."""
    items: list[ManifestItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        required_columns = {"image_path", "value"}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError("Manifest must include image_path and value columns.")

        for row in reader:
            items.append(
                ManifestItem(
                    image_path=_resolve_image_path(row["image_path"]),
                    value=float(row["value"]),
                )
            )

    print(f"[CAL] Loaded {len(items)} items from {manifest_path}.", flush=True)
    return items


def _quantize_input(batch: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    """Quantize a float32 batch to the scalar reader's input tensor dtype."""
    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(batch / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(np.int8)


def _dequantize_output(output_tensor: np.ndarray, output_details: dict[str, Any]) -> float:
    """Convert the scalar reader output tensor back to a float prediction."""
    scale = float(output_details["quantization"][0])
    zero_point = int(output_details["quantization"][1])
    return float(scale * (int(output_tensor) - zero_point))


def _dequantize_tensor(
    tensor: np.ndarray,
    output_details: dict[str, Any],
) -> np.ndarray:
    """Convert a quantized tensor into float values using its tensor metadata."""
    scale = float(output_details["quantization"][0])
    zero_point = int(output_details["quantization"][1])
    return scale * (np.asarray(tensor, dtype=np.float32) - zero_point)


def _resolve_obb_kind(model_path: Path, model_kind: ObbModelKind) -> Literal["keras", "tflite"]:
    """Pick the OBB backend from the CLI flag or the file suffix."""
    if model_kind == "auto":
        return "tflite" if model_path.suffix.lower() == ".tflite" else "keras"
    return cast(Literal["keras", "tflite"], model_kind)


def _load_obb_session(model_path: Path, model_kind: ObbModelKind) -> ObbSession:
    """Load the OBB backend with the right tensor handling."""
    resolved_kind = _resolve_obb_kind(model_path, model_kind)
    print(f"[CAL] Loading OBB model from {model_path} as {resolved_kind}...", flush=True)
    if resolved_kind == "keras":
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
            },
            compile=False,
            safe_mode=False,
        )
        print(f"[CAL] OBB model loaded: {model.name}", flush=True)
        return ObbSession(kind="keras", model=model)

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"[CAL] OBB input details: {input_details}", flush=True)
    print(f"[CAL] OBB output details: {output_details}", flush=True)
    return ObbSession(
        kind="tflite",
        model=interpreter,
        input_details=input_details,
        output_details=output_details,
    )


def _extract_obb_params(outputs: Any) -> np.ndarray:
    """Flatten the localizer prediction to the normalized OBB parameter vector."""
    if isinstance(outputs, dict):
        if "obb_params" in outputs:
            return np.asarray(outputs["obb_params"]).reshape(-1)
        first_key = next(iter(outputs))
        return np.asarray(outputs[first_key]).reshape(-1)
    return np.asarray(outputs).reshape(-1)


def _expand_axis_aligned_box(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    image_width: int,
    image_height: int,
    min_size: float,
) -> tuple[float, float, float, float]:
    """Expand a box in-place so it stays usable by the scalar reader."""
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    target_width = max(width, min_size)
    target_height = max(height, min_size)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)

    new_x_min = center_x - 0.5 * target_width
    new_y_min = center_y - 0.5 * target_height
    new_x_max = new_x_min + target_width
    new_y_max = new_y_min + target_height

    if new_x_min < 0.0:
        new_x_max -= new_x_min
        new_x_min = 0.0
    if new_y_min < 0.0:
        new_y_max -= new_y_min
        new_y_min = 0.0
    if new_x_max > float(image_width):
        shift = new_x_max - float(image_width)
        new_x_min = max(0.0, new_x_min - shift)
        new_x_max = float(image_width)
    if new_y_max > float(image_height):
        shift = new_y_max - float(image_height)
        new_y_min = max(0.0, new_y_min - shift)
        new_y_max = float(image_height)

    if new_x_max <= new_x_min + 1.0:
        new_x_max = min(float(image_width), new_x_min + 1.0)
    if new_y_max <= new_y_min + 1.0:
        new_y_max = min(float(image_height), new_y_min + 1.0)
    return (new_x_min, new_y_min, new_x_max, new_y_max)


def _obb_params_to_crop_box(
    obb_params: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    input_size: int,
    obb_crop_scale: float,
    min_crop_size: float,
) -> tuple[tuple[float, float, float, float], dict[str, float]]:
    """Convert normalized OBB parameters into a source-image crop box."""
    if obb_params.size < 6:
        raise ValueError("OBB prediction did not contain six parameters.")

    center_x_norm = float(np.clip(obb_params[0], 0.0, 1.0))
    center_y_norm = float(np.clip(obb_params[1], 0.0, 1.0))
    box_w_norm = float(np.clip(obb_params[2], 0.05, 1.0))
    box_h_norm = float(np.clip(obb_params[3], 0.05, 1.0))
    angle_cos = float(obb_params[4])
    angle_sin = float(obb_params[5])
    theta_rad = 0.5 * math.atan2(angle_sin, angle_cos)

    canvas_center_x = center_x_norm * float(input_size)
    canvas_center_y = center_y_norm * float(input_size)
    half_width = 0.5 * box_w_norm * float(input_size) * obb_crop_scale
    half_height = 0.5 * box_h_norm * float(input_size) * obb_crop_scale

    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    canvas_corners = (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    )

    source_points: list[tuple[float, float]] = []
    for dx, dy in canvas_corners:
        canvas_x = canvas_center_x + (dx * cos_theta - dy * sin_theta)
        canvas_y = canvas_center_y + (dx * sin_theta + dy * cos_theta)
        source_point = source_xy_from_resized_xy(
            (canvas_x, canvas_y),
            crop_box_xyxy=(0.0, 0.0, float(source_width), float(source_height)),
            image_height=source_height,
            image_width=source_width,
        )
        source_points.append(source_point)

    x_values = [point[0] for point in source_points]
    y_values = [point[1] for point in source_points]
    crop_box = _expand_axis_aligned_box(
        min(x_values),
        min(y_values),
        max(x_values),
        max(y_values),
        image_width=source_width,
        image_height=source_height,
        min_size=min_crop_size,
    )
    geometry = {
        "obb_center_x": center_x_norm,
        "obb_center_y": center_y_norm,
        "obb_width": box_w_norm,
        "obb_height": box_h_norm,
        "obb_angle_deg": math.degrees(theta_rad),
    }
    return crop_box, geometry


def _predict_cascade_raw(
    obb: ObbSession,
    scalar_interpreter: tf.lite.Interpreter,
    input_details: dict[str, Any],
    output_details: dict[str, Any],
    image_path: Path,
    *,
    image_size: int,
    obb_crop_scale: float,
    min_crop_size: float,
) -> float:
    """Predict the raw scalar output for one image through the OBB cascade."""
    source_image = load_rgb_image(image_path)
    source_height, source_width = source_image.shape[:2]
    full_frame = resize_with_pad_rgb(
        source_image,
        (
            0.0,
            0.0,
            float(source_width),
            float(source_height),
        ),
        image_size=image_size,
    )
    obb_batch = np.expand_dims(full_frame.astype(np.float32) / 255.0, axis=0)

    if obb.kind == "keras":
        obb_model = cast(tf.keras.Model, obb.model)
        obb_prediction = obb_model.predict(obb_batch, verbose=0)
        obb_params = _extract_obb_params(obb_prediction)
    else:
        obb_interpreter = cast(tf.lite.Interpreter, obb.model)
        assert obb.input_details is not None
        assert obb.output_details is not None
        obb_input = _quantize_input(obb_batch, obb.input_details)
        obb_interpreter.set_tensor(int(obb.input_details["index"]), obb_input)
        obb_interpreter.invoke()
        obb_output = obb_interpreter.get_tensor(int(obb.output_details["index"]))[0]
        obb_params = _dequantize_tensor(obb_output, obb.output_details)

    crop_box_xyxy, _geometry = _obb_params_to_crop_box(
        obb_params,
        source_width=source_width,
        source_height=source_height,
        input_size=image_size,
        obb_crop_scale=obb_crop_scale,
        min_crop_size=min_crop_size,
    )

    crop = resize_with_pad_rgb(source_image, crop_box_xyxy, image_size=image_size)
    batch = np.expand_dims(crop.astype(np.float32) / 255.0, axis=0)
    quantized_batch = _quantize_input(batch, input_details)
    scalar_interpreter.set_tensor(int(input_details["index"]), quantized_batch)
    scalar_interpreter.invoke()
    raw_output = scalar_interpreter.get_tensor(int(output_details["index"]))[0][0]
    return _dequantize_output(raw_output, output_details)


def _load_prediction_set(
    *,
    name: str,
    obb: ObbSession,
    scalar_interpreter: tf.lite.Interpreter,
    input_details: dict[str, Any],
    output_details: dict[str, Any],
    items: list[ManifestItem],
    image_size: int,
    obb_crop_scale: float,
    min_crop_size: float,
) -> PredictionSet:
    """Run the cascade on one manifest and collect raw outputs and labels."""
    raw_predictions: list[float] = []
    labels: list[float] = []
    for item in items:
        print(f"[CAL] Predicting {name}:{item.image_path.name}...", flush=True)
        raw_prediction = _predict_cascade_raw(
            obb,
            scalar_interpreter,
            input_details,
            output_details,
            item.image_path,
            image_size=image_size,
            obb_crop_scale=obb_crop_scale,
            min_crop_size=min_crop_size,
        )
        raw_predictions.append(raw_prediction)
        labels.append(item.value)
        print(
            f"[CAL] {name}:{item.image_path.name}: true={item.value:.4f} "
            f"raw={raw_prediction:.4f}",
            flush=True,
        )

    return PredictionSet(
        name=name,
        raw_predictions=np.asarray(raw_predictions, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
    )


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
    """Fit a fixed piecewise-linear calibration model on the raw output."""
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


def _apply_affine(predictions: np.ndarray, *, scale: float, bias: float) -> np.ndarray:
    """Apply an affine calibration to a vector of predictions."""
    return (scale * predictions) + bias


def _apply_piecewise(
    predictions: np.ndarray,
    *,
    bias: float,
    weights: np.ndarray,
    knots: np.ndarray,
) -> np.ndarray:
    """Apply the piecewise-linear calibration used for the firmware tail."""
    if weights.ndim != 1 or knots.ndim != 1:
        raise ValueError("Piecewise calibration arrays must be one-dimensional.")
    if weights.size != knots.size + 1:
        raise ValueError("Expected one linear weight plus one weight per knot.")

    design_terms = [predictions]
    for knot in knots:
        design_terms.append(np.maximum(predictions - knot, 0.0))
    design = np.column_stack(design_terms)
    return design @ weights + bias


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


def _mean_abs_error(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute the mean absolute error for two one-dimensional arrays."""
    if predictions.ndim != 1 or labels.ndim != 1:
        raise ValueError("Calibration inputs must be one-dimensional.")
    if predictions.size != labels.size:
        raise ValueError("Prediction and label counts must match.")
    if predictions.size == 0:
        return 0.0
    return float(np.mean(np.abs(predictions - labels)))


def _load_items_for_manifests(manifests: list[Path]) -> list[ManifestItem]:
    """Load and concatenate multiple manifests into one dataset."""
    items: list[ManifestItem] = []
    for manifest in manifests:
        items.extend(_load_manifest(manifest))
    return items


def main() -> None:
    """Fit the firmware calibration and summarize hard-case performance."""
    args = _parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    obb = _load_obb_session(args.obb_model, args.obb_model_kind)
    print(f"[CAL] Loading scalar reader from {args.scalar_model}...", flush=True)
    scalar_interpreter = tf.lite.Interpreter(model_path=str(args.scalar_model), num_threads=1)
    scalar_interpreter.allocate_tensors()
    input_details = scalar_interpreter.get_input_details()[0]
    output_details = scalar_interpreter.get_output_details()[0]
    print(f"[CAL] Scalar input details: {input_details}", flush=True)
    print(f"[CAL] Scalar output details: {output_details}", flush=True)

    fit_items = _load_items_for_manifests(args.fit_manifest)
    fit_names = [manifest.as_posix() for manifest in args.fit_manifest]
    fit_set = _load_prediction_set(
        name="fit",
        obb=obb,
        scalar_interpreter=scalar_interpreter,
        input_details=input_details,
        output_details=output_details,
        items=fit_items,
        image_size=args.image_size,
        obb_crop_scale=args.obb_crop_scale,
        min_crop_size=args.min_crop_size,
    )

    test_sets: list[PredictionSet] = []
    for index, manifest in enumerate(args.test_manifest):
        test_items = _load_manifest(manifest)
        test_sets.append(
            _load_prediction_set(
                name=f"test{index}",
                obb=obb,
                scalar_interpreter=scalar_interpreter,
                input_details=input_details,
                output_details=output_details,
                items=test_items,
                image_size=args.image_size,
                obb_crop_scale=args.obb_crop_scale,
                min_crop_size=args.min_crop_size,
            )
        )

    reference_fit = _apply_affine(
        fit_set.raw_predictions,
        scale=args.reference_scale,
        bias=args.reference_bias,
    )
    reference_fit_metrics = _summarize_errors(reference_fit, fit_set.labels, title="fit_reference")
    reference_test_metrics: list[dict[str, float]] = []
    for test_set in test_sets:
        reference_test = _apply_affine(
            test_set.raw_predictions,
            scale=args.reference_scale,
            bias=args.reference_bias,
        )
        reference_test_metrics.append(
            _summarize_errors(reference_test, test_set.labels, title=f"{test_set.name}_reference")
        )

    affine_scale, affine_bias = _fit_affine_calibration(fit_set.raw_predictions, fit_set.labels)
    affine_fit = _apply_affine(fit_set.raw_predictions, scale=affine_scale, bias=affine_bias)
    affine_fit_metrics = _summarize_errors(affine_fit, fit_set.labels, title="fit_affine")
    affine_test_metrics: list[dict[str, float]] = []
    for test_set in test_sets:
        affine_test = _apply_affine(
            test_set.raw_predictions,
            scale=affine_scale,
            bias=affine_bias,
        )
        affine_test_metrics.append(
            _summarize_errors(affine_test, test_set.labels, title=f"{test_set.name}_affine")
        )

    piecewise_bias, piecewise_weights, piecewise_knots = _fit_piecewise_calibration(
        fit_set.raw_predictions,
        fit_set.labels,
        knot_mode=args.spline_knot_mode,
        knot_count=args.spline_knot_count,
    )
    piecewise_fit = _apply_piecewise(
        fit_set.raw_predictions,
        bias=piecewise_bias,
        weights=piecewise_weights,
        knots=piecewise_knots,
    )
    piecewise_fit_metrics = _summarize_errors(piecewise_fit, fit_set.labels, title="fit_piecewise")
    piecewise_test_metrics: list[dict[str, float]] = []
    for test_set in test_sets:
        piecewise_test = _apply_piecewise(
            test_set.raw_predictions,
            bias=piecewise_bias,
            weights=piecewise_weights,
            knots=piecewise_knots,
        )
        piecewise_test_metrics.append(
            _summarize_errors(
                piecewise_test,
                test_set.labels,
                title=f"{test_set.name}_piecewise",
            )
        )

    affine_test_predictions = np.concatenate(
        [
            _apply_affine(
                test_set.raw_predictions,
                scale=affine_scale,
                bias=affine_bias,
            )
            for test_set in test_sets
        ]
    )
    affine_test_labels = np.concatenate([test_set.labels for test_set in test_sets])
    affine_test_mae = _mean_abs_error(affine_test_predictions, affine_test_labels)

    piecewise_test_predictions = np.concatenate(
        [
            _apply_piecewise(
                test_set.raw_predictions,
                bias=piecewise_bias,
                weights=piecewise_weights,
                knots=piecewise_knots,
            )
            for test_set in test_sets
        ]
    )
    piecewise_test_labels = np.concatenate([test_set.labels for test_set in test_sets])
    piecewise_test_mae = _mean_abs_error(piecewise_test_predictions, piecewise_test_labels)

    selected_mode = "piecewise" if piecewise_test_mae <= affine_test_mae else "affine"
    print(
        f"[CAL] Selected calibration mode: {selected_mode} "
        f"(affine_test_mae={affine_test_mae:.4f}, "
        f"piecewise_test_mae={piecewise_test_mae:.4f})",
        flush=True,
    )

    payload: dict[str, object] = {
        "selected_mode": selected_mode,
        "fit_manifests": fit_names,
        "test_manifests": [manifest.as_posix() for manifest in args.test_manifest],
        "image_size": args.image_size,
        "obb_crop_scale": args.obb_crop_scale,
        "min_crop_size": args.min_crop_size,
        "reference_scale": args.reference_scale,
        "reference_bias": args.reference_bias,
        "reference_fit_metrics": reference_fit_metrics,
        "reference_test_metrics": reference_test_metrics,
        "affine": {
            "scale": affine_scale,
            "bias": affine_bias,
            "test_mae": affine_test_mae,
            "fit_metrics": affine_fit_metrics,
            "test_metrics": affine_test_metrics,
        },
        "piecewise": {
            "bias": piecewise_bias,
            "weights": piecewise_weights.tolist(),
            "knots": piecewise_knots.tolist(),
            "test_mae": piecewise_test_mae,
            "fit_metrics": piecewise_fit_metrics,
            "test_metrics": piecewise_test_metrics,
        },
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[CAL] Wrote calibration payload to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
