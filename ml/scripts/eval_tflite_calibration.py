"""Fit lightweight scalar calibration on top of a TFLite model's predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Final

import numpy as np
import tensorflow as tf

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the TFLite calibration probe."""
    parser = argparse.ArgumentParser(
        description="Fit affine and piecewise calibration on TFLite scalar predictions."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model_int8.tflite.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV manifest with image_path,value rows used for calibration.",
    )
    parser.add_argument(
        "--knot-mode",
        type=str,
        choices=["all", "interior", "quantile"],
        default="all",
        help="How to choose spline knot locations for the piecewise fit.",
    )
    parser.add_argument(
        "--knot-count",
        type=int,
        default=8,
        help="How many quantile knots to use when --knot-mode quantile is selected.",
    )
    parser.add_argument(
        "--weight-prefix",
        type=str,
        default="",
        help="Optional image-name prefix to upweight during the calibration fit.",
    )
    parser.add_argument(
        "--weight-factor",
        type=float,
        default=1.0,
        help="Weight multiplier to apply to images matching --weight-prefix.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the fitted calibration parameters as JSON.",
    )
    return parser.parse_args()


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest image path relative to the repo root when needed."""
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return REPO_ROOT / image_path


def _load_items(manifest_path: Path) -> tuple[list[Path], np.ndarray]:
    """Load image paths and scalar labels from a CSV manifest."""
    image_paths: list[Path] = []
    values: list[float] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_paths.append(_resolve_image_path(row["image_path"]))
            values.append(float(row["value"]))
    return image_paths, np.asarray(values, dtype=np.float32)


def _predict_tflite(model_path: Path, image_paths: list[Path]) -> np.ndarray:
    """Run a scalar TFLite model on each image and return raw float predictions."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_index = int(input_detail["index"])
    output_index = int(output_detail["index"])
    input_dtype = input_detail["dtype"]
    output_dtype = output_detail["dtype"]
    input_scale = float(input_detail["quantization"][0])
    input_zero_point = int(input_detail["quantization"][1])
    output_scale = float(output_detail["quantization"][0])
    output_zero_point = int(output_detail["quantization"][1])

    print(
        "[TFLITE] input_quant="
        f"({input_scale:.10f}, {input_zero_point}) "
        "output_quant="
        f"({output_scale:.10f}, {output_zero_point})",
        flush=True,
    )

    predictions: list[float] = []
    for image_path in image_paths:
        image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        image_array = tf.keras.utils.img_to_array(image).astype(np.float32) / 255.0
        if np.issubdtype(input_dtype, np.floating):
            interpreter.set_tensor(input_index, image_array[None, ...].astype(input_dtype))
        else:
            quantized = np.round(image_array / input_scale + input_zero_point).astype(input_dtype)
            interpreter.set_tensor(input_index, quantized[None, ...])
        interpreter.invoke()
        output_value = interpreter.get_tensor(output_index)[0][0]
        if np.issubdtype(output_dtype, np.floating):
            prediction = float(output_value)
        else:
            prediction = (float(output_value) - output_zero_point) * output_scale
        predictions.append(prediction)
        print(
            f"[TFLITE] {image_path.name}: pred={prediction:.4f}",
            flush=True,
        )

    return np.asarray(predictions, dtype=np.float32)


def _fit_affine(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit a simple affine calibration mapping."""
    design = np.vstack([predictions, np.ones_like(predictions)]).T
    scale, bias = np.linalg.lstsq(design, labels, rcond=None)[0]
    return float(scale), float(bias)


def _weighted_lstsq(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Solve a weighted least-squares system."""
    if weights is None:
        solution = np.linalg.lstsq(design, labels, rcond=None)[0]
        return np.asarray(solution, dtype=np.float32)

    if weights.ndim != 1:
        raise ValueError("Sample weights must be one-dimensional.")
    if weights.size != labels.size:
        raise ValueError("Sample weight count must match label count.")

    sqrt_weights = np.sqrt(weights).astype(np.float32)
    weighted_design = design * sqrt_weights[:, None]
    weighted_labels = labels * sqrt_weights
    solution = np.linalg.lstsq(weighted_design, weighted_labels, rcond=None)[0]
    return np.asarray(solution, dtype=np.float32)


def _fit_piecewise(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    knot_mode: str,
    knot_count: int,
    sample_weights: np.ndarray | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a fixed piecewise-linear calibration mapping."""
    unique_predictions = np.sort(np.unique(predictions))
    if unique_predictions.size < 2:
        raise ValueError("Need at least two unique predictions for spline calibration.")

    if knot_mode == "interior":
        knots = unique_predictions[1:-1]
    elif knot_mode == "quantile":
        if knot_count < 2:
            raise ValueError("Need at least two quantile knots for spline calibration.")
        if unique_predictions.size <= knot_count:
            knots = unique_predictions
        else:
            quantiles = np.linspace(0.0, 1.0, knot_count + 2, dtype=np.float32)[1:-1]
            knots = np.quantile(predictions, quantiles).astype(np.float32)
            knots = np.unique(knots)
            knots = knots[(knots > unique_predictions[0]) & (knots < unique_predictions[-1])]
            if knots.size == 0:
                knots = unique_predictions[1:-1]
    else:
        knots = unique_predictions

    features = [predictions]
    for knot in knots:
        features.append(np.maximum(predictions - knot, 0.0))
    design = np.vstack(features).T
    augmented_design = np.column_stack([design, np.ones_like(predictions)])
    solution = _weighted_lstsq(augmented_design, labels, sample_weights)
    weights = solution[:-1]
    bias = float(solution[-1])
    return bias, np.asarray(weights, dtype=np.float32), np.asarray(knots, dtype=np.float32)


def main() -> None:
    """Run the TFLite calibration fit and print raw and calibrated errors."""
    args = _parse_args()
    image_paths, labels = _load_items(args.manifest)
    predictions = _predict_tflite(args.model, image_paths)

    sample_weights: np.ndarray | None = None
    if args.weight_prefix:
        sample_weights = np.ones_like(labels, dtype=np.float32)
        match_mask = np.asarray(
            [path.name.startswith(args.weight_prefix) for path in image_paths],
            dtype=bool,
        )
        sample_weights[match_mask] = float(args.weight_factor)
        print(
            f"[CAL] Weighting {int(match_mask.sum())} samples with factor {args.weight_factor:.4f} "
            f"for prefix {args.weight_prefix!r}.",
            flush=True,
        )

    raw_mae = float(np.mean(np.abs(predictions - labels)))
    raw_max = float(np.max(np.abs(predictions - labels)))
    print(f"[CAL] raw_mae={raw_mae:.4f} raw_max={raw_max:.4f}", flush=True)

    scale, bias = _fit_affine(predictions, labels)
    affine_predictions = scale * predictions + bias
    affine_mae = float(np.mean(np.abs(affine_predictions - labels)))
    affine_max = float(np.max(np.abs(affine_predictions - labels)))
    print(
        f"[CAL] affine_mae={affine_mae:.4f} affine_max={affine_max:.4f} "
        f"scale={scale:.6f} bias={bias:.6f}",
        flush=True,
    )

    piecewise_bias, piecewise_weights, knots = _fit_piecewise(
        predictions,
        labels,
        knot_mode=args.knot_mode,
        knot_count=args.knot_count,
        sample_weights=sample_weights,
    )
    piecewise_design = np.column_stack(
        [predictions] + [np.maximum(predictions - knot, 0.0) for knot in knots]
    )
    piecewise_predictions = piecewise_design @ piecewise_weights + piecewise_bias
    piecewise_mae = float(np.mean(np.abs(piecewise_predictions - labels)))
    piecewise_max = float(np.max(np.abs(piecewise_predictions - labels)))
    print(
        f"[CAL] piecewise_mae={piecewise_mae:.4f} piecewise_max={piecewise_max:.4f} "
        f"knot_mode={args.knot_mode} knots={len(knots)}",
        flush=True,
    )

    worst_indices = np.argsort(np.abs(piecewise_predictions - labels))[::-1][:10]
    for idx in worst_indices:
        print(
            "[CAL] worst: "
            f"{image_paths[idx].name} true={labels[idx]:.1f} "
            f"raw={predictions[idx]:.4f} cal={piecewise_predictions[idx]:.4f} "
            f"err={abs(piecewise_predictions[idx] - labels[idx]):.4f}",
            flush=True,
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(args.model),
            "manifest": str(args.manifest),
            "knot_mode": args.knot_mode,
            "knot_count": args.knot_count,
            "weight_prefix": args.weight_prefix,
            "weight_factor": args.weight_factor,
            "raw_mae": raw_mae,
            "raw_max": raw_max,
            "calibrated_mae": piecewise_mae,
            "calibrated_max": piecewise_max,
            "bias": float(piecewise_bias),
            "weights": piecewise_weights.tolist(),
            "knots": knots.tolist(),
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[CAL] Wrote calibration JSON to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
