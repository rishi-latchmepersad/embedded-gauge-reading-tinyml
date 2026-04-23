"""Evaluate an OBB localizer + scalar reader cascade on a labeled manifest."""

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

# Make the package importable when this script is run from the `ml/` directory.
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
class EvalRow:
    """Per-sample results for the OBB + scalar benchmark."""

    image_path: Path
    true_value: float
    prediction: float
    abs_err: float
    crop_x_min: float
    crop_y_min: float
    crop_x_max: float
    crop_y_max: float
    obb_center_x: float
    obb_center_y: float
    obb_width: float
    obb_height: float
    obb_angle_deg: float


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the OBB cascade evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate an OBB localizer + scalar reader cascade on a manifest."
    )
    parser.add_argument(
        "--obb-model",
        type=Path,
        required=True,
        help="Path to the saved OBB localizer model.",
    )
    parser.add_argument(
        "--obb-model-kind",
        choices=["auto", "keras", "tflite"],
        default="auto",
        help=(
            "OBB backend to load. 'auto' picks TFLite for .tflite paths "
            "and Keras otherwise."
        ),
    )
    parser.add_argument(
        "--scalar-model",
        type=Path,
        required=True,
        help="Path to the quantized scalar TFLite reader.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV file with image_path,value rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "cascade_eval" / "obb_scalar",
        help="Directory where the report CSV and summary JSON should be written.",
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
    return parser.parse_args()


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest image path relative to the repo root when needed."""
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return REPO_ROOT / image_path


def _load_manifest(manifest_path: Path) -> list[tuple[Path, float]]:
    """Load labeled image paths and scalar values from the CSV manifest."""
    items: list[tuple[Path, float]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            items.append((_resolve_image_path(row["image_path"]), float(row["value"])))
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
    print(
        f"[OBB-EVAL] Loading OBB model from {model_path} as {resolved_kind}...",
        flush=True,
    )
    if resolved_kind == "keras":
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
            },
            compile=False,
            safe_mode=False,
        )
        print(f"[OBB-EVAL] OBB model loaded: {model.name}", flush=True)
        return ObbSession(kind="keras", model=model)

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"[OBB-EVAL] OBB input details: {input_details}", flush=True)
    print(f"[OBB-EVAL] OBB output details: {output_details}", flush=True)
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


def _predict_obb_scalar(
    obb: ObbSession,
    scalar_interpreter: tf.lite.Interpreter,
    input_details: dict[str, Any],
    output_details: dict[str, Any],
    image_path: Path,
    *,
    image_size: int,
    obb_crop_scale: float,
    min_crop_size: float,
) -> dict[str, float]:
    """Predict a crop with the OBB localizer, then read that crop with the scalar model."""
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

    crop_box_xyxy, geometry = _obb_params_to_crop_box(
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
    prediction = _dequantize_output(raw_output, output_details)

    return {
        "prediction": prediction,
        "_crop_x_min": crop_box_xyxy[0],
        "_crop_y_min": crop_box_xyxy[1],
        "_crop_x_max": crop_box_xyxy[2],
        "_crop_y_max": crop_box_xyxy[3],
        **geometry,
    }


def _run_obb_scalar_on_manifest(
    *,
    obb: ObbSession,
    scalar_interpreter: tf.lite.Interpreter,
    input_details: dict[str, Any],
    output_details: dict[str, Any],
    items: list[tuple[Path, float]],
    image_size: int,
    obb_crop_scale: float,
    min_crop_size: float,
    output_dir: Path,
) -> list[EvalRow]:
    """Run the cascade over every manifest item and write a machine-readable log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[EvalRow] = []
    report_path = output_dir / "rows.csv"

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_path",
                "true_value",
                "prediction",
                "abs_err",
                "crop_x_min",
                "crop_y_min",
                "crop_x_max",
                "crop_y_max",
                "obb_center_x",
                "obb_center_y",
                "obb_width",
                "obb_height",
                "obb_angle_deg",
            ]
        )

        for image_path, true_value in items:
            print(f"[OBB-EVAL] Predicting {image_path.name}...", flush=True)
            pred = _predict_obb_scalar(
                obb,
                scalar_interpreter,
                input_details,
                output_details,
                image_path,
                image_size=image_size,
                obb_crop_scale=obb_crop_scale,
                min_crop_size=min_crop_size,
            )
            abs_error = abs(pred["prediction"] - true_value)
            row = EvalRow(
                image_path=image_path,
                true_value=true_value,
                prediction=pred["prediction"],
                abs_err=abs_error,
                crop_x_min=pred["_crop_x_min"],
                crop_y_min=pred["_crop_y_min"],
                crop_x_max=pred["_crop_x_max"],
                crop_y_max=pred["_crop_y_max"],
                obb_center_x=pred["obb_center_x"],
                obb_center_y=pred["obb_center_y"],
                obb_width=pred["obb_width"],
                obb_height=pred["obb_height"],
                obb_angle_deg=pred["obb_angle_deg"],
            )
            rows.append(row)
            writer.writerow(
                [
                    row.image_path.as_posix(),
                    row.true_value,
                    row.prediction,
                    row.abs_err,
                    row.crop_x_min,
                    row.crop_y_min,
                    row.crop_x_max,
                    row.crop_y_max,
                    row.obb_center_x,
                    row.obb_center_y,
                    row.obb_width,
                    row.obb_height,
                    row.obb_angle_deg,
                ]
            )
            print(
                f"[OBB-EVAL] {row.image_path.name}: true={row.true_value:.4f} "
                f"pred={row.prediction:.4f} abs_err={row.abs_err:.4f} "
                f"crop=({row.crop_x_min:.1f},{row.crop_y_min:.1f},"
                f"{row.crop_x_max:.1f},{row.crop_y_max:.1f}) "
                f"obb=({row.obb_center_x:.3f},{row.obb_center_y:.3f},"
                f"{row.obb_width:.3f},{row.obb_height:.3f},"
                f"{row.obb_angle_deg:.1f}deg)",
                flush=True,
            )

    return rows


def main() -> None:
    """Run the OBB cascade benchmark and print a summary table."""
    args = _parse_args()
    obb = _load_obb_session(args.obb_model, args.obb_model_kind)
    items = _load_manifest(args.manifest)

    print(f"[OBB-EVAL] Loading scalar reader from {args.scalar_model}...", flush=True)
    scalar_interpreter = tf.lite.Interpreter(model_path=str(args.scalar_model), num_threads=1)
    scalar_interpreter.allocate_tensors()
    input_details = scalar_interpreter.get_input_details()[0]
    output_details = scalar_interpreter.get_output_details()[0]
    print(f"[OBB-EVAL] Scalar input details: {input_details}", flush=True)
    print(f"[OBB-EVAL] Scalar output details: {output_details}", flush=True)

    rows = _run_obb_scalar_on_manifest(
        obb=obb,
        scalar_interpreter=scalar_interpreter,
        input_details=input_details,
        output_details=output_details,
        items=items,
        image_size=args.image_size,
        obb_crop_scale=args.obb_crop_scale,
        min_crop_size=args.min_crop_size,
        output_dir=args.output_dir,
    )

    if not rows:
        print("[OBB-EVAL] No samples were scored.", flush=True)
        return

    abs_errors = np.asarray([row.abs_err for row in rows], dtype=np.float32)
    worst = max(rows, key=lambda row: row.abs_err)
    summary = {
        "samples": len(rows),
        "skipped": len(items) - len(rows),
        "mean_abs_err": float(np.mean(abs_errors)),
        "max_abs_err": float(np.max(abs_errors)),
        "cases_over_5c": int(np.sum(abs_errors > 5.0)),
        "worst_image": worst.image_path.as_posix(),
        "worst_true": worst.true_value,
        "worst_pred": worst.prediction,
        "worst_abs_err": worst.abs_err,
        "obb_model": str(args.obb_model),
        "scalar_model": str(args.scalar_model),
        "obb_crop_scale": args.obb_crop_scale,
        "min_crop_size": args.min_crop_size,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OBB-EVAL] samples={summary['samples']} skipped={summary['skipped']}", flush=True)
    print(f"[OBB-EVAL] mean_abs_err={summary['mean_abs_err']:.4f}", flush=True)
    print(f"[OBB-EVAL] max_abs_err={summary['max_abs_err']:.4f}", flush=True)
    print(f"[OBB-EVAL] cases_over_5c={summary['cases_over_5c']}", flush=True)
    print(
        f"[OBB-EVAL] worst={summary['worst_image']} true={summary['worst_true']:.4f} "
        f"pred={summary['worst_pred']:.4f} abs_err={summary['worst_abs_err']:.4f}",
        flush=True,
    )
    print(f"[OBB-EVAL] report_csv={args.output_dir / 'rows.csv'}", flush=True)
    print(f"[OBB-EVAL] summary_json={summary_path}", flush=True)


if __name__ == "__main__":
    main()
