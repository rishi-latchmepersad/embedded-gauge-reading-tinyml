#!/usr/bin/env python3
"""Package and replay the spatial SimCC PTQ model on the hard-case set.

This script does two related jobs:
1. Stage the saved PTQ TFLite artifact into a deployment directory.
2. Replay the packaged model on the 19 hard-case crops and save metrics.

The hard-case replay uses the exact crop metadata produced by
``load_heatmap_sample`` so we evaluate the same 224-space inputs that the
training and export flow already use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

# Keep TensorFlow on CPU so the replay is deterministic in WSL.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (  # noqa: E402
    dequantize_output_tensor,
    quantize_input_batch,
    summarize_tflite_contract,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import (  # noqa: E402
    load_clean_geometry_examples,
    load_heatmap_sample,
)

MODEL_NAME: Final[str] = "simcc_gauge_v2_spatial_qat_sc128_int8"
SOURCE_MODEL: Final[Path] = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "training"
    / "simcc_gauge_v2_spatial_qat_sc128"
    / "model_int8.tflite"
)
DEFAULT_MANIFEST: Final[Path] = (
    REPO_ROOT
    / "ml"
    / "data"
    / "geometry_heatmap_v13_trusted_train_manifest.csv"
)
DEFAULT_HARD_CASE_MANIFEST: Final[Path] = REPO_ROOT / "ml" / "data" / "hard_cases.csv"
DEFAULT_OUTPUT_DIR: Final[Path] = (
    REPO_ROOT / "ml" / "artifacts" / "deployment" / MODEL_NAME
)
PACKAGED_MODEL_NAME: Final[str] = "model_int8.tflite"
CONTRACT_NAME: Final[str] = "tflite_tensor_contract.json"
METADATA_NAME: Final[str] = "metadata.json"
SUMMARY_NAME: Final[str] = "hardcase_replay_summary.json"
PREDICTIONS_NAME: Final[str] = "hardcase_replay_predictions.csv"
NUM_BINS: Final[int] = 112
CROP_INDEX_MAX: Final[float] = 223.0
TEMP_MIN_C: Final[float] = -30.0
TEMP_MAX_C: Final[float] = 50.0


@dataclass(frozen=True, slots=True)
class HardCaseReplayRow:
    """One replay row for the hard-case CSV output."""

    image_name: str
    true_temperature_c: float
    predicted_temperature_c: float
    absolute_error_c: float
    true_angle_deg: float
    predicted_angle_deg: float
    angle_error_deg: float
    center_x_err_px: float
    center_y_err_px: float
    tip_x_err_px: float
    tip_y_err_px: float
    confidence: float


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for packaging and replay."""
    parser = argparse.ArgumentParser(
        description="Stage the SimCC PTQ model and replay it on the hard cases."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=SOURCE_MODEL,
        help="Path to the saved PTQ model to package and replay.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Clean manifest used to find the labeled geometry examples.",
    )
    parser.add_argument(
        "--hard-case-manifest",
        type=Path,
        default=DEFAULT_HARD_CASE_MANIFEST,
        help="CSV containing the hard-case image basenames to replay.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Deployment directory where the packaged model and replay artifacts will be written.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Square crop size expected by the SimCC model.",
    )
    parser.add_argument(
        "--heatmap-size",
        type=int,
        default=NUM_BINS,
        help="Number of SimCC bins per coordinate head.",
    )
    return parser.parse_args()


def _soft_argmax_1d(logits: np.ndarray) -> np.ndarray:
    """Convert a 1D SimCC distribution into a normalized coordinate."""
    bins = np.linspace(0.0, 1.0, int(logits.shape[-1]), dtype=np.float32)
    return np.dot(np.asarray(logits, dtype=np.float32), bins)


def _angle_from_coords(cx: np.ndarray, cy: np.ndarray, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    """Compute the needle angle from center and tip coordinates."""
    return np.degrees(np.arctan2(-(ty - cy), tx - cx)) % 360.0


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


def _detail_map(details: Any) -> dict[str, dict[str, Any]]:
    """Map tensor names to their TFLite tensor details.

    TFLite sometimes returns a list of tensor dictionaries and sometimes a
    dict keyed by output name, depending on whether we are looking at a plain
    interpreter or a signature runner.
    """
    if isinstance(details, dict):
        return {str(name): dict(detail) for name, detail in details.items()}
    return {str(detail["name"]): dict(detail) for detail in details}


def _dequantize_tflite_tensor(tensor: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    """Convert one TFLite tensor to float32, regardless of storage dtype."""
    return dequantize_output_tensor(tensor, detail)


def _predict_tflite(
    model_path: Path,
    inputs: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Run the TFLite model and return semantic outputs for each sample.

    The model may expose either a signature runner or plain output tensors.
    We try the semantic names first, then fall back to the saved output order.
    """
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()

    semantic_results: dict[str, list[np.ndarray]] = {
        "cx": [],
        "cy": [],
        "tx": [],
        "ty": [],
        "conf": [],
    }

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
            for input_array in inputs:
                batch = np.expand_dims(np.asarray(input_array, dtype=np.float32), axis=0)
                raw_outputs = runner(**{input_name: batch})
                for semantic, output_name in semantic_to_name.items():
                    assert output_name is not None
                    detail = runner_output_details[output_name]
                    semantic_results[semantic].append(
                        _dequantize_tflite_tensor(raw_outputs[output_name], detail)
                    )
            return {key: np.concatenate(value, axis=0) for key, value in semantic_results.items()}

        # If the signature names are opaque, fall back to the stable output order.
        ordered_output_details = list(runner_output_details.values())
        for input_array in inputs:
            batch = np.expand_dims(np.asarray(input_array, dtype=np.float32), axis=0)
            raw_outputs = runner(**{input_name: batch})
            ordered_outputs = list(raw_outputs.values())
            for semantic, tensor, detail in zip(
                semantic_results.keys(),
                ordered_outputs,
                ordered_output_details,
            ):
                semantic_results[semantic].append(_dequantize_tflite_tensor(tensor, detail))
        return {key: np.concatenate(value, axis=0) for key, value in semantic_results.items()}

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
        input_detail = input_details[0]
        for input_array in inputs:
            batch = np.expand_dims(np.asarray(input_array, dtype=np.float32), axis=0)
            quantized_batch = quantize_input_batch(batch, input_detail)
            interpreter.set_tensor(int(input_detail["index"]), quantized_batch)
            interpreter.invoke()
            for semantic, output_name in semantic_to_name.items():
                assert output_name is not None
                detail = output_name_map[output_name]
                semantic_results[semantic].append(
                    _dequantize_tflite_tensor(
                        interpreter.get_tensor(int(detail["index"])),
                        detail,
                    )
                )
        return {key: np.concatenate(value, axis=0) for key, value in semantic_results.items()}

    # Final fallback: preserve the model output order and use the known tensor shapes.
    simcc_indices = [o["index"] for o in output_details if o["shape"][-1] == NUM_BINS]
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
    if len(key_map) != 5:
        raise RuntimeError(f"Could not resolve SimCC output order for {model_path}.")

    semantic_results = {key: [] for key in key_map}
    input_detail = input_details[0]
    for input_array in inputs:
        batch = np.expand_dims(np.asarray(input_array, dtype=np.float32), axis=0)
        quantized_batch = quantize_input_batch(batch, input_detail)
        interpreter.set_tensor(int(input_detail["index"]), quantized_batch)
        interpreter.invoke()
        for semantic, output_index in key_map.items():
            detail = next(detail for detail in output_details if detail["index"] == output_index)
            semantic_results[semantic].append(
                _dequantize_tflite_tensor(
                    interpreter.get_tensor(int(output_index)),
                    detail,
                )
            )
    return {key: np.concatenate(value, axis=0) for key, value in semantic_results.items()}


def _load_hard_case_examples(
    manifest_path: Path,
    hard_case_manifest: Path,
) -> list[Any]:
    """Load the hard-case examples from the manifest and filter by basename."""
    all_examples = load_clean_geometry_examples(manifest_path)
    with hard_case_manifest.open("r", encoding="utf-8") as handle:
        hard_names = {Path(row["image_path"]).name for row in csv.DictReader(handle)}
    hard_examples = [example for example in all_examples if Path(example.image_path).name in hard_names]
    hard_examples.sort(key=lambda example: float(example.temperature_c))
    return hard_examples


def _build_samples(
    examples: list[Any],
    *,
    input_size: int,
    heatmap_size: int,
    repo_root: Path,
) -> list[Any]:
    """Load the cropped inputs and ground-truth metadata for each hard case."""
    samples = [
        load_heatmap_sample(
            example,
            repo_root,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=2.0,
            jitter=None,
        )
        for example in examples
    ]
    return samples


def _write_csv(rows: list[HardCaseReplayRow], output_path: Path) -> None:
    """Write the per-sample replay results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HardCaseReplayRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_name": row.image_name,
                    "true_temperature_c": row.true_temperature_c,
                    "predicted_temperature_c": row.predicted_temperature_c,
                    "absolute_error_c": row.absolute_error_c,
                    "true_angle_deg": row.true_angle_deg,
                    "predicted_angle_deg": row.predicted_angle_deg,
                    "angle_error_deg": row.angle_error_deg,
                    "center_x_err_px": row.center_x_err_px,
                    "center_y_err_px": row.center_y_err_px,
                    "tip_x_err_px": row.tip_x_err_px,
                    "tip_y_err_px": row.tip_y_err_px,
                    "confidence": row.confidence,
                }
            )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Package the PTQ model and run the hard-case replay."""
    args = _parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.hard_case_manifest.is_file():
        raise FileNotFoundError(f"Hard-case manifest not found: {args.hard_case_manifest}")

    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage the PTQ artifact in the deployment directory so the package is self-contained.
    packaged_model_path = output_dir / PACKAGED_MODEL_NAME
    shutil.copy2(args.model_path, packaged_model_path)

    # Record the tensor contract that the packaged model exposes.
    contract = summarize_tflite_contract(packaged_model_path)
    contract_path = output_dir / CONTRACT_NAME
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    # Load the 19 hard cases using the same crop metadata as the training flow.
    hard_examples = _load_hard_case_examples(args.manifest, args.hard_case_manifest)
    samples = _build_samples(
        hard_examples,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        repo_root=REPO_ROOT,
    )
    print(f"[SIMCC] Hard cases loaded: {len(samples)}", flush=True)

    gt_cx = np.array([float(sample.metadata["center_x_224"]) for sample in samples], dtype=np.float32)
    gt_cy = np.array([float(sample.metadata["center_y_224"]) for sample in samples], dtype=np.float32)
    gt_tx = np.array([float(sample.metadata["tip_x_224"]) for sample in samples], dtype=np.float32)
    gt_ty = np.array([float(sample.metadata["tip_y_224"]) for sample in samples], dtype=np.float32)
    gt_angles = _angle_from_coords(gt_cx, gt_cy, gt_tx, gt_ty)
    true_temps = np.array([float(sample.metadata["temperature_c"]) for sample in samples], dtype=np.float32)
    names = [Path(sample.metadata["image_path"]).name for sample in samples]

    # Run the packaged TFLite model on each crop and decode the SimCC heads.
    predicted = _predict_tflite(packaged_model_path, [sample.crop_image for sample in samples])
    pred_cx = np.asarray([float(_soft_argmax_1d(row)) for row in predicted["cx"]], dtype=np.float32)
    pred_cy = np.asarray([float(_soft_argmax_1d(row)) for row in predicted["cy"]], dtype=np.float32)
    pred_tx = np.asarray([float(_soft_argmax_1d(row)) for row in predicted["tx"]], dtype=np.float32)
    pred_ty = np.asarray([float(_soft_argmax_1d(row)) for row in predicted["ty"]], dtype=np.float32)
    pred_conf = np.asarray([float(row.reshape(-1)[0]) for row in predicted["conf"]], dtype=np.float32)
    pred_angles = _angle_from_coords(pred_cx, pred_cy, pred_tx, pred_ty)

    # Convert the normalized coordinates back into 224-space pixel errors.
    cx_err = np.abs(pred_cx * CROP_INDEX_MAX - gt_cx)
    cy_err = np.abs(pred_cy * CROP_INDEX_MAX - gt_cy)
    tx_err = np.abs(pred_tx * CROP_INDEX_MAX - gt_tx)
    ty_err = np.abs(pred_ty * CROP_INDEX_MAX - gt_ty)
    angle_err = np.abs(pred_angles - gt_angles)
    angle_err = np.minimum(angle_err, 360.0 - angle_err)

    # Fit the same temperature calibration used during the manual replay.
    order = np.argsort(true_temps)
    gt_u = gt_angles[order].copy()
    pred_u = pred_angles[order].copy()
    for index in range(1, len(gt_u)):
        while gt_u[index] > gt_u[index - 1] + 180.0:
            gt_u[index] -= 360.0
        while gt_u[index] < gt_u[index - 1] - 180.0:
            gt_u[index] += 360.0
        while pred_u[index] > pred_u[index - 1] + 180.0:
            pred_u[index] -= 360.0
        while pred_u[index] < pred_u[index - 1] - 180.0:
            pred_u[index] += 360.0

    from numpy.polynomial import polynomial as P

    calibration = P.polyfit(gt_u, true_temps[order], 1)
    predicted_temps = calibration[0] + calibration[1] * pred_u
    predicted_temps = np.clip(predicted_temps, TEMP_MIN_C, TEMP_MAX_C)
    temp_errors = np.abs(predicted_temps - true_temps[order])

    rows: list[HardCaseReplayRow] = []
    for rank, index in enumerate(order):
        rows.append(
            HardCaseReplayRow(
                image_name=names[index],
                true_temperature_c=float(true_temps[index]),
                predicted_temperature_c=float(predicted_temps[rank]),
                absolute_error_c=float(temp_errors[rank]),
                true_angle_deg=float(gt_angles[index]),
                predicted_angle_deg=float(pred_angles[index]),
                angle_error_deg=float(angle_err[index]),
                center_x_err_px=float(cx_err[index]),
                center_y_err_px=float(cy_err[index]),
                tip_x_err_px=float(tx_err[index]),
                tip_y_err_px=float(ty_err[index]),
                confidence=float(pred_conf[index]),
            )
        )

    predictions_path = output_dir / PREDICTIONS_NAME
    _write_csv(rows, predictions_path)

    summary = {
        "model_name": MODEL_NAME,
        "source_model_path": str(args.model_path),
        "packaged_model_path": str(packaged_model_path),
        "tflite_sha256": _sha256(packaged_model_path),
        "tflite_bytes": int(packaged_model_path.stat().st_size),
        "hard_case_manifest": str(args.hard_case_manifest),
        "hard_case_count": int(len(samples)),
        "input_size": int(args.input_size),
        "heatmap_size": int(args.heatmap_size),
        "coordinate_metrics": {
            "center_mae_px": float(np.mean((cx_err + cy_err) / 2.0)),
            "tip_mae_px": float(np.mean((tx_err + ty_err) / 2.0)),
            "center_max_px": float(np.maximum(cx_err, cy_err).max()),
            "tip_max_px": float(np.maximum(tx_err, ty_err).max()),
        },
        "angle_metrics": {
            "mae_deg": float(angle_err.mean()),
            "median_deg": float(np.median(angle_err)),
            "worst_deg": float(angle_err.max()),
        },
        "temperature_calibration": {
            "slope": float(calibration[1]),
            "intercept": float(calibration[0]),
        },
        "temperature_metrics": {
            "mae_c": float(temp_errors.mean()),
            "rmse_c": float(np.sqrt(np.mean(temp_errors ** 2))),
            "worst_c": float(temp_errors.max()),
            "under_2c_pct": float((temp_errors < 2.0).mean() * 100.0),
            "under_5c_pct": float((temp_errors < 5.0).mean() * 100.0),
            "under_10c_pct": float((temp_errors < 10.0).mean() * 100.0),
        },
        "confidence": {
            "mean": float(pred_conf.mean()),
            "min": float(pred_conf.min()),
            "max": float(pred_conf.max()),
        },
        "worst_case": {
            "image_name": names[order[int(np.argmax(temp_errors))]],
            "absolute_error_c": float(temp_errors.max()),
        },
    }

    summary_path = output_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metadata = {
        "source_model_path": str(args.model_path),
        "packaged_model_path": str(packaged_model_path),
        "tflite_contract_path": str(contract_path),
        "hard_case_manifest": str(args.hard_case_manifest),
        "hard_case_count": int(len(samples)),
        "hard_case_replay_summary_path": str(summary_path),
        "hard_case_replay_predictions_path": str(predictions_path),
        "tflite_sha256": summary["tflite_sha256"],
        "tflite_bytes": summary["tflite_bytes"],
        "board_notes": (
            "Use the packaged TFLite file for the spatial SimCC replay path. "
            "The model keeps the 224x224 crop contract and decodes four 112-bin "
            "SimCC heads plus confidence."
        ),
        "replay_summary": summary,
    }
    metadata_path = output_dir / METADATA_NAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[SIMCC] Packaged model: {packaged_model_path}", flush=True)
    print(f"[SIMCC] Contract: {contract_path}", flush=True)
    print(f"[SIMCC] Summary: {summary_path}", flush=True)
    print(f"[SIMCC] Predictions: {predictions_path}", flush=True)
    print(f"[SIMCC] Metadata: {metadata_path}", flush=True)
    print(
        "[SIMCC] Hard-case metrics: "
        f"center_mae={summary['coordinate_metrics']['center_mae_px']:.3f}px "
        f"tip_mae={summary['coordinate_metrics']['tip_mae_px']:.3f}px "
        f"angle_mae={summary['angle_metrics']['mae_deg']:.4f}deg "
        f"temp_mae={summary['temperature_metrics']['mae_c']:.4f}C "
        f"temp_worst={summary['temperature_metrics']['worst_c']:.4f}C",
        flush=True,
    )


if __name__ == "__main__":
    main()
