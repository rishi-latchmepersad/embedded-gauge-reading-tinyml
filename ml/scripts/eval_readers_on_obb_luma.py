#!/usr/bin/env python3
"""Compare multiple reader models on OBB+luma crops from the hard manifest.

Evaluates:
  - scalar   : MobileNetV2 backbone, direct temperature regression
  - v28      : Custom CNN, 7-channel polar input, 36-bin circular vote
  - errwt    : Custom CNN, 7-channel polar input, 224-bin topk-expectation

All use the same OBB+luma crop source so the comparison is fair.
Results include both raw predictions and post-calibration metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))

from embedded_gauge_reading_tinyml.board_pipeline import (
    decode_obb_crop_box,
    load_capture_image,
    load_model_session,
    refine_crop_with_luma_under_obb_constraint,
)
from embedded_gauge_reading_tinyml.firmware_preprocessing import (
    build_training_style_polar_vote_float32,
    decode_circular_vote_logits,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.polar_vote_v28 import build_polar_vote_v28_model

DEFAULT_MANIFEST: Path = PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv"
DEFAULT_CAPTURE_ROOT: Path = PROJECT_ROOT / "data" / "captured_images"
DEFAULT_OBB_MODEL: Path = PROJECT_ROOT / "artifacts" / "deployment" / "prod_model_v0.3_obb_int8" / "model_int8.tflite"
DEFAULT_SCALAR_MODEL: Path = PROJECT_ROOT / "artifacts" / "deployment" / "scalar_full_finetune_from_best_piecewise_calibrated_int8" / "model_int8.tflite"
DEFAULT_V28_WEIGHTS: Path = PROJECT_ROOT / "artifacts" / "training" / "polar_vote_circular_v28" / "best_weights.weights.h5"
DEFAULT_ERRWT_TFLITE: Path = PROJECT_ROOT / "artifacts" / "deployment" / "polar_vote_hardcases_errweighted_v1_int8" / "model_int8.tflite"

DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "tmp" / "reader_comparison_obb_luma"
DEFAULT_GAUGE_ID: str = "littlegood_home_temp_gauge_c"
DEFAULT_CALIBRATION: Path = PROJECT_ROOT / "artifacts" / "calibration" / "prodv0_3_obb_scalar_calibration.json"


@dataclass(frozen=True)
class EvalItem:
    image_path: Path
    value: float
    sample_weight: float


@dataclass(frozen=True)
class RowResult:
    image_path: str
    value: float
    scalar_pred: float | None
    v28_pred: float | None
    errwt_pred: float | None
    scalar_abs_err: float | None
    v28_abs_err: float | None
    errwt_abs_err: float | None
    crop_accepted: bool
    crop_source: str


def _resolve_path(raw: str) -> Path:
    p = Path(raw.replace("\\", "/"))
    return p if p.is_absolute() else REPO_ROOT / p


def _load_manifest(path: Path, capture_root: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ip = _resolve_path(row["image_path"])
            try:
                ip.resolve().relative_to(capture_root.resolve())
            except ValueError:
                continue
            items.append(EvalItem(
                image_path=ip,
                value=float(row["value"]),
                sample_weight=float(row.get("sample_weight", 1.0) or 1.0),
            ))
    return items


def _predict_tflite_scalar(model_session, crop_224: np.ndarray) -> float:
    """Run scalar TFLite on a 224x224x3 uint8 crop → temperature."""
    interpreter = model_session.model
    input_details = model_session.input_details
    output_details = model_session.output_details
    scale = float(input_details["quantization"][0])
    zp = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(crop_224.astype(np.float32) / scale + zp)
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
    interpreter.set_tensor(int(input_details["index"]), quantized[None, ...])
    interpreter.invoke()
    out = interpreter.get_tensor(int(output_details["index"]))[0, 0]
    out_scale = float(output_details["quantization"][0])
    out_zp = int(output_details["quantization"][1])
    return float(out_scale * (float(out) - out_zp))


def _predict_v28(exact_model, polar_tensor: np.ndarray) -> float:
    """Run V28 Keras model on 1×224×224×7 polar tensor → temperature."""
    logits = exact_model.predict(polar_tensor[None, ...], verbose=0)[0]
    return float(decode_circular_vote_logits(logits))


def _predict_tflite_polar(model_session, polar_tensor: np.ndarray,
                          min_val: float, max_val: float,
                          topk: int = 8) -> float:
    """Run polar-vote TFLite on 1×224×224×7 float [0,1] tensor → temperature."""
    interpreter = model_session.model
    input_details = model_session.input_details
    output_details = model_session.output_details

    in_scale = float(input_details["quantization"][0])
    in_zp = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(polar_tensor / in_scale + in_zp)
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
    interpreter.set_tensor(int(input_details["index"]), quantized[None, ...])
    interpreter.invoke()

    out_tensor = interpreter.get_tensor(int(output_details["index"]))[0]
    out_scale = float(output_details["quantization"][0])
    out_zp = int(output_details["quantization"][1])
    logits = out_scale * (out_tensor.astype(np.float32) - out_zp)

    # Top-k expectation decode over value bins
    num_bins = len(logits)
    bin_centers = np.linspace(min_val, max_val, num_bins, dtype=np.float32)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    topk_indices = np.argsort(probs)[-topk:]
    topk_probs = probs[topk_indices]
    topk_probs /= topk_probs.sum()
    return float(np.sum(bin_centers[topk_indices] * topk_probs))


def _apply_piecewise_calibration(raw_pred: float, cal_table: list[dict]) -> float:
    """Apply piecewise-linear calibration to a raw prediction."""
    for segment in cal_table:
        low, high = segment["raw_range"]
        if low <= raw_pred <= high:
            a, b = segment["coeffs"]
            return float(a * raw_pred + b)
    return raw_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reader models on OBB+luma crops")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--obb-model", type=Path, default=DEFAULT_OBB_MODEL)
    parser.add_argument("--scalar-model", type=Path, default=DEFAULT_SCALAR_MODEL)
    parser.add_argument("--v28-weights", type=Path, default=DEFAULT_V28_WEIGHTS)
    parser.add_argument("--errwt-tflite", type=Path, default=DEFAULT_ERRWT_TFLITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gauge-id", type=str, default=DEFAULT_GAUGE_ID)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--obb-crop-scale", type=float, default=1.2)
    parser.add_argument("--no-luma", action="store_true", help="Skip luma refinement, use raw OBB crops")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    items = _load_manifest(args.manifest, args.capture_root)
    if args.max_samples > 0:
        items = items[:args.max_samples]
    print(f"Loaded {len(items)} samples from {args.manifest}")

    gauge_spec = load_gauge_specs()[args.gauge_id]

    # Load OBB model
    obb_session = load_model_session(str(args.obb_model), "tflite")
    print(f"OBB model loaded: {args.obb_model}")

    # Load scalar model
    scalar_session = load_model_session(str(args.scalar_model), "tflite") if args.scalar_model.exists() else None
    print(f"Scalar model: {args.scalar_model} {'loaded' if scalar_session else 'SKIPPED'}")

    # Load V28 exact model
    exact_model = None
    if args.v28_weights.exists():
        exact_model = build_polar_vote_v28_model(polar_size=224, input_channels=7, base_filters=32, head_units=128, dropout=0.2)
        exact_model.load_weights(str(args.v28_weights))
        print(f"V28 weights loaded: {args.v28_weights}")
    else:
        print(f"V28 weights NOT FOUND: {args.v28_weights}")

    # Load errweighted TFLite
    errwt_session = load_model_session(str(args.errwt_tflite), "tflite") if args.errwt_tflite.exists() else None
    print(f"Errweighted TFLite: {args.errwt_tflite} {'loaded' if errwt_session else 'SKIPPED'}")

    # Load calibration
    cal_table: list[dict] = []
    if args.calibration.exists():
        cal_data = json.loads(args.calibration.read_text())
        cal_table = cal_data.get("piecewise_calibration", [])
        print(f"Calibration loaded: {args.calibration} ({len(cal_table)} segments)")
    else:
        print("No calibration found")

    results: list[RowResult] = []
    for idx, item in enumerate(items, 1):
        source_image, _ = load_capture_image(
            item.image_path,
            image_width=224,
            image_height=224,
        )

        # Run OBB
        full_batch = (source_image.astype(np.float32) / 255.0)[None, ...]
        obb_session.model.set_tensor(obb_session.input_details["index"],
            np.round(full_batch / obb_session.input_details["quantization"][0] + obb_session.input_details["quantization"][1]).clip(-128, 127).astype(np.int8))
        obb_session.model.invoke()
        obb_out = obb_session.model.get_tensor(obb_session.output_details["index"])[0]
        out_scale = float(obb_session.output_details["quantization"][0])
        out_zp = int(obb_session.output_details["quantization"][1])
        obb_params_float = out_scale * (obb_out.astype(np.float32) - out_zp)

        crop_dec = decode_obb_crop_box(
            obb_params_float,
            source_width=source_image.shape[1],
            source_height=source_image.shape[0],
            input_size=224,
            obb_crop_scale=args.obb_crop_scale,
        )

        crop_source = "obb"
        crop_xyxy = crop_dec.crop_box_xyxy

        if not args.no_luma and crop_dec.accepted:
            refined = refine_crop_with_luma_under_obb_constraint(source_image, crop_xyxy)
            crop_xyxy = refined
            crop_source = "obb_plus_luma"

        # Extract RGB crop for scalar model
        from embedded_gauge_reading_tinyml.board_pipeline import _crop_and_resize_frame
        rgb_crop = _crop_and_resize_frame(source_image, crop_xyxy, 224)

        scalar_pred = None
        v28_pred = None
        errwt_pred = None

        if scalar_session:
            scalar_pred = _predict_tflite_scalar(scalar_session, rgb_crop)

        # Build polar tensor once, share for both polar models
        if exact_model is not None or errwt_session is not None:
            polar_tensor = build_training_style_polar_vote_float32(
                source_image,
                crop_box_xyxy=crop_xyxy,
                output_dim=224,
                center_search_px=0,
                center_mode="image_center",
                gauge_spec=gauge_spec,
            )

            if exact_model is not None:
                v28_pred = _predict_v28(exact_model, polar_tensor)

            if errwt_session is not None:
                errwt_pred = _predict_tflite_polar(
                    errwt_session, polar_tensor,
                    min_val=-30.0, max_val=50.0, topk=8,
                )

        # Apply calibration to scalar only (V28/errwt have their own output range)
        if scalar_pred is not None and cal_table:
            calibrated = _apply_piecewise_calibration(scalar_pred, cal_table)
            print(f"  [{idx:03d}] {item.image_path.name}: scalar={scalar_pred:6.2f}→{calibrated:6.2f} v28={v28_pred or -999:6.2f} errwt={errwt_pred or -999:6.2f} gt={item.value:6.2f}")
            scalar_pred = calibrated
        else:
            print(f"  [{idx:03d}] {item.image_path.name}: scalar={scalar_pred or -999:6.2f} v28={v28_pred or -999:6.2f} errwt={errwt_pred or -999:6.2f} gt={item.value:6.2f}")

        results.append(RowResult(
            image_path=item.image_path.as_posix(),
            value=item.value,
            scalar_pred=scalar_pred,
            v28_pred=v28_pred,
            errwt_pred=errwt_pred,
            scalar_abs_err=abs(scalar_pred - item.value) if scalar_pred is not None else None,
            v28_abs_err=abs(v28_pred - item.value) if v28_pred is not None else None,
            errwt_abs_err=abs(errwt_pred - item.value) if errwt_pred is not None else None,
            crop_accepted=crop_dec.accepted,
            crop_source=crop_source,
        ))

    # Summarize
    def _print_metrics(name: str, pred_key: str, err_key: str):
        vals = [r for r in results if getattr(r, pred_key) is not None]
        if not vals:
            print(f"  {name}: NO DATA")
            return
        errs = np.array([getattr(r, err_key) for r in vals])
        print(f"  {name} ({len(vals)}/{len(results)}): "
              f"MAE={errs.mean():6.2f}°C  "
              f"RMSE={np.sqrt((errs**2).mean()):6.2f}°C  "
              f"median={np.median(errs):6.2f}°C  "
              f"p90={np.percentile(errs, 90):6.2f}°C  "
              f"max={errs.max():6.2f}°C")

    print(f"\n{'='*60}")
    print(f"Reader comparison on {len(results)} OBB+luma crops")
    print(f"{'='*60}")
    _print_metrics("Scalar (MobileNetV2)", "scalar_pred", "scalar_abs_err")
    _print_metrics("V28 (custom CNN, 36-bin vote)", "v28_pred", "v28_abs_err")
    _print_metrics("Errweighted (custom CNN, 224-bin vote)", "errwt_pred", "errwt_abs_err")

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    summary_path = args.output_dir / "summary.json"
    summary: dict[str, Any] = {
        "num_samples": len(results),
        "calibration": str(args.calibration),
        "obb_crop_scale": args.obb_crop_scale,
        "luma_refinement": not args.no_luma,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
