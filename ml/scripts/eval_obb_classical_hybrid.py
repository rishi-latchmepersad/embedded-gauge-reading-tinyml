#!/usr/bin/env python3
"""
OBB + Classical Hybrid: OBB localizer → crop → classical polar vote → temperature.

Compares three paths on board captures:
  1. Classical baseline — classical hypotheses + polar vote on the source image
  2. OBB + classical polar vote (hybrid, this experiment)
  3. OBB + scalar reader (current shipped NPU path, sanity reference)

Uses the same OBB+scalar pipeline as eval_readers_on_obb_luma.py so the
scalar results are directly comparable. The hybrid path replaces the scalar
reader with the classical polar vote on the OBB crop.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from embedded_gauge_reading_tinyml.board_pipeline import (
    decode_obb_crop_box,
    load_capture_image,
    load_model_session,
    _resize_with_pad_rgb_pil,
)
from embedded_gauge_reading_tinyml.gauge.processing import (
    load_gauge_specs,
    angle_rad_to_fraction,
    fraction_to_value,
)
from embedded_gauge_reading_tinyml.hybrid_localizer import (
    compute_all_hypotheses,
    estimate_dial_radius,
    polar_spoke_vote,
    smooth_and_find_peak,
    rgb_to_luma,
)

GAUGE_ID: str = "littlegood_home_temp_gauge_c"
INPUT_SIZE: int = 224

DEFAULT_MANIFEST: Path = PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new6.csv"
DEFAULT_OBB_MODEL: Path = (
    PROJECT_ROOT / "artifacts" / "deployment" / "prod_model_v0.3_obb_int8" / "model_int8.tflite"
)
DEFAULT_SCALAR_MODEL: Path = (
    PROJECT_ROOT / "artifacts" / "deployment"
    / "scalar_full_finetune_from_best_piecewise_calibrated_int8" / "model_int8.tflite"
)
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "tmp" / "obb_classical_hybrid"


# ---------------------------------------------------------------------------
# Path 1: Classical baseline — multi-hypothesis polar vote on source image
# ---------------------------------------------------------------------------

def classical_predict_on_image(source_image: np.ndarray, spec) -> float:
    """Run the 5-hypothesis classical polar vote on the source image."""
    hypotheses_px = compute_all_hypotheses(source_image)
    dial_r = estimate_dial_radius(source_image.shape[0])
    luma = rgb_to_luma(source_image)
    best_temp = 0.0
    best_strength = -1.0
    for i in range(hypotheses_px.shape[0]):
        cx, cy = hypotheses_px[i]
        votes = polar_spoke_vote(luma, cx, cy, dial_r)
        angle, peak_vote, mean_vote = smooth_and_find_peak(votes)
        strength = peak_vote / max(mean_vote, 1e-6)
        if strength > best_strength:
            best_strength = strength
            angle_rad = math.radians(angle)
            fraction = angle_rad_to_fraction(angle_rad, spec, strict=False)
            best_temp = fraction_to_value(fraction, spec)
    return best_temp


# ---------------------------------------------------------------------------
# Path 2 & 3 — OBB inference, crop extraction, classical/scalar readout
# ---------------------------------------------------------------------------

def _run_tflite_manual(
    session,
    batch_float: np.ndarray,
) -> np.ndarray:
    """Manual TFLite inference matching eval_readers_on_obb_luma.py exactly."""
    inp = session.input_details
    out = session.output_details
    q = np.round(batch_float / float(inp["quantization"][0]) + int(inp["quantization"][1]))
    q = np.clip(q, -128, 127).astype(np.int8)
    session.model.set_tensor(int(inp["index"]), q)
    session.model.invoke()
    raw = session.model.get_tensor(int(out["index"]))[0]
    scale = float(out["quantization"][0])
    zp = int(out["quantization"][1])
    return np.asarray(scale * (raw.astype(np.float32) - zp), dtype=np.float32)


def run_obb(source_image: np.ndarray, obb_session) -> np.ndarray:
    """Run OBB TFLite, return 6 obb_params in float [0,1] normalized space."""
    batch = (source_image.astype(np.float32) / 255.0)[None, ...]
    return _run_tflite_manual(obb_session, batch).reshape(-1)


def extract_obb_crop(
    source_image: np.ndarray,
    obb_params: np.ndarray,
    *,
    input_size: int = INPUT_SIZE,
    obb_crop_scale: float = 1.2,
) -> tuple[np.ndarray, bool]:
    """Decode OBB params and extract a padded 224x224 crop.

    Follows eval_readers_on_obb_luma.py: always returns the crop, with
    the boolean indicating whether the OBB crop was within the training
    window (``accepted``). A rejected crop (strict) is still used.
    """
    h, w = source_image.shape[:2]
    dec = decode_obb_crop_box(
        obb_params,
        source_width=w,
        source_height=h,
        input_size=input_size,
        obb_crop_scale=obb_crop_scale,
    )
    crop = _resize_with_pad_rgb_pil(source_image, dec.crop_box_xyxy, image_size=input_size)
    return crop, dec.accepted


def classical_predict_on_crop(crop_224: np.ndarray, spec) -> float:
    """Classical polar vote on the OBB-extracted 224x224 crop."""
    return classical_predict_on_image(crop_224, spec)


def scalar_predict_on_crop(crop_224: np.ndarray, scalar_session) -> float:
    """Run scalar TFLite on a 224x224 uint8 crop → temperature."""
    batch = crop_224.astype(np.float32)[None, ...] / 255.0
    return float(_run_tflite_manual(scalar_session, batch).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ip = row["image_path"].replace("\\", "/")
            p = Path(ip)
            if not p.is_absolute():
                p = REPO_ROOT / ip
            items.append({
                "image_path": p,
                "value": float(row["value"]),
            })
    return items


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize(errs: list[float]) -> dict:
    e = np.array(errs) if errs else np.array([float("inf")])
    return {
        "mae": float(np.mean(e)),
        "med": float(np.median(e)),
        "max": float(np.max(e)),
        "p90": float(np.percentile(e, 90)) if len(e) > 1 else float(np.max(e)),
        "under_5c": int(np.sum(e < 5.0)),
        "under_5c_rate": float(np.mean(e < 5.0)),
        "n": int(len(e)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="OBB + classical polar hybrid evaluation")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--obb-model", type=Path, default=DEFAULT_OBB_MODEL)
    ap.add_argument("--scalar-model", type=Path, default=DEFAULT_SCALAR_MODEL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--gauge-id", type=str, default=GAUGE_ID)
    ap.add_argument("--obb-crop-scale", type=float, default=1.2)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    print("=" * 65)
    print("OBB + CLASSICAL HYBRID — Board Replay")
    print("=" * 65)

    # Load gauge spec
    specs = load_gauge_specs()
    if args.gauge_id not in specs:
        print(f"ERROR: Gauge '{args.gauge_id}' not found")
        sys.exit(1)
    spec = specs[args.gauge_id]
    print(f"  Gauge: {spec.gauge_id} ({spec.min_value}–{spec.max_value} {spec.units})")

    # Load models
    if not args.obb_model.exists():
        print(f"ERROR: OBB model not found at {args.obb_model}")
        sys.exit(1)
    obb_session = load_model_session(str(args.obb_model), "tflite")
    print(f"  OBB: {args.obb_model}")

    scalar_session = None
    if args.scalar_model and args.scalar_model.exists():
        scalar_session = load_model_session(str(args.scalar_model), "tflite")
        print(f"  Scalar: {args.scalar_model}")
    else:
        print("  Scalar: SKIPPED")

    # Load manifest
    samples = load_manifest(args.manifest)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"  Samples: {len(samples)} from {args.manifest}")

    # Run evaluation
    per_sample: list[dict] = []
    classical_errs: list[float] = []
    obb_classical_errs: list[float] = []
    obb_scalar_errs: list[float] = []
    obb_accepted = 0
    obb_latencies: list[float] = []

    for idx, s in enumerate(samples):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Processing {idx+1}/{len(samples)}...", flush=True)

        if not s["image_path"].exists():
            print(f"  WARNING: image not found: {s['image_path']}")
            continue

        source_image, _ = load_capture_image(s["image_path"])
        true_temp = s["value"]

        # Path 1: Classical baseline
        classical_temp = classical_predict_on_image(source_image, spec)
        classical_errs.append(abs(classical_temp - true_temp))

        # Run OBB
        obb_start = time.perf_counter()
        obb_params = run_obb(source_image, obb_session)
        obb_latency = (time.perf_counter() - obb_start) * 1000.0
        obb_latencies.append(obb_latency)

        # Decode OBB crop (always use, even if outside training window)
        crop_224, obb_accepted_flag = extract_obb_crop(
            source_image, obb_params, obb_crop_scale=args.obb_crop_scale,
        )
        if obb_accepted_flag:
            obb_accepted += 1

        row: dict = {
            "image_path": str(s["image_path"]),
            "true_temp": true_temp,
            "classical_temp": classical_temp,
            "obb_classical_temp": 0.0,
            "obb_scalar_temp": 0.0,
            "obb_crop_accepted": obb_accepted_flag,
            "obb_latency_ms": round(obb_latency, 2),
        }

        # Path 2: OBB → classical polar vote
        obb_classical_temp = classical_predict_on_crop(crop_224, spec)
        obb_classical_errs.append(abs(obb_classical_temp - true_temp))
        row["obb_classical_temp"] = obb_classical_temp

        # Path 3: OBB → scalar reader
        if scalar_session:
            scalar_temp = scalar_predict_on_crop(crop_224, scalar_session)
            obb_scalar_errs.append(abs(scalar_temp - true_temp))
            row["obb_scalar_temp"] = scalar_temp

        per_sample.append(row)

    # Summarize
    summary: dict = {
        "total": len(per_sample),
        "obb_accepted": obb_accepted,
        "classical": summarize(classical_errs),
        "obb_classical": summarize(obb_classical_errs),
    }
    if scalar_session:
        summary["obb_scalar"] = summarize(obb_scalar_errs)

    # Print results
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    def show(label: str, key: str):
        d = summary[key]
        print(f"\n{label}:")
        print(f"  MAE: {d['mae']:.2f}C  Med: {d['med']:.2f}C  Max: {d['max']:.2f}C  P90: {d['p90']:.2f}C")
        print(f"  Under 5C: {d['under_5c']}/{d['n']} ({d['under_5c_rate']*100:.0f}%)")

    show("Classical baseline (source image)", "classical")
    show("OBB → classical polar vote (hybrid)", "obb_classical")
    if scalar_session:
        show("OBB → scalar reader (shipped ref)", "obb_scalar")

    print(f"\nOBB accepted: {obb_accepted}/{len(per_sample)}")
    if obb_latencies:
        print(f"Mean OBB latency: {np.mean(obb_latencies):.1f}ms")

    # Success criteria
    h = summary["obb_classical"]
    c = summary["classical"]
    print("\n" + "-" * 65)
    print("SUCCESS CRITERIA")
    print("-" * 65)
    if h["mae"] <= c["mae"]:
        print(f"  [PASS] Hybrid MAE ({h['mae']:.2f}C) <= Classical MAE ({c['mae']:.2f}C)")
    else:
        print(f"  [FAIL] Hybrid MAE ({h['mae']:.2f}C) > Classical MAE ({c['mae']:.2f}C)")
    if h["max"] <= c["max"]:
        print(f"  [PASS] Hybrid max ({h['max']:.2f}C) <= Classical max ({c['max']:.2f}C)")
    else:
        print(f"  [WARN] Hybrid max ({h['max']:.2f}C) > Classical max ({c['max']:.2f}C)")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.output_dir / "summary.json", "w"), indent=2)
    json.dump(per_sample, open(args.output_dir / "per_sample.json", "w"), indent=2)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
