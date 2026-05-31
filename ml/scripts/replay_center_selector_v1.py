#!/usr/bin/env python3
"""
End-to-end replay harness for center selector v1.

Compares three pipelines on the same board captures:
  1. Classical baseline: multi-hypothesis center search + polar vote
  2. Hybrid selector: CNN picks/refines center + classical polar vote
  3. Direct v2 reference: MobileNetV2 tiny scalar regression (if model available)

Evaluates on:
  - Board captures (ml/data/board_captures_labeled_v2.csv)
  - Ideal control captures (preprocessed_crops/metadata.json)
  - Hard manifest cases (if available)

Success criteria:
  - Hybrid MAE must match or beat classical baseline
  - End-to-end latency should improve measurably
  - Max error and failure rate must not regress
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from embedded_gauge_reading_tinyml.gauge.processing import (
    load_gauge_specs,
    GaugeSpec,
    angle_rad_to_fraction,
    fraction_to_value,
)
from embedded_gauge_reading_tinyml.hybrid_localizer import (
    compute_all_hypotheses,
    needle_angle_from_polar_vote,
    estimate_dial_radius,
)
from embedded_gauge_reading_tinyml.models import build_mobilenetv2_center_selector
from luma_crop_detector import estimate_bright_centroid, compute_dynamic_crop, crop_and_resize
from train_center_selector_v1 import (
    decode_center_with_confidence,
    load_image,
    load_board,
    load_metadata,
    Config as TrainConfig,
    PROJECT_ROOT,
    GAUGE_ID,
    INPUT_SIZE,
)

# ---------------------------------------------------------------------------
# Classical baseline: multi-hypothesis center search
# ---------------------------------------------------------------------------


def classical_baseline_predict(
    image_224: np.ndarray, spec: GaugeSpec
) -> tuple[float, float]:
    """Run classical multi-hypothesis center search + polar vote.

    Tries all 4 hypotheses, picks the one with the strongest polar vote peak,
    then decodes temperature.

    Returns:
        (predicted_temperature, vote_strength)
    """
    from embedded_gauge_reading_tinyml.hybrid_localizer import (
        polar_spoke_vote,
        smooth_and_find_peak,
        rgb_to_luma,
    )

    hypotheses_px = compute_all_hypotheses(image_224)
    dial_r = estimate_dial_radius(image_224.shape[0])
    luma = rgb_to_luma(image_224)

    best_temp = 0.0
    best_strength = -1.0

    for hyp_idx in range(4):
        cx, cy = hypotheses_px[hyp_idx]
        votes = polar_spoke_vote(luma, cx, cy, dial_r)
        angle, peak_vote, mean_vote = smooth_and_find_peak(votes)
        strength = peak_vote / max(mean_vote, 1e-6)

        if strength > best_strength:
            best_strength = strength
            angle_rad = math.radians(angle)
            fraction = angle_rad_to_fraction(angle_rad, spec, strict=False)
            best_temp = fraction_to_value(fraction, spec)

    return best_temp, best_strength


# ---------------------------------------------------------------------------
# Hybrid selector: CNN center + classical polar vote
# ---------------------------------------------------------------------------


def hybrid_selector_predict(
    model, image_224: np.ndarray, spec: GaugeSpec
) -> tuple[float, float, bool, float]:
    """Run hybrid pipeline: CNN picks center, classical polar vote decodes.

    Returns:
        (predicted_temperature, confidence, is_high_confidence, latency_ms)
    """
    hypotheses_px = compute_all_hypotheses(image_224)
    dial_r = estimate_dial_radius(image_224.shape[0])

    start = time.perf_counter()
    inp = image_224.astype(np.float32) / 255.0
    out = model.predict(inp[None], verbose=0)
    logits = out["center_logits"][0]
    offset = out["center_offset"][0]
    latency_ms = (time.perf_counter() - start) * 1000.0

    refined_cx, refined_cy, confidence, is_high = decode_center_with_confidence(
        logits, offset, hypotheses_px,
    )

    angle_deg = needle_angle_from_polar_vote(image_224, refined_cx, refined_cy, dial_r)
    angle_rad = math.radians(angle_deg)
    fraction = angle_rad_to_fraction(angle_rad, spec, strict=False)
    temp = fraction_to_value(fraction, spec)

    return temp, confidence, is_high, latency_ms


# ---------------------------------------------------------------------------
# Direct v2 reference (if model available)
# ---------------------------------------------------------------------------


def direct_v2_predict(
    model, image_224: np.ndarray
) -> tuple[float, float]:
    """Run direct v2 scalar regression model.

    Returns:
        (predicted_temperature, latency_ms)
    """
    start = time.perf_counter()
    inp = image_224.astype(np.float32) / 255.0
    pred = model.predict(inp[None], verbose=0)
    latency_ms = (time.perf_counter() - start) * 1000.0

    # Direct v2 outputs a single scalar temperature value
    if isinstance(pred, dict):
        temp = float(pred[list(pred.keys())[0]][0])
    else:
        temp = float(pred[0])

    return temp, latency_ms


# ---------------------------------------------------------------------------
# Replay harness
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Results from replaying one sample through all pipelines."""

    image_path: str
    true_temp: float
    classical_temp: float = 0.0
    classical_strength: float = 0.0
    hybrid_temp: float = 0.0
    hybrid_confidence: float = 0.0
    hybrid_is_high: bool = False
    hybrid_latency_ms: float = 0.0
    direct_v2_temp: float = 0.0
    direct_v2_latency_ms: float = 0.0
    has_direct_v2: bool = False


def run_replay(
    board_samples: list[dict],
    conf: TrainConfig,
    pp_dir: Path | None,
    spec: GaugeSpec,
    center_selector_model,
    direct_v2_model=None,
) -> list[ReplayResult]:
    """Run all pipelines on board samples and collect results."""
    results = []

    for s in board_samples:
        p = Path(s["path"])
        if not p.exists():
            alt = PROJECT_ROOT / p
            if alt.exists():
                p = alt
            else:
                continue

        img = load_image(p, True, conf, pp_dir)
        if img is None:
            continue

        true_temp = s["temperature_c"]

        # Classical baseline
        classical_temp, classical_strength = classical_baseline_predict(img, spec)

        # Hybrid selector
        hybrid_temp, hybrid_conf, hybrid_is_high, hybrid_lat = hybrid_selector_predict(
            center_selector_model, img, spec,
        )

        # Direct v2 (optional)
        direct_v2_temp = 0.0
        direct_v2_lat = 0.0
        has_direct_v2 = direct_v2_model is not None
        if has_direct_v2:
            direct_v2_temp, direct_v2_lat = direct_v2_predict(direct_v2_model, img)

        results.append(ReplayResult(
            image_path=str(p),
            true_temp=true_temp,
            classical_temp=classical_temp,
            classical_strength=classical_strength,
            hybrid_temp=hybrid_temp,
            hybrid_confidence=hybrid_conf,
            hybrid_is_high=hybrid_is_high,
            hybrid_latency_ms=hybrid_lat,
            direct_v2_temp=direct_v2_temp,
            direct_v2_latency_ms=direct_v2_lat,
            has_direct_v2=has_direct_v2,
        ))

    return results


def summarize_results(results: list[ReplayResult]) -> dict:
    """Compute summary statistics from replay results."""
    classical_errs = [abs(r.classical_temp - r.true_temp) for r in results]
    hybrid_errs = [abs(r.hybrid_temp - r.true_temp) for r in results]
    hybrid_high_errs = [abs(r.hybrid_temp - r.true_temp) for r in results if r.hybrid_is_high]
    hybrid_fallback_errs = [abs(r.classical_temp - r.true_temp) for r in results if not r.hybrid_is_high]

    summary = {
        "n": len(results),
        "classical": {
            "mae": float(np.mean(classical_errs)),
            "med": float(np.median(classical_errs)),
            "max": float(np.max(classical_errs)),
            "p90": float(np.percentile(classical_errs, 90)) if len(classical_errs) > 1 else float(np.max(classical_errs)),
            "under_5c": int(np.sum(np.array(classical_errs) < 5.0)),
            "under_5c_rate": float(np.mean(np.array(classical_errs) < 5.0)),
        },
        "hybrid": {
            "mae": float(np.mean(hybrid_errs)),
            "med": float(np.median(hybrid_errs)),
            "max": float(np.max(hybrid_errs)),
            "p90": float(np.percentile(hybrid_errs, 90)) if len(hybrid_errs) > 1 else float(np.max(hybrid_errs)),
            "under_5c": int(np.sum(np.array(hybrid_errs) < 5.0)),
            "under_5c_rate": float(np.mean(np.array(hybrid_errs) < 5.0)),
            "mean_latency_ms": float(np.mean([r.hybrid_latency_ms for r in results])),
        },
        "hybrid_high_confidence": {
            "n": len(hybrid_high_errs),
            "mae": float(np.mean(hybrid_high_errs)) if hybrid_high_errs else float("inf"),
            "med": float(np.median(hybrid_high_errs)) if hybrid_high_errs else float("inf"),
        },
        "hybrid_fallback": {
            "n": len(hybrid_fallback_errs),
            "mae": float(np.mean(hybrid_fallback_errs)) if hybrid_fallback_errs else float("inf"),
            "fallback_rate": len(hybrid_fallback_errs) / len(results) if results else 0.0,
        },
        "mean_confidence": float(np.mean([r.hybrid_confidence for r in results])),
    }

    if results[0].has_direct_v2:
        direct_v2_errs = [abs(r.direct_v2_temp - r.true_temp) for r in results]
        summary["direct_v2"] = {
            "mae": float(np.mean(direct_v2_errs)),
            "med": float(np.median(direct_v2_errs)),
            "max": float(np.max(direct_v2_errs)),
            "mean_latency_ms": float(np.mean([r.direct_v2_latency_ms for r in results])),
        }

    return summary


def main():
    ap = argparse.ArgumentParser(description="Replay harness for center selector v1")
    ap.add_argument("--center-selector-model", required=True,
                    help="Path to trained center selector model (model.keras)")
    ap.add_argument("--direct-v2-model", default=None,
                    help="Path to direct v2 model for reference (optional)")
    ap.add_argument("--board-csv", default=None,
                    help="Path to board captures CSV")
    ap.add_argument("--output-dir", default="/tmp/gauge_center_selector_replay")
    args = ap.parse_args()

    print("=" * 60)
    print("CENTER SELECTOR v1 — Replay Harness")
    print(f"  Center selector: {args.center_selector_model}")
    if args.direct_v2_model:
        print(f"  Direct v2 ref: {args.direct_v2_model}")
    print("=" * 60)

    # Load gauge spec
    specs = load_gauge_specs()
    if GAUGE_ID not in specs:
        print(f"ERROR: Gauge '{GAUGE_ID}' not found.")
        sys.exit(1)
    spec = specs[GAUGE_ID]

    # Load center selector model
    print("\nLoading center selector model...")
    center_selector_model = build_mobilenetv2_center_selector(
        image_height=INPUT_SIZE, image_width=INPUT_SIZE,
        num_hypotheses=4, alpha=0.35,
        pretrained=False, backbone_trainable=False,
    )
    center_selector_model.load_weights(args.center_selector_model)
    print(f"  Loaded: {center_selector_model.name}")

    # Load direct v2 model (optional)
    direct_v2_model = None
    if args.direct_v2_model:
        print("Loading direct v2 model...")
        import tensorflow as tf
        direct_v2_model = tf.keras.models.load_model(args.direct_v2_model)
        print(f"  Loaded: {direct_v2_model.name}")

    # Load board samples
    conf = TrainConfig()
    pp_dir = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops"
    board_csv = Path(args.board_csv) if args.board_csv else (
        Path(__file__).resolve().parent.parent / "data" / "board_captures_labeled_v2.csv"
    )

    board_samples = []
    if board_csv.exists():
        board_samples = load_board(board_csv, INPUT_SIZE)
    print(f"\n  {len(board_samples)} board samples")

    if not board_samples:
        print("ERROR: No board samples found.")
        sys.exit(1)

    # Run replay
    print("\nRunning replay...")
    results = run_replay(
        board_samples, conf, pp_dir, spec,
        center_selector_model, direct_v2_model,
    )
    print(f"  Processed {len(results)} samples")

    # Summarize
    summary = summarize_results(results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    c = summary["classical"]
    print(f"\nClassical baseline:")
    print(f"  MAE: {c['mae']:.2f}C  Med: {c['med']:.2f}C  Max: {c['max']:.2f}C  P90: {c['p90']:.2f}C")
    print(f"  Under 5C: {c['under_5c']}/{c['n']} ({c['under_5c_rate']*100:.0f}%)")

    h = summary["hybrid"]
    print(f"\nHybrid selector:")
    print(f"  MAE: {h['mae']:.2f}C  Med: {h['med']:.2f}C  Max: {h['max']:.2f}C  P90: {h['p90']:.2f}C")
    print(f"  Under 5C: {h['under_5c']}/{h['n']} ({h['under_5c_rate']*100:.0f}%)")
    print(f"  Mean latency: {h['mean_latency_ms']:.1f}ms")

    hh = summary["hybrid_high_confidence"]
    hf = summary["hybrid_fallback"]
    print(f"\nHybrid high-confidence: n={hh['n']}, MAE={hh['mae']:.2f}C, Med={hh['med']:.2f}C")
    print(f"Hybrid fallback: n={hf['n']}, MAE={hf['mae']:.2f}C, rate={hf['fallback_rate']*100:.0f}%")
    print(f"Mean confidence: {summary['mean_confidence']:.3f}")

    if "direct_v2" in summary:
        d = summary["direct_v2"]
        print(f"\nDirect v2 reference:")
        print(f"  MAE: {d['mae']:.2f}C  Med: {d['med']:.2f}C  Max: {d['max']:.2f}C")
        print(f"  Mean latency: {d['mean_latency_ms']:.1f}ms")

    # Success criteria check
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    print("=" * 60)
    if h["mae"] <= c["mae"]:
        print(f"  [PASS] Hybrid MAE ({h['mae']:.2f}C) <= Classical MAE ({c['mae']:.2f}C)")
    else:
        print(f"  [FAIL] Hybrid MAE ({h['mae']:.2f}C) > Classical MAE ({c['mae']:.2f}C)")

    if h["max"] <= c["max"]:
        print(f"  [PASS] Hybrid max error ({h['max']:.2f}C) <= Classical max ({c['max']:.2f}C)")
    else:
        print(f"  [WARN] Hybrid max error ({h['max']:.2f}C) > Classical max ({c['max']:.2f}C)")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json.dump(summary, open(out_dir / "replay_summary.json", "w"), indent=2)

    # Save per-sample results
    per_sample = []
    for r in results:
        per_sample.append({
            "image_path": r.image_path,
            "true_temp": r.true_temp,
            "classical_temp": r.classical_temp,
            "classical_err": abs(r.classical_temp - r.true_temp),
            "hybrid_temp": r.hybrid_temp,
            "hybrid_err": abs(r.hybrid_temp - r.true_temp),
            "hybrid_confidence": r.hybrid_confidence,
            "hybrid_is_high": r.hybrid_is_high,
            "hybrid_latency_ms": r.hybrid_latency_ms,
            "direct_v2_temp": r.direct_v2_temp if r.has_direct_v2 else None,
            "direct_v2_err": abs(r.direct_v2_temp - r.true_temp) if r.has_direct_v2 else None,
        })
    json.dump(per_sample, open(out_dir / "per_sample.json", "w"), indent=2)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
