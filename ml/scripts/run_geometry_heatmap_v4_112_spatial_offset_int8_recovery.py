#!/usr/bin/env python3
"""Phase 11F: Spatially-Aware Local Offset Head for INT8 Recovery.

Runs smoke + 6-candidate validation matrix for the local offset aux head.

Smoke: 5 epochs frozen, 0 unfrozen with default params.
Matrix: 6 candidates varying loss_weight, scale, sigma, tip_weight.

Usage:
  poetry run python3 scripts/run_geometry_heatmap_v4_112_spatial_offset_int8_recovery.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "ml" / "scripts"
TRAIN_SCRIPT = SCRIPTS_DIR / "train_geometry_heatmap_v4_112_quant_native.py"
EXPORT_SCRIPT = SCRIPTS_DIR / "export_geometry_heatmap_v4_112_int8.py"
REPLAY_SCRIPT = SCRIPTS_DIR / "eval_geometry_heatmap_v4_112_tflite_replay.py"

SOURCE_MODEL_PATH = REPO_ROOT / "ml" / "artifacts" / "training" / "geometry_heatmap_v4_112_int8_recovery" / "candidate_08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3" / "model_v4_112.keras"
MANIFEST_PATH = REPO_ROOT / "ml" / "data" / "geometry_reader_manifest_v2_clean.csv"
CALIBRATION_PATH = REPO_ROOT / "ml" / "artifacts" / "training" / "inner_dial_angle_calibration_v1" / "calibration_candidates.json"
THRESHOLDS_PATH = REPO_ROOT / "ml" / "artifacts" / "training" / "geometry_heatmap_v4_112_quant_native" / "v4_112_guardrail_thresholds.json"
DECODER_PATH = REPO_ROOT / "ml" / "artifacts" / "deployment" / "geometry_heatmap_v2_tflite_v2" / "selected_decode_method_corrected.json"

TRAIN_DIR = REPO_ROOT / "ml" / "artifacts" / "training" / "geometry_heatmap_v4_112_spatial_offset_int8_recovery"
TFLITE_DIR = REPO_ROOT / "ml" / "artifacts" / "deployment" / "geometry_heatmap_v4_112_tflite"
DEBUG_DIR = REPO_ROOT / "ml" / "debug" / "geometry_heatmap_v4_112_spatial_offset_int8_recovery"
REPORTS_DIR = REPO_ROOT / "ml" / "reports"

FROZEN_EPOCHS = 5
UNFROZEN_EPOCHS = 0
WARMUP_EPOCHS = 0

MATRIX_REPORT = REPORTS_DIR / "geometry_heatmap_v4_112_spatial_offset_int8_recovery_matrix.md"
DECISION_REPORT = REPORTS_DIR / "geometry_heatmap_v4_112_spatial_offset_int8_recovery_decision.md"


@dataclasses.dataclass(frozen=True)
class Candidate:
    label: str
    tag: str
    loss_weight: float
    scale_px: float
    sigma_px: float
    tip_weight: float


SMOKE_CANDIDATES = [
    Candidate("smoke_w20", "smoke_w20__lw20_s8_sig4_tw1", 20.0, 8.0, 4.0, 1.0),
]

MATRIX_CANDIDATES = [
    Candidate("01_w5_s8_s4_t1", "01_w5_s8_s4_t1__lw5_s8_sig4_tw1", 5.0, 8.0, 4.0, 1.0),
    Candidate("02_w20_s8_s4_t1", "02_w20_s8_s4_t1__lw20_s8_sig4_tw1", 20.0, 8.0, 4.0, 1.0),
    Candidate("03_w50_s8_s4_t1", "03_w50_s8_s4_t1__lw50_s8_sig4_tw1", 50.0, 8.0, 4.0, 1.0),
    Candidate("04_w20_s12_s5_t1", "04_w20_s12_s5_t1__lw20_s12_sig5_tw1", 20.0, 12.0, 5.0, 1.0),
    Candidate("05_w20_s8_s4_t2", "05_w20_s8_s4_t2__lw20_s8_sig4_tw2", 20.0, 8.0, 4.0, 2.0),
    Candidate("06_w50_s12_s5_t2", "06_w50_s12_s5_t2__lw50_s12_sig5_tw2", 50.0, 12.0, 5.0, 2.0),
]


def _run(cmd: list[str], label: str) -> int:
    print(f"[MATRIX] {' '.join(str(c) for c in cmd[:6])}...  (label={label})", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[MATRIX] FAIL: {label}", flush=True)
        print(proc.stderr[-3000:] if proc.stderr else "(no stderr)", flush=True)
    else:
        print(f"[MATRIX] OK: {label}", flush=True)
    return proc.returncode


def _run_candidate(
    candidate: Candidate,
    *,
    total_epochs: int = 5,
    warmup_epochs: int = 0,
) -> Path | None:
    candidate_dir = TRAIN_DIR / f"candidate_{candidate.tag}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--initialization-mode", "source_model",
        "--model-path", str(SOURCE_MODEL_PATH),
        "--manifest-path", str(MANIFEST_PATH),
        "--calibration-json-path", str(CALIBRATION_PATH),
        "--thresholds-path", str(THRESHOLDS_PATH),
        "--decoder-path", str(DECODER_PATH),
        "--output-dir", str(candidate_dir),
        "--batch-size", "8",
        "--frozen-epochs", str(total_epochs),
        "--unfrozen-epochs", "0",
        "--warmup-epochs", str(warmup_epochs),
        "--frozen-learning-rate", "1e-5",
        "--output-noise-stddev", "0.008",
        "--output-noise-ramp-epochs", "2",
        "--aux-head-type", "local_offset",
        "--local-offset-loss-weight", str(candidate.loss_weight),
        "--local-offset-scale-px", str(candidate.scale_px),
        "--local-offset-sigma-px", str(candidate.sigma_px),
        "--local-offset-tip-weight", str(candidate.tip_weight),
    ]

    rc = _run(train_cmd, f"train {candidate.label}")
    if rc != 0:
        return None

    # Determine which checkpoint to export
    frozen_best = candidate_dir / "model_v4_112_frozen_best.keras"
    model_to_export = frozen_best if frozen_best.exists() else candidate_dir / "model_v4_112.keras"

    export_dir = TFLITE_DIR / f"spatial_offset_recovery_{candidate.tag}"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_cmd = [
        sys.executable, str(EXPORT_SCRIPT),
        "--model-path", str(model_to_export),
        "--manifest-path", str(MANIFEST_PATH),
        "--calibration-json-path", str(CALIBRATION_PATH),
        "--thresholds-path", str(THRESHOLDS_PATH),
        "--decoder-path", str(DECODER_PATH),
        "--output-dir", str(export_dir),
        "--heatmap-size", "112",
        "--sigma-pixels", "2.5",
    ]

    rc = _run(export_cmd, f"export {candidate.label}")
    if rc != 0:
        return None

    float_tflite = export_dir / "model_v4_112_float32.tflite"
    int8_tflite = export_dir / "model_v4_112_int8.tflite"
    contract_path = export_dir / "tflite_tensor_contract.json"

    debug_dir = DEBUG_DIR / f"candidate_{candidate.tag}"
    debug_dir.mkdir(parents=True, exist_ok=True)

    replay_cmd = [
        sys.executable, str(REPLAY_SCRIPT),
        "--split", "val",
        "--model-path", str(model_to_export),
        "--float-tflite-path", str(float_tflite),
        "--int8-tflite-path", str(int8_tflite),
        "--contract-path", str(contract_path),
        "--manifest-path", str(MANIFEST_PATH),
        "--calibration-path", str(CALIBRATION_PATH),
        "--thresholds-path", str(THRESHOLDS_PATH),
        "--selected-decode-path", str(DECODER_PATH),
        "--predictions-path", str(debug_dir / "replay_predictions.csv"),
        "--summary-path", str(debug_dir / "val_summary.csv"),
        "--remaining-path", str(debug_dir / "remaining_worst.csv"),
        "--report-path", str(debug_dir / "replay_report.md"),
        "--keras-validation-report-path", str(debug_dir / "keras_val_report.md"),
        "--offset-scale-px", str(candidate.scale_px),
    ]

    rc = _run(replay_cmd, f"replay {candidate.label}")
    if rc != 0:
        return None

    return debug_dir


def _extract_gates(debug_dir: Path) -> dict[str, Any] | None:
    summary_path = debug_dir / "val_summary.csv"
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if len(lines) < 4:
        return None

    header = lines[0].strip().split(",")
    int8_line = lines[3].strip().split(",") if len(lines) > 3 else []
    if not int8_line:
        return None

    fields = dict(zip(header, int8_line))
    return {
        "accepted_mae_c": float(fields.get("accepted_mae_c", math.nan)),
        "acceptance_rate": float(fields.get("acceptance_rate", math.nan)),
        "worst_accepted_error_c": float(fields.get("worst_accepted_error_c", math.nan)),
        "accepted_gt20_failures": float(fields.get("accepted_gt20_failures", math.nan)),
        "temperature_delta_mean": float(fields.get("temperature_delta_mean", math.nan)),
        "tip_delta_mean": float(fields.get("tip_delta_mean", math.nan)),
        "center_delta_mean": float(fields.get("center_delta_mean", math.nan)),
    }


def _evaluate_gates(gates: dict[str, Any]) -> tuple[bool, str]:
    checks = []
    passed = True
    if gates["accepted_mae_c"] > 4.5:
        passed = False; checks.append("accepted_mae_c=FAIL")
    else:
        checks.append("accepted_mae_c=PASS")
    if gates["acceptance_rate"] < 0.65:
        passed = False; checks.append("acceptance_rate=FAIL")
    else:
        checks.append("acceptance_rate=PASS")
    if gates["worst_accepted_error_c"] >= 20.0:
        passed = False; checks.append("worst_accepted_error_c=FAIL")
    else:
        checks.append("worst_accepted_error_c=PASS")
    if gates["accepted_gt20_failures"] > 0:
        passed = False; checks.append("accepted_gt20_failures=FAIL")
    else:
        checks.append("accepted_gt20_failures=PASS")
    if gates["temperature_delta_mean"] > 1.0:
        passed = False; checks.append("temperature_delta_mean=FAIL")
    else:
        checks.append("temperature_delta_mean=PASS")
    return passed, " | ".join(checks)


def _smoke_check(candidate: Candidate, debug_dir: Path) -> bool:
    """Quick validation that the local offset pipeline works end-to-end."""
    print(f"\n[SMOKE] {candidate.label}", flush=True)
    gates = _extract_gates(debug_dir)
    if gates is None:
        print("[SMOKE] FAIL: no gates extracted", flush=True)
        return False

    print(f"[SMOKE] Keras-vs-INT8 drift: {gates['temperature_delta_mean']:.4f} C", flush=True)
    print(f"[SMOKE] INT8 MAE: {gates['accepted_mae_c']:.4f}, acceptance: {gates['acceptance_rate']:.4f}", flush=True)

    # Smoke checks: training completed, export succeeded, replay ran
    if math.isnan(gates["temperature_delta_mean"]):
        print("[SMOKE] FAIL: NaN drift", flush=True)
        return False

    print("[SMOKE] PASS: end-to-end pipeline working", flush=True)
    return True


def main() -> None:
    print("[MATRIX] Phase 11F: Spatially-Aware Local Offset Head", flush=True)
    start = time.time()

    # === SMOKE ===
    print("\n[MATRIX] === SMOKE PHASE ===", flush=True)
    smoke = SMOKE_CANDIDATES[0]
    smoke_dir = _run_candidate(smoke, total_epochs=FROZEN_EPOCHS, warmup_epochs=WARMUP_EPOCHS)
    if smoke_dir is None or not _smoke_check(smoke, smoke_dir):
        print("[MATRIX] SMOKE FAILED - aborting matrix", flush=True)
        return

    # === MATRIX ===
    print("\n[MATRIX] === MATRIX PHASE ===", flush=True)
    matrix_start = time.time()
    results: list[tuple[Candidate, dict[str, Any], bool]] = []
    passing = 0

    base_frozen = 15
    base_warmup = 3
    base_unfrozen = 5

    for i, candidate in enumerate(MATRIX_CANDIDATES):
        elapsed = (time.time() - matrix_start) / 60.0
        print(f"\n[MATRIX] Candidate: {candidate.label} (tag={candidate.tag})", flush=True)

        debug_dir = _run_candidate(
            candidate,
            total_epochs=base_frozen + base_unfrozen,
            warmup_epochs=base_warmup,
        )

        if debug_dir is None:
            print(f"[MATRIX] Candidate {candidate.label} FAILED (pipeline error)", flush=True)
            continue

        gates = _extract_gates(debug_dir)
        if gates is None:
            print(f"[MATRIX] Candidate {candidate.label} FAILED (no gates)", flush=True)
            continue

        passed, gate_str = _evaluate_gates(gates)
        results.append((candidate, gates, passed))
        if passed:
            passing += 1

        print(f"[MATRIX] Progress: {i + 1}/{len(MATRIX_CANDIDATES)} done, {passing} passing, {elapsed:.1f} min elapsed", flush=True)
        print(f"[MATRIX] Gates: {gate_str}", flush=True)

    # === WRITE REPORTS ===
    print("\n[MATRIX] === RESULTS ===", flush=True)
    total_time = (time.time() - start) / 60.0
    print(f"[MATRIX] Complete: {len(results)}/{len(MATRIX_CANDIDATES)} candidates, {passing} passing gates", flush=True)
    print(f"[MATRIX] Total time: {total_time:.1f} minutes", flush=True)

    matrix_lines = [
        "# Phase 11F Spatial Offset INT8 Recovery Matrix",
        "",
        f"- Candidates: {len(MATRIX_CANDIDATES)}",
        f"- Training schedule: {base_warmup} warmup + {base_frozen} frozen + {base_unfrozen} unfrozen = {base_warmup + base_frozen + base_unfrozen} epochs",
        f"- Source checkpoint: {SOURCE_MODEL_PATH}",
        "",
        "## Gates",
        "",
        "| Gate | Threshold |",
        "|------|-----------|",
        "| accepted_mae_c | 4.5 |",
        "| acceptance_rate | 0.65 |",
        "| worst_accepted_error_c | 20.0 |",
        "| accepted_gt20_failures | 0 |",
        "| temperature_delta_mean | 1.0 |",
        "",
        "## Results",
        "",
        "| # | Candidate | Weight | Scale | Sigma | Tip W | Status | MAE | Accept | Drift | Tip Drift | Gates |",
        "|---|-----------|--------|-------|-------|-------|--------|-----|--------|-------|-----------|-------|",
    ]

    sorted_results = sorted(results, key=lambda r: r[1].get("temperature_delta_mean", math.inf))
    champion: Candidate | None = None
    champion_gates: dict[str, Any] | None = None

    for result in sorted_results:
        cand, gates, passed = result
        gate_str = _evaluate_gates(gates)[1]
        status = "pass" if passed else "fail"
        if passed and champion is None:
            champion = cand
            champion_gates = gates
        matrix_lines.append(
            f"| {sorted_results.index(result) + 1} | {cand.label} | {cand.loss_weight} | "
            f"{cand.scale_px} | {cand.sigma_px} | {cand.tip_weight} | "
            f"{status} | {gates['accepted_mae_c']:.2f} | {gates['acceptance_rate']:.2%} | "
            f"{gates['temperature_delta_mean']:.4f} | {gates['tip_delta_mean']:.2f} | {gate_str} |"
        )

    matrix_lines.extend([
        "",
        "## Baseline Comparison",
        "",
        "| Baseline | INT8 Drift |",
        "|----------|------------|",
        "| Phase 10E original | ~1.99 C |",
        "| Phase 11B 08_tip_focus | 1.8405 C |",
        "| Phase 11E best (GAP aux) | 1.8423 C |",
    ])

    MATRIX_REPORT.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_REPORT.write_text("\n".join(matrix_lines), encoding="utf-8")
    print(f"[MATRIX] Wrote {MATRIX_REPORT}", flush=True)

    if champion is not None:
        print(f"\n[MATRIX] CHAMPION: {champion.label}", flush=True)
        print(f"[MATRIX] Drift: {champion_gates['temperature_delta_mean']:.4f} C", flush=True)
        decision_lines = [
            "# Phase 11F Spatial Offset INT8 Recovery Decision",
            "",
            "## Question",
            "",
            "Can a spatially-aware local offset head reduce Keras-vs-INT8 temperature drift below 1.0 C?",
            "",
            "## Answer",
            "",
            "**YES** -- a champion passed all validation gates.",
            "",
            f"Champion: {champion.label}",
            f"INT8 drift: {champion_gates['temperature_delta_mean']:.4f} C",
            f"INT8 MAE: {champion_gates['accepted_mae_c']:.2f} C",
            f"INT8 acceptance: {champion_gates['acceptance_rate']:.2%}",
            "",
            "### Next steps",
            "",
            "Decision A: spatial offset INT8 champion ready for sealed test replay.",
            "",
            "Do not run the test split automatically.",
        ]
        DECISION_REPORT.write_text("\n".join(decision_lines), encoding="utf-8")
        print(f"[MATRIX] Wrote {DECISION_REPORT}", flush=True)
    else:
        print("\n[MATRIX] No champion found.", flush=True)
        best = sorted_results[0] if sorted_results else None
        best_drift = best[1]["temperature_delta_mean"] if best else math.nan
        print(f"[MATRIX] Best drift: {best[0].label if best else 'N/A'} = {best_drift:.4f}", flush=True)

        decision_lines = [
            "# Phase 11F Spatial Offset INT8 Recovery Decision",
            "",
            "## Question",
            "",
            "Can a spatially-aware local offset head reduce Keras-vs-INT8 temperature drift below 1.0 C?",
            "",
            "## Answer",
            "",
            "**NO** -- no candidate passed all validation gates.",
            "",
            f"Best candidate: {best[0].label if best else 'N/A'}",
            f"INT8 drift: {best_drift:.4f} C",
            "",
            "### Analysis",
            "",
            "The local offset head did not reduce INT8 drift enough to pass the 1.0 C gate.",
            "",
            "### Next steps",
            "",
            "Decision B: local offset head failed.",
            "1. Compare against Phase 11B/11E drift baselines.",
            "2. Consider larger backbone, different decoder architecture, or per-tensor calibration.",
            "3. Or accept the current drift as the architectural limit.",
        ]
        DECISION_REPORT.write_text("\n".join(decision_lines), encoding="utf-8")
        print(f"[MATRIX] Wrote {DECISION_REPORT}", flush=True)

    print(f"\n[MATRIX] DONE. Total time: {total_time:.1f} min", flush=True)


if __name__ == "__main__":
    main()
