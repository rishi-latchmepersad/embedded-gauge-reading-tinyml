#!/usr/bin/env python3
"""Run a bounded INT8 recovery matrix for the auxiliary coordinate head.

12 candidates, each trained with the aux_coords head from Phase 11D, varying
aux weight, head size, and loss type.  Produces a matrix summary report and
a decision report ranking candidates by gate pass -> acceptance -> MAE -> drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Shared paths
MANIFEST_PATH = REPO_ROOT / "ml/data/geometry_reader_manifest_v2_clean.csv"
CALIBRATION_PATH = REPO_ROOT / "ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json"
THRESHOLDS_PATH = REPO_ROOT / "ml/artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json"
DECODER_PATH = REPO_ROOT / "ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json"
INIT_MODEL_PATH = REPO_ROOT / "ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras"

# Output directory roots
TRAINING_BASE = REPO_ROOT / "ml/artifacts/training/geometry_heatmap_v4_112_aux_int8_recovery"
DEPLOY_BASE = REPO_ROOT / "ml/artifacts/deployment/geometry_heatmap_v4_112_tflite"
REPORT_DIR = REPO_ROOT / "ml/reports"
DEBUG_BASE = REPO_ROOT / "ml/debug/geometry_heatmap_v4_112_aux_int8_recovery"

# Reports
MATRIX_REPORT_PATH = REPORT_DIR / "geometry_heatmap_v4_112_aux_int8_recovery_matrix.md"
DECISION_REPORT_PATH = REPORT_DIR / "geometry_heatmap_v4_112_aux_int8_recovery_decision.md"

# Scripts
TRAIN_SCRIPT = REPO_ROOT / "ml/scripts/train_geometry_heatmap_v4_112_quant_native.py"
EXPORT_SCRIPT = REPO_ROOT / "ml/scripts/export_geometry_heatmap_v4_112_int8.py"
REPLAY_SCRIPT = REPO_ROOT / "ml/scripts/eval_geometry_heatmap_v4_112_tflite_replay.py"

# Validation gates
GATES: dict[str, float] = {
    "accepted_mae_c": 4.5,
    "acceptance_rate": 0.65,
    "worst_accepted_error_c": 20.0,
    "accepted_gt20_failures": 0,
    "temperature_delta_mean": 1.0,
}

# Training params (shortened schedule for matrix)
WARMUP_EPOCHS = 3
FROZEN_EPOCHS = 15
UNFROZEN_EPOCHS = 5
FROZEN_LR = 1e-5
UNFROZEN_LR = 5e-6
BATCH_SIZE = 16
HEATMAP_SIZE = 112
SIGMA_PIXELS = 2.5

# Baseline 08_tip_focus params (fixed across all candidates)
PEAK_TARGET = 0.25
PEAK_SHAPE_CENTER_WEIGHT = 0.05
PEAK_SHAPE_TIP_WEIGHT = 0.20
CONFIDENCE_FLOOR_WEIGHT = 0.05


# ---------------------------------------------------------------------------
# Candidate definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One cell in the aux recovery matrix."""

    name: str
    aux_coord_weight: float
    aux_head_size: str  # "small" or "large"
    aux_loss_type: str  # "mse" or "huber"

    @property
    def tag(self) -> str:
        return (
            f"{self.name}__aw{self.aux_coord_weight:.1f}"
            f"_hs{self.aux_head_size}"
            f"_lt{self.aux_loss_type}"
        )

    @property
    def training_dir(self) -> Path:
        return TRAINING_BASE / f"candidate_{self.tag}"

    @property
    def deploy_dir(self) -> Path:
        return DEPLOY_BASE / f"aux_recovery_{self.tag}"

    @property
    def debug_dir(self) -> Path:
        return DEBUG_BASE / f"candidate_{self.tag}"

    @property
    def model_path(self) -> Path:
        return self.training_dir / "model_v4_112.keras"

    @property
    def float_tflite_path(self) -> Path:
        return self.deploy_dir / "model_v4_112_float32.tflite"

    @property
    def int8_tflite_path(self) -> Path:
        return self.deploy_dir / "model_v4_112_int8.tflite"

    @property
    def contract_path(self) -> Path:
        return self.deploy_dir / "tflite_tensor_contract.json"

    @property
    def replay_summary_path(self) -> Path:
        return self.debug_dir / "val_summary.csv"

    @property
    def replay_report_path(self) -> Path:
        return self.debug_dir / "replay_report.md"


# 12 candidates: 3 weights x 2 head sizes x 2 loss types
CANDIDATES: list[Candidate] = [
    # Small head, MSE
    Candidate("01_w02_small_mse", 0.2, "small", "mse"),
    Candidate("02_w05_small_mse", 0.5, "small", "mse"),
    Candidate("03_w10_small_mse", 1.0, "small", "mse"),
    # Small head, Huber
    Candidate("04_w02_small_huber", 0.2, "small", "huber"),
    Candidate("05_w05_small_huber", 0.5, "small", "huber"),
    Candidate("06_w10_small_huber", 1.0, "small", "huber"),
    # Large head, MSE
    Candidate("07_w02_large_mse", 0.2, "large", "mse"),
    Candidate("08_w05_large_mse", 0.5, "large", "mse"),
    Candidate("09_w10_large_mse", 1.0, "large", "mse"),
    # Large head, Huber
    Candidate("10_w02_large_huber", 0.2, "large", "huber"),
    Candidate("11_w05_large_huber", 0.5, "large", "huber"),
    Candidate("12_w10_large_huber", 1.0, "large", "huber"),
]


# ---------------------------------------------------------------------------
# Structured result container
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """Metrics collected for one candidate after replay."""
    candidate: Candidate
    status: str = "unknown"
    gates: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    drift: dict[str, float] = field(default_factory=dict)
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout_minutes: int = 480,
    label: str = "",
) -> subprocess.CompletedProcess:
    """Run a command with streaming output and a long timeout."""
    print(f"\n{'='*72}", flush=True)
    print(f"[MATRIX] {' '.join(cmd[:4])}...  (label={label})", flush=True)
    print(f"{'='*72}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=False,
        timeout=timeout_minutes * 60,
        check=False,
    )
    if result.returncode != 0:
        print(f"[MATRIX] FAILED (rc={result.returncode}): {label}", flush=True)
    else:
        print(f"[MATRIX] OK: {label}", flush=True)
    return result


def _parse_summary_csv(path: Path, *, model_type: str | None = None) -> dict[str, float] | None:
    """Read a summary CSV and return a float dict for the given model_type row."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_model = str(row.get("model_type", "")).strip()
            if model_type is not None and row_model != model_type:
                continue
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (ValueError, TypeError):
                    pass
            return parsed
    return None


def _check_gates(metrics: dict[str, float], drift: dict[str, float]) -> dict[str, bool]:
    """Evaluate all gates. Returns {gate_name: passed_bool}."""
    gates: dict[str, bool] = {}
    for gate_name, threshold in GATES.items():
        value = drift.get(gate_name) if gate_name in drift else metrics.get(gate_name)
        if value is None:
            gates[gate_name] = False
        elif gate_name == "accepted_gt20_failures":
            gates[gate_name] = float(value) <= threshold
        elif gate_name == "acceptance_rate":
            gates[gate_name] = float(value) >= threshold
        elif gate_name == "accepted_mae_c":
            gates[gate_name] = float(value) <= threshold
        elif gate_name == "worst_accepted_error_c":
            gates[gate_name] = float(value) < threshold
        elif gate_name == "temperature_delta_mean":
            gates[gate_name] = float(value) <= threshold
        else:
            gates[gate_name] = float(value) <= threshold
    return gates


def _all_gates_pass(gates: dict[str, bool]) -> bool:
    return all(gates.values())


def _gate_summary_line(gates: dict[str, bool]) -> str:
    parts = []
    for name, passed in gates.items():
        icon = "PASS" if passed else "FAIL"
        parts.append(f"{name}={icon}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Per-candidate steps
# ---------------------------------------------------------------------------

def _run_training(candidate: Candidate, *, quick: bool, force: bool) -> bool:
    """Run training for one candidate. Returns True on success."""
    train_dir = candidate.training_dir
    if not force and train_dir.exists() and (train_dir / "model_v4_112.keras").exists():
        print(f"[MATRIX] Skipping training for {candidate.tag}: output exists", flush=True)
        return True

    train_dir.mkdir(parents=True, exist_ok=True)
    frozen = 5 if quick else FROZEN_EPOCHS
    unfrozen = 0 if quick else UNFROZEN_EPOCHS

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--initialization-mode", "source_model",
        "--model-path", str(INIT_MODEL_PATH),
        "--manifest-path", str(MANIFEST_PATH),
        "--calibration-json-path", str(CALIBRATION_PATH),
        "--thresholds-path", str(THRESHOLDS_PATH),
        "--decoder-path", str(DECODER_PATH),
        "--output-dir", str(train_dir),
        "--batch-size", str(BATCH_SIZE),
        "--frozen-epochs", str(frozen),
        "--unfrozen-epochs", str(unfrozen),
        "--frozen-learning-rate", str(FROZEN_LR),
        "--unfrozen-learning-rate", str(UNFROZEN_LR),
        "--heatmap-size", str(HEATMAP_SIZE),
        "--sigma-pixels", str(SIGMA_PIXELS),
        "--peak-target", str(PEAK_TARGET),
        "--peak-shape-center-weight", str(PEAK_SHAPE_CENTER_WEIGHT),
        "--peak-shape-tip-weight", str(PEAK_SHAPE_TIP_WEIGHT),
        "--confidence-floor-weight", str(CONFIDENCE_FLOOR_WEIGHT),
        "--warmup-epochs", str(WARMUP_EPOCHS),
        "--include-aux-coords",
        "--aux-coord-weight", str(candidate.aux_coord_weight),
        "--aux-head-size", candidate.aux_head_size,
        "--aux-loss-type", candidate.aux_loss_type,
    ]
    result = _run_subprocess(cmd, label=f"train {candidate.tag}")
    if result.returncode != 0:
        print(f"[MATRIX] Training failed for {candidate.tag}", flush=True)
        return False

    if not candidate.model_path.exists():
        print(f"[MATRIX] Training did not produce model at {candidate.model_path}", flush=True)
        return False
    return True


def _run_export(candidate: Candidate, *, force: bool) -> bool:
    """Export Keras model to float32 + INT8 TFLite. Returns True on success."""
    deploy_dir = candidate.deploy_dir
    if not force and candidate.int8_tflite_path.exists() and candidate.float_tflite_path.exists():
        print(f"[MATRIX] Skipping export for {candidate.tag}: TFLite files exist", flush=True)
        return True

    if not candidate.model_path.exists():
        print(f"[MATRIX] Cannot export {candidate.tag}: model missing at {candidate.model_path}", flush=True)
        return False

    deploy_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(EXPORT_SCRIPT),
        "--model-path", str(candidate.model_path),
        "--manifest-path", str(MANIFEST_PATH),
        "--calibration-json-path", str(CALIBRATION_PATH),
        "--selected-thresholds-path", str(THRESHOLDS_PATH),
        "--decoder-path", str(DECODER_PATH),
        "--output-dir", str(deploy_dir),
        "--input-size", "224",
        "--heatmap-size", str(HEATMAP_SIZE),
        "--sigma-pixels", str(SIGMA_PIXELS),
    ]
    result = _run_subprocess(cmd, label=f"export {candidate.tag}")
    if result.returncode != 0:
        print(f"[MATRIX] Export failed for {candidate.tag}", flush=True)
        return False

    if not candidate.int8_tflite_path.exists():
        print(f"[MATRIX] Export did not produce INT8 TFLite at {candidate.int8_tflite_path}", flush=True)
        return False
    return True


def _run_replay(candidate: Candidate, *, force: bool) -> CandidateResult:
    """Replay candidate on val, parse metrics, return CandidateResult."""
    result = CandidateResult(candidate=candidate)

    if not force and candidate.replay_summary_path.exists():
        print(f"[MATRIX] Re-using cached replay for {candidate.tag}", flush=True)
        int8_metrics = _parse_summary_csv(candidate.replay_summary_path, model_type="tflite_int8")
        keras_metrics = _parse_summary_csv(candidate.replay_summary_path, model_type="keras_v3")
        if int8_metrics is not None:
            result.metrics = int8_metrics
            result.drift = {k: v for k, v in int8_metrics.items() if "delta" in k or "disagreement" in k}
            if keras_metrics is not None:
                result.drift["keras_accepted_mae_c"] = keras_metrics.get("accepted_mae_c", math.nan)
            result.gates = _check_gates(result.metrics, result.drift)
            result.status = "pass" if _all_gates_pass(result.gates) else "fail"
            return result

    if not candidate.int8_tflite_path.exists():
        result.status = "skip_replay"
        result.error = f"INT8 TFLite missing at {candidate.int8_tflite_path}"
        return result

    debug_dir = candidate.debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(REPLAY_SCRIPT),
        "--split", "val",
        "--batch-size", "16",
        "--manifest-path", str(MANIFEST_PATH),
        "--model-path", str(candidate.model_path),
        "--float-tflite-path", str(candidate.float_tflite_path),
        "--int8-tflite-path", str(candidate.int8_tflite_path),
        "--contract-path", str(candidate.contract_path),
        "--calibration-path", str(CALIBRATION_PATH),
        "--thresholds-path", str(THRESHOLDS_PATH),
        "--selected-decode-path", str(DECODER_PATH),
        "--predictions-path", str(debug_dir / "val_predictions.csv"),
        "--summary-path", str(candidate.replay_summary_path),
        "--remaining-path", str(debug_dir / "val_remaining_worst.csv"),
        "--report-path", str(candidate.replay_report_path),
        "--keras-validation-report-path", str(debug_dir / "keras_validation_report.md"),
    ]
    result_proc = _run_subprocess(cmd, label=f"replay {candidate.tag}", timeout_minutes=120)
    if result_proc.returncode != 0:
        result.status = "skip_replay"
        result.error = f"Replay failed for {candidate.tag}"
        return result

    int8_metrics = _parse_summary_csv(candidate.replay_summary_path, model_type="tflite_int8")
    keras_metrics = _parse_summary_csv(candidate.replay_summary_path, model_type="keras_v3")
    if int8_metrics is None:
        result.status = "skip_replay"
        result.error = f"No INT8 summary found at {candidate.replay_summary_path}"
        return result

    result.metrics = int8_metrics
    result.drift = {k: v for k, v in int8_metrics.items() if "delta" in k or "disagreement" in k}
    if keras_metrics is not None:
        result.drift["keras_accepted_mae_c"] = keras_metrics.get("accepted_mae_c", math.nan)
    result.gates = _check_gates(result.metrics, result.drift)
    result.status = "pass" if _all_gates_pass(result.gates) else "fail"
    return result


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_matrix_report(results: list[CandidateResult]) -> None:
    """Write the matrix summary markdown report."""
    lines = [
        "# Phase 11E Aux INT8 Recovery Matrix",
        "",
        f"- Candidates: {len(results)}",
        f"- Training schedule: {WARMUP_EPOCHS} warmup + {FROZEN_EPOCHS} frozen + {UNFROZEN_EPOCHS} unfrozen = {WARMUP_EPOCHS + FROZEN_EPOCHS + UNFROZEN_EPOCHS} epochs",
        f"- Baseline config: 08_tip_focus (peak_target={PEAK_TARGET}, tip_weight={PEAK_SHAPE_TIP_WEIGHT})",
        f"- Source checkpoint: {INIT_MODEL_PATH.name}",
        "",
        "## Gates",
        "",
        "| Gate | Threshold |",
        "|------|-----------|",
    ]
    for name, threshold in GATES.items():
        lines.append(f"| {name} | {threshold} |")

    lines += [
        "",
        "## Results",
        "",
        "| # | Candidate | Weight | Head | Loss | Status | MAE | Accept | Drift | Tip Drift | Gates |",
        "|---|-----------|--------|------|------|--------|-----|--------|-------|-----------|-------|",
    ]

    for i, r in enumerate(results, 1):
        c = r.candidate
        mae = r.metrics.get("accepted_mae_c", math.nan)
        accept = r.metrics.get("acceptance_rate", math.nan)
        drift = r.drift.get("temperature_delta_mean", math.nan)
        tip_drift = r.drift.get("tip_delta_mean", math.nan)
        gate_line = _gate_summary_line(r.gates) if r.gates else r.error
        lines.append(
            f"| {i} | {c.name} | {c.aux_coord_weight} | {c.aux_head_size} | {c.aux_loss_type} "
            f"| {r.status} | {mae:.2f} | {accept:.2%} | {drift:.4f} | {tip_drift:.2f} | {gate_line} |"
        )

    # Baseline comparison
    lines += [
        "",
        "## Baseline Comparison",
        "",
        "| Baseline | INT8 Drift |",
        "|----------|------------|",
        "| Phase 10E original | ~1.99 C |",
        "| Phase 11B 08_tip_focus | 1.8405 C |",
        "| Phase 11D aux smoke | 1.89 C |",
        "",
    ]

    # Best candidate detail
    passing = [r for r in results if r.status == "pass"]
    if passing:
        best = min(passing, key=lambda r: r.drift.get("temperature_delta_mean", math.inf))
        lines += [
            "## Champion",
            "",
            f"- **Candidate**: {best.candidate.name}",
            f"- **Weight**: {best.candidate.aux_coord_weight}",
            f"- **Head**: {best.candidate.aux_head_size}",
            f"- **Loss**: {best.candidate.aux_loss_type}",
            f"- **INT8 MAE**: {best.metrics.get('accepted_mae_c', math.nan):.4f} C",
            f"- **INT8 Acceptance**: {best.metrics.get('acceptance_rate', math.nan):.2%}",
            f"- **INT8 Drift**: {best.drift.get('temperature_delta_mean', math.nan):.4f} C",
            f"- **Tip Drift**: {best.drift.get('tip_delta_mean', math.nan):.2f} px",
            f"- **Gates**: {_gate_summary_line(best.gates)}",
            "",
        ]
    else:
        # Find best drift among failures
        failures = [r for r in results if r.status == "fail" and r.drift.get("temperature_delta_mean") is not None]
        if failures:
            best = min(failures, key=lambda r: r.drift.get("temperature_delta_mean", math.inf))
            lines += [
                "## Best Non-Champion",
                "",
                f"- **Candidate**: {best.candidate.name}",
                f"- **Weight**: {best.candidate.aux_coord_weight}",
                f"- **Head**: {best.candidate.aux_head_size}",
                f"- **Loss**: {best.candidate.aux_loss_type}",
                f"- **INT8 MAE**: {best.metrics.get('accepted_mae_c', math.nan):.4f} C",
                f"- **INT8 Acceptance**: {best.metrics.get('acceptance_rate', math.nan):.2%}",
                f"- **INT8 Drift**: {best.drift.get('temperature_delta_mean', math.nan):.4f} C",
                f"- **Tip Drift**: {best.drift.get('tip_delta_mean', math.nan):.2f} px",
                f"- **Gates**: {_gate_summary_line(best.gates)}",
                "",
            ]

    MATRIX_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[MATRIX] Wrote {MATRIX_REPORT_PATH}", flush=True)


def _write_decision_report(results: list[CandidateResult]) -> None:
    """Write the decision report."""
    passing = [r for r in results if r.status == "pass"]
    all_sorted = sorted(results, key=lambda r: r.drift.get("temperature_delta_mean", math.inf))

    lines = [
        "# Phase 11E Aux INT8 Recovery Decision",
        "",
        "## Question",
        "",
        "Can an INT8-friendly auxiliary coordinate head reduce Keras-vs-INT8 temperature drift below 1.0 C?",
        "",
        "## Answer",
        "",
    ]

    if passing:
        champion = passing[0]
        drift = champion.drift.get("temperature_delta_mean", math.nan)
        baseline_drift = 1.8405
        improvement = baseline_drift - drift
        lines += [
            f"**YES** -- one candidate passed all validation gates.",
            "",
            f"- Champion: {champion.candidate.name}",
            f"- INT8 drift: {drift:.4f} C (baseline: {baseline_drift:.4f} C, improvement: {improvement:.4f} C)",
            "",
            "### Recommendation",
            "",
            "1. Freeze the champion checkpoint, export settings, tensor contract, decode mode, and guardrail profile.",
            "2. Run one confirmation replay on val before any test split run.",
            "3. If confirmation passes, proceed to test split and then Cube.AI.",
            "",
        ]
    else:
        best = all_sorted[0] if all_sorted else None
        if best:
            drift = best.drift.get("temperature_delta_mean", math.nan)
            baseline_drift = 1.8405
            lines += [
                f"**NO** -- no candidate passed all validation gates.",
                "",
                f"Best candidate: {best.candidate.name}",
                f"INT8 drift: {drift:.4f} C (baseline: {baseline_drift:.4f} C)",
                "",
                "### Analysis",
                "",
            ]

            # Check if drift improved at all
            if drift < baseline_drift - 0.05:
                lines += [
                    f"Drift improved by {baseline_drift - drift:.4f} C vs baseline but still above 1.0 C gate.",
                    "The aux head direction is promising but insufficient with current architecture.",
                    "",
                    "### Next steps",
                    "",
                    "1. Try a spatially-aware aux head (local offset from heatmap peaks) instead of GAP-only.",
                    "2. Or increase aux head capacity further (Dense(256)->Dense(128)->Dense(64)).",
                    "3. Or add a dedicated tip-coordinate aux head (tip is the weak link).",
                    "",
                ]
            else:
                lines += [
                    "Drift did not materially improve vs baseline. The GAP-only aux head architecture",
                    "is fundamentally limited for INT8 robustness. The pooled features do not provide",
                    "enough spatial resolution to resist INT8 quantization noise in the tip heatmap.",
                    "",
                    "### Next steps",
                    "",
                    "1. Move to a spatially-aware point head that reads from the decoder feature maps",
                    "   (e.g., a small conv head that predicts offsets from the heatmap peak locations).",
                    "2. Or add a tip-only aux head with dedicated spatial features.",
                    "3. Or accept the 1.8-1.9 C drift as the architectural limit for GAP-based aux and",
                    "   focus on other improvements (larger backbone, better calibration, etc.).",
                    "",
                ]

        lines += [
            "## All Candidates",
            "",
            "| # | Candidate | Drift | MAE | Accept | Status |",
            "|---|-----------|-------|-----|--------|--------|",
        ]
        for i, r in enumerate(all_sorted, 1):
            c = r.candidate
            drift_val = r.drift.get("temperature_delta_mean", math.nan)
            mae = r.metrics.get("accepted_mae_c", math.nan)
            accept = r.metrics.get("acceptance_rate", math.nan)
            lines.append(f"| {i} | {c.name} | {drift_val:.4f} | {mae:.2f} | {accept:.2%} | {r.status} |")

    DECISION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[MATRIX] Wrote {DECISION_REPORT_PATH}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 11E aux INT8 recovery matrix")
    parser.add_argument("--quick", action="store_true", help="Shortened training (5 frozen, 0 unfrozen)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only export+replay")
    parser.add_argument("--skip-export", action="store_true", help="Skip export, only replay")
    parser.add_argument("--force", action="store_true", help="Force re-run of all steps")
    parser.add_argument("--candidates", type=str, nargs="*", help="Run only these candidate names")
    args = parser.parse_args()

    candidates = CANDIDATES
    if args.candidates:
        candidates = [c for c in CANDIDATES if c.name in args.candidates]
        if not candidates:
            print(f"[MATRIX] No matching candidates for: {args.candidates}", flush=True)
            sys.exit(1)

    print(f"[MATRIX] Phase 11E Aux INT8 Recovery Matrix", flush=True)
    print(f"[MATRIX]   Candidates: {len(candidates)}", flush=True)
    print(f"[MATRIX]   Quick: {args.quick}", flush=True)
    print(f"[MATRIX]   Skip train: {args.skip_train}", flush=True)
    print(f"[MATRIX]   Skip export: {args.skip_export}", flush=True)
    print(f"[MATRIX]   Force: {args.force}", flush=True)

    results: list[CandidateResult] = []
    start_time = time.time()

    for cand in candidates:
        print(f"\n{'#'*72}", flush=True)
        print(f"[MATRIX] Candidate: {cand.name} (tag={cand.tag})", flush=True)
        print(f"  weight={cand.aux_coord_weight}, head={cand.aux_head_size}, loss={cand.aux_loss_type}", flush=True)
        print(f"{'#'*72}", flush=True)

        # Train
        if not args.skip_train:
            train_ok = _run_training(cand, quick=args.quick, force=args.force)
            if not train_ok:
                results.append(CandidateResult(candidate=cand, status="skip_train", error="Training failed"))
                continue

        # Export
        if not args.skip_export:
            export_ok = _run_export(cand, force=args.force)
            if not export_ok:
                results.append(CandidateResult(candidate=cand, status="skip_export", error="Export failed"))
                continue

        # Replay
        replay_result = _run_replay(cand, force=args.force)
        results.append(replay_result)

        # Print progress
        elapsed = time.time() - start_time
        passing = sum(1 for r in results if r.status == "pass")
        print(f"\n[MATRIX] Progress: {len(results)}/{len(candidates)} done, {passing} passing, {elapsed/60:.1f} min elapsed", flush=True)
        if replay_result.gates:
            print(f"[MATRIX] Gates: {_gate_summary_line(replay_result.gates)}", flush=True)

    # Summary
    elapsed = time.time() - start_time
    passing = [r for r in results if r.status == "pass"]
    print(f"\n{'='*72}", flush=True)
    print(f"[MATRIX] Complete: {len(results)}/{len(candidates)} candidates, {len(passing)} passing gates", flush=True)
    print(f"[MATRIX] Total time: {elapsed/60:.1f} minutes", flush=True)

    # Sort by drift
    all_with_drift = [r for r in results if r.drift.get("temperature_delta_mean") is not None]
    all_with_drift.sort(key=lambda r: r.drift["temperature_delta_mean"])
    for r in all_with_drift:
        drift = r.drift.get("temperature_delta_mean", math.nan)
        mae = r.metrics.get("accepted_mae_c", math.nan)
        accept = r.metrics.get("acceptance_rate", math.nan)
        print(f"[MATRIX]   {r.candidate.name}: drift={drift:.4f} mae={mae:.2f} accept={accept:.2%} {r.status}", flush=True)

    # Write reports
    _write_matrix_report(results)
    _write_decision_report(results)

    if passing:
        champion = passing[0]
        print(f"\n[MATRIX] CHAMPION: {champion.candidate.name}", flush=True)
        print(f"[MATRIX]   drift={champion.drift.get('temperature_delta_mean', math.nan):.4f}", flush=True)
        print(f"[MATRIX]   mae={champion.metrics.get('accepted_mae_c', math.nan):.2f}", flush=True)
    else:
        print(f"\n[MATRIX] No champion found.", flush=True)
        if all_with_drift:
            best = all_with_drift[0]
            print(f"[MATRIX] Best drift: {best.candidate.name} = {best.drift['temperature_delta_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
