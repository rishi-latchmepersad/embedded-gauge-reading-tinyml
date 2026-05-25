#!/usr/bin/env python3
"""Run a bounded INT8 recovery matrix for geometry_heatmap_v4_112.

8 curated candidates, each trained with Phase 11 anti-collapse losses, then
exported to INT8 and replayed on val.  Produces a matrix summary report and
a decision report ranking candidates by gate pass -> acceptance -> MAE -> drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Shared paths used by every candidate (not per-candidate configurable in this matrix)
MANIFEST_PATH = REPO_ROOT / "ml/data/geometry_reader_manifest_v2_clean.csv"
CALIBRATION_PATH = REPO_ROOT / "ml/artifacts/training/inner_dial_angle_calibration_v1/calibration_candidates.json"
THRESHOLDS_PATH = REPO_ROOT / "ml/artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json"
DECODER_PATH = REPO_ROOT / "ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method_corrected.json"
INIT_MODEL_PATH = REPO_ROOT / "ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras"

# Output directory roots
TRAINING_BASE = REPO_ROOT / "ml/artifacts/training/geometry_heatmap_v4_112_int8_recovery"
DEPLOY_BASE = REPO_ROOT / "ml/artifacts/deployment/geometry_heatmap_v4_112_tflite"
REPORT_DIR = REPO_ROOT / "ml/reports"
DEBUG_BASE = REPO_ROOT / "ml/debug/geometry_heatmap_v4_112_int8_recovery"

# Reports
MATRIX_REPORT_PATH = REPORT_DIR / "geometry_heatmap_v4_112_int8_recovery_matrix.md"
DECISION_REPORT_PATH = REPORT_DIR / "geometry_heatmap_v4_112_int8_recovery_decision.md"

# Scripts
TRAIN_SCRIPT = REPO_ROOT / "ml/scripts/train_geometry_heatmap_v4_112_quant_native.py"
EXPORT_SCRIPT = REPO_ROOT / "ml/scripts/export_geometry_heatmap_v4_112_int8.py"
REPLAY_SCRIPT = REPO_ROOT / "ml/scripts/eval_geometry_heatmap_v4_112_tflite_replay.py"

# Validation gates -- a candidate must pass ALL to be considered deployable
GATES: dict[str, float] = {
    "accepted_mae_c": 4.5,
    "acceptance_rate": 0.65,
    "worst_accepted_error_c": 20.0,
    "accepted_gt20_failures": 0,
    "temperature_delta_mean": 1.0,
}

# Default training params (not swept in this matrix)
FROZEN_EPOCHS = 40
UNFROZEN_EPOCHS = 20
FROZEN_LR = 1e-5
UNFROZEN_LR = 5e-6
BATCH_SIZE = 16
HEATMAP_SIZE = 112
SIGMA_PIXELS = 2.5


# ---------------------------------------------------------------------------
# Candidate definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One cell in the INT8 recovery matrix.

    The tag is auto-derived from the swept hyperparameters and used as the
    filesystem suffix for all artifacts so every run is traceable.
    """

    name: str
    peak_target: float
    peak_shape_center_weight: float
    peak_shape_tip_weight: float
    confidence_floor_weight: float
    warmup_epochs: int

    @property
    def tag(self) -> str:
        return (
            f"{self.name}__pt{self.peak_target:.2f}"
            f"_c{self.peak_shape_center_weight:.2f}"
            f"_t{self.peak_shape_tip_weight:.2f}"
            f"_cf{self.confidence_floor_weight:.2f}"
            f"_wu{self.warmup_epochs}"
        )

    @property
    def training_dir(self) -> Path:
        return TRAINING_BASE / f"candidate_{self.tag}"

    @property
    def deploy_dir(self) -> Path:
        return DEPLOY_BASE / f"recovery_{self.tag}"

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
    def replay_predictions_path(self) -> Path:
        return self.debug_dir / "val_predictions.csv"

    @property
    def replay_summary_path(self) -> Path:
        return self.debug_dir / "val_summary.csv"

    @property
    def replay_remaining_path(self) -> Path:
        return self.debug_dir / "val_remaining_worst.csv"

    @property
    def replay_report_path(self) -> Path:
        return self.debug_dir / "replay_report.md"


# 8 curated candidates covering the Phase 11 hyperparameter space.
# These were hand-picked to span the bounded grid without exploding cartesian.
CANDIDATES: list[Candidate] = [
    Candidate(
        name="01_conservative",
        peak_target=0.30,
        peak_shape_center_weight=0.10,
        peak_shape_tip_weight=0.20,
        confidence_floor_weight=0.05,
        warmup_epochs=5,
    ),
    Candidate(
        name="02_lower_peak_target",
        peak_target=0.25,
        peak_shape_center_weight=0.10,
        peak_shape_tip_weight=0.20,
        confidence_floor_weight=0.05,
        warmup_epochs=5,
    ),
    Candidate(
        name="03_lighter_shaping",
        peak_target=0.30,
        peak_shape_center_weight=0.05,
        peak_shape_tip_weight=0.15,
        confidence_floor_weight=0.05,
        warmup_epochs=5,
    ),
    Candidate(
        name="04_short_warmup",
        peak_target=0.30,
        peak_shape_center_weight=0.10,
        peak_shape_tip_weight=0.20,
        confidence_floor_weight=0.05,
        warmup_epochs=3,
    ),
    Candidate(
        name="05_light_all",
        peak_target=0.25,
        peak_shape_center_weight=0.05,
        peak_shape_tip_weight=0.15,
        confidence_floor_weight=0.03,
        warmup_epochs=5,
    ),
    Candidate(
        name="06_aggressive",
        peak_target=0.25,
        peak_shape_center_weight=0.05,
        peak_shape_tip_weight=0.15,
        confidence_floor_weight=0.03,
        warmup_epochs=3,
    ),
    Candidate(
        name="07_high_peak_low_floor",
        peak_target=0.30,
        peak_shape_center_weight=0.05,
        peak_shape_tip_weight=0.20,
        confidence_floor_weight=0.03,
        warmup_epochs=5,
    ),
    Candidate(
        name="08_tip_focus",
        peak_target=0.25,
        peak_shape_center_weight=0.05,
        peak_shape_tip_weight=0.20,
        confidence_floor_weight=0.05,
        warmup_epochs=3,
    ),
]


# ---------------------------------------------------------------------------
# Structured result container
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """Metrics collected for one candidate after replay."""
    candidate: Candidate
    status: str = "unknown"  # pass | fail | skip_train | skip_export | skip_replay | error
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
    """Read a summary CSV and return a float dict for the given model_type row.

    If model_type is None, returns the first row (legacy behaviour for
    single-row summaries).  The replay summary CSV has one row per model
    variant (keras, tflite_float32, tflite_int8).
    """
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
        "--peak-target", str(candidate.peak_target),
        "--peak-shape-center-weight", str(candidate.peak_shape_center_weight),
        "--peak-shape-tip-weight", str(candidate.peak_shape_tip_weight),
        "--confidence-floor-weight", str(candidate.confidence_floor_weight),
        "--warmup-epochs", str(candidate.warmup_epochs),
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
        metrics = _parse_summary_csv(candidate.replay_summary_path, model_type="tflite_int8")
        if metrics is not None:
            result.metrics = metrics
            result.drift = {k: v for k, v in metrics.items() if "delta" in k or "disagreement" in k}
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
        "--output-suffix", "",
        "--batch-size", "16",
        "--manifest-path", str(MANIFEST_PATH),
        "--model-path", str(candidate.model_path),
        "--float-tflite-path", str(candidate.float_tflite_path),
        "--int8-tflite-path", str(candidate.int8_tflite_path),
        "--contract-path", str(candidate.contract_path),
        "--calibration-path", str(CALIBRATION_PATH),
        "--thresholds-path", str(THRESHOLDS_PATH),
        "--selected-decode-path", str(DECODER_PATH),
        "--predictions-path", str(candidate.replay_predictions_path),
        "--summary-path", str(candidate.replay_summary_path),
        "--remaining-path", str(candidate.replay_remaining_path),
        "--report-path", str(candidate.replay_report_path),
    ]
    proc_result = _run_subprocess(cmd, label=f"replay {candidate.tag}")
    if proc_result.returncode != 0:
        result.status = "error"
        result.error = f"Replay subprocess returned rc={proc_result.returncode}"
        return result

    # Parse the tflite_int8 row from the summary CSV to extract metrics
    metrics = _parse_summary_csv(candidate.replay_summary_path, model_type="tflite_int8")
    if metrics is None:
        result.status = "error"
        result.error = f"Could not parse tflite_int8 row from summary CSV at {candidate.replay_summary_path}"
        return result

    result.metrics = metrics
    result.drift = {k: v for k, v in metrics.items() if "delta" in k or "disagreement" in k}
    result.gates = _check_gates(result.metrics, result.drift)
    result.status = "pass" if _all_gates_pass(result.gates) else "fail"
    return result


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_matrix_summary(results: list[CandidateResult], *, quick: bool) -> None:
    """Write the matrix summary markdown report."""
    lines: list[str] = [
        "# Geometry Heatmap v4 112 INT8 Recovery Matrix",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Candidates: {len(results)}",
        f"- Quick mode: {quick}",
        "",
        "## Candidate Grid",
        "",
        "| # | Name | peak_target | center_weight | tip_weight | conf_floor_weight | warmup_epochs |",
        "|---|------|-------------|---------------|------------|-------------------|---------------|",
    ]
    for idx, c in enumerate(CANDIDATES, 1):
        lines.append(
            f"| {idx} | {c.name} | {c.peak_target} | "
            f"{c.peak_shape_center_weight} | {c.peak_shape_tip_weight} | "
            f"{c.confidence_floor_weight} | {c.warmup_epochs} |"
        )
    lines += [
        "",
        "## Gates (must all pass)",
        "",
    ]
    for gate_name, threshold in GATES.items():
        comp = "<=" if gate_name != "acceptance_rate" else ">="
        if gate_name == "worst_accepted_error_c":
            comp = "<"
        lines.append(f"- **{gate_name}**: {comp} {threshold}")
    lines += [
        "",
        "## Results by Candidate",
        "",
        "| Rank | Name | Status | Accepted MAE | Acceptance | Worst Error | >20C Fail | "
        "Temp Drift Mean | Center Drift | Tip Drift | Gates |",
        "|------|------|--------|--------------|------------|-------------|-----------|"
        "--------------|--------------|-----------|-------|",
    ]

    # Sort: pass first, then by acceptance desc, MAE asc, drift asc
    def _sort_key(r: CandidateResult) -> tuple:
        gate_bonus = 0 if _all_gates_pass(r.gates) else 1
        acc = -r.metrics.get("acceptance_rate", 0.0)
        mae = r.metrics.get("accepted_mae_c", math.inf)
        drift_val = r.drift.get("temperature_delta_mean", math.inf)
        return (gate_bonus, acc, mae, drift_val)

    sorted_results = sorted(results, key=_sort_key)

    for rank, r in enumerate(sorted_results, 1):
        c = r.candidate
        mae = r.metrics.get("accepted_mae_c", math.nan)
        acc = r.metrics.get("acceptance_rate", math.nan)
        worst = r.metrics.get("worst_accepted_error_c", math.nan)
        gt20_raw = r.metrics.get("accepted_gt20_failures", math.nan)
        gt20 = int(gt20_raw) if not math.isnan(gt20_raw) else -1
        t_drift = r.drift.get("temperature_delta_mean", math.nan)
        c_drift = r.drift.get("center_delta_mean", math.nan)
        t_drift_str = f"{t_drift:.4f}" if not math.isnan(t_drift) else "N/A"
        c_drift_str = f"{c_drift:.4f}" if not math.isnan(c_drift) else "N/A"
        tip_drift = r.drift.get("tip_delta_mean", math.nan)
        tip_drift_str = f"{tip_drift:.4f}" if not math.isnan(tip_drift) else "N/A"
        gate_str = _gate_summary_line(r.gates)
        lines.append(
            f"| {rank} | {c.name} | {r.status} | {mae:.4f} | {acc:.4f} | "
            f"{worst:.2f} | {gt20} | {t_drift_str} | "
            f"{c_drift_str} | {tip_drift_str} | {gate_str} |"
        )

    lines += [
        "",
        "## Detailed Metrics",
        "",
    ]
    for r in sorted_results:
        c = r.candidate
        lines += [
            f"### {c.name} ({c.tag})",
            f"- Status: {r.status}",
        ]
        if r.error:
            lines.append(f"- Error: {r.error}")
        for m_name, m_val in sorted(r.metrics.items()):
            if isinstance(m_val, float):
                lines.append(f"- {m_name}: {m_val:.6f}")
            else:
                lines.append(f"- {m_name}: {m_val}")
        for m_name, m_val in sorted(r.drift.items()):
            if isinstance(m_val, float):
                lines.append(f"- {m_name}: {m_val:.6f}")
            else:
                lines.append(f"- {m_name}: {m_val}")
        if r.gates:
            lines.append(f"- Gate results: {_gate_summary_line(r.gates)}")
        lines.append("")

    MATRIX_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[MATRIX] Wrote {MATRIX_REPORT_PATH}", flush=True)


def _write_decision_report(results: list[CandidateResult]) -> None:
    """Rank candidates and recommend champion or document failure mode."""
    passing = [r for r in results if r.status == "pass"]
    failing = [r for r in results if r.status == "fail"]

    lines: list[str] = [
        "# INT8 Recovery Decision",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Candidates: {len(results)}",
        f"- Passing: {len(passing)}",
        f"- Failing: {len(failing)}",
        "",
    ]

    if passing:
        # Rank by: acceptance desc, MAE asc, drift asc
        def _pass_key(r: CandidateResult) -> tuple:
            return (
                -r.metrics.get("acceptance_rate", 0.0),
                r.metrics.get("accepted_mae_c", math.inf),
                r.drift.get("temperature_delta_mean", math.inf),
            )

        passing_sorted = sorted(passing, key=_pass_key)
        champion = passing_sorted[0]

        lines += [
            "## Champion",
            f"- **{champion.candidate.name}** ({champion.candidate.tag})",
            f"- Gate result: {_gate_summary_line(champion.gates)}",
            f"- Accepted MAE: {champion.metrics.get('accepted_mae_c', math.nan):.4f} C",
            f"- Acceptance rate: {champion.metrics.get('acceptance_rate', math.nan):.4f}",
            f"- Temperature drift mean: {champion.drift.get('temperature_delta_mean', math.nan):.4f} C",
            "",
            "### All Passing Candidates (ranked)",
            "",
            "| Rank | Name | Accepted MAE | Acceptance | Temp Drift Mean |",
            "|------|------|--------------|------------|--------------|",
        ]
        for rank, r in enumerate(passing_sorted, 1):
            mae = r.metrics.get("accepted_mae_c", math.nan)
            acc = r.metrics.get("acceptance_rate", math.nan)
            td = r.drift.get("temperature_delta_mean", math.nan)
            lines.append(
                f"| {rank} | {r.candidate.name} | {mae:.4f} | {acc:.4f} | {td:.4f} |"
            )

        lines += [
            "",
            "## Decision",
            "",
            f"**Decision A**: Proceed to Cube.AI INT8 feasibility with champion **{champion.candidate.name}**.",
            "",
            "The champion passes all gates. Next step: run one final test replay, then generate Cube.AI package.",
        ]
    else:
        # No passing candidate -- identify dominant failing gate(s)
        gate_fail_counts: dict[str, int] = {}
        for r in failing:
            for gate_name, passed in r.gates.items():
                if not passed:
                    gate_fail_counts[gate_name] = gate_fail_counts.get(gate_name, 0) + 1

        dominant_gates = sorted(gate_fail_counts.items(), key=lambda x: -x[1])
        lines += [
            "## No Candidate Passes All Gates",
            "",
            "### Dominant Failing Gates",
            "",
        ]
        for gate_name, count in dominant_gates:
            lines.append(f"- **{gate_name}**: failed by {count}/{len(failing)} candidates")
        lines += [
            "",
            "### Top 2 Near-Miss Candidates",
            "",
        ]

        def _fail_key(r: CandidateResult) -> tuple:
            passed_count = sum(1 for v in r.gates.values() if v)
            acc = -r.metrics.get("acceptance_rate", 0.0)
            mae = r.metrics.get("accepted_mae_c", math.inf)
            return (-passed_count, acc, mae)

        failing_sorted = sorted(failing, key=_fail_key)
        for rank, r in enumerate(failing_sorted[:2], 1):
            lines += [
                f"#### {rank}. {r.candidate.name} ({r.candidate.tag})",
                f"- Gates: {_gate_summary_line(r.gates)}",
                f"- Accepted MAE: {r.metrics.get('accepted_mae_c', math.nan):.4f} C",
                f"- Acceptance rate: {r.metrics.get('acceptance_rate', math.nan):.4f}",
                f"- Temperature drift mean: {r.drift.get('temperature_delta_mean', math.nan):.4f} C",
                "",
            ]

        lines += [
            "## Decision",
            "",
            "**Decision B**: No champion found. All candidates fail one or more gates.",
            "Dominant failure pattern: " + ", ".join(f"{g} ({c}/{len(failing)})" for g, c in dominant_gates[:3]),
            "",
            "Recommended next step: Investigate dominant failing gates before escalating to architecture tweaks.",
            "Consider adjusting peak_target, loss weights, or warmup schedule based on failure pattern.",
        ]

    DECISION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[MATRIX] Wrote {DECISION_REPORT_PATH}", flush=True)


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def _run_preflight_checks() -> None:
    """Verify all invariant files and contracts before the matrix starts.

    Fails fast (sys.exit) on the first broken invariant so the user knows
    exactly what to fix.
    """

    # 1. All referenced scripts exist
    for script_path, label in [
        (TRAIN_SCRIPT, "training script"),
        (EXPORT_SCRIPT, "export script"),
        (REPLAY_SCRIPT, "replay script"),
    ]:
        if not script_path.exists():
            print(f"[PREFLIGHT] FAIL: {label} not found at {script_path}", flush=True)
            sys.exit(1)
        print(f"[PREFLIGHT] OK: {label} at {script_path}", flush=True)

    # 2. Init model exists
    if not INIT_MODEL_PATH.exists():
        print(f"[PREFLIGHT] FAIL: init model not found at {INIT_MODEL_PATH}", flush=True)
        sys.exit(1)
    print(f"[PREFLIGHT] OK: init model at {INIT_MODEL_PATH}", flush=True)

    # 3. Guardrail JSON has v4 112 contract
    if not THRESHOLDS_PATH.exists():
        print(f"[PREFLIGHT] FAIL: guardrail JSON not found at {THRESHOLDS_PATH}", flush=True)
        sys.exit(1)
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as f:
        guardrails_payload = json.load(f)
    model_family = guardrails_payload.get("model_family", "")
    if model_family != "geometry_heatmap_v4_112":
        print(
            f"[PREFLIGHT] FAIL: guardrail model_family is "
            f"'{model_family}', expected 'geometry_heatmap_v4_112'",
            flush=True,
        )
        sys.exit(1)
    selected = guardrails_payload.get("selected_thresholds", {})
    spread_px = float(selected.get("max_heatmap_spread_px", -1.0))
    if spread_px != 55.0:
        print(
            f"[PREFLIGHT] FAIL: guardrail max_heatmap_spread_px is "
            f"{spread_px}, expected 55.0",
            flush=True,
        )
        sys.exit(1)
    print(f"[PREFLIGHT] OK: guardrail JSON model_family={model_family} spread={spread_px}", flush=True)

    # 4. Decoder JSON resolves to softargmax w3
    if not DECODER_PATH.exists():
        print(f"[PREFLIGHT] FAIL: decoder JSON not found at {DECODER_PATH}", flush=True)
        sys.exit(1)
    with DECODER_PATH.open("r", encoding="utf-8") as f:
        decoder_payload = json.load(f)
    decode_method = str(decoder_payload.get("decode_method", ""))
    window_size = int(decoder_payload.get("window_size", -1))
    if decode_method != "softargmax" or window_size != 3:
        print(
            f"[PREFLIGHT] FAIL: decoder is {decode_method} w{window_size}, "
            f"expected softargmax w3",
            flush=True,
        )
        sys.exit(1)
    print(f"[PREFLIGHT] OK: decoder {decode_method} w{window_size}", flush=True)

    # 5. Data manifests and calibration exist
    for path, label in [
        (MANIFEST_PATH, "manifest CSV"),
        (CALIBRATION_PATH, "calibration JSON"),
    ]:
        if not path.exists():
            print(f"[PREFLIGHT] FAIL: {label} not found at {path}", flush=True)
            sys.exit(1)
        print(f"[PREFLIGHT] OK: {label} at {path}", flush=True)

    print("[PREFLIGHT] All checks passed.\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded INT8 recovery matrix for geometry_heatmap_v4_112",
    )
    parser.add_argument(
        "--candidates", type=str, nargs="*", default=None,
        help="Specific candidate names to run (default: all 8). "
        "Use '01_conservative 05_light_all' etc.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training steps (use existing artifacts).",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Skip TFLite export steps.",
    )
    parser.add_argument(
        "--skip-replay", action="store_true",
        help="Skip replay eval steps.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if output artifacts already exist.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduce epochs for fast smoke-test iteration.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the candidate grid and exit without running anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.dry_run:
        _run_preflight_checks()

    # Filter candidates
    if args.candidates:
        candidates = [c for c in CANDIDATES if c.name in args.candidates]
        missing = set(args.candidates) - {c.name for c in candidates}
        if missing:
            print(f"[MATRIX] Unknown candidate names: {missing}", flush=True)
            print(f"[MATRIX] Available: {[c.name for c in CANDIDATES]}", flush=True)
            sys.exit(1)
    else:
        candidates = list(CANDIDATES)

    print(f"[MATRIX] Running {len(candidates)} candidates", flush=True)
    print(f"[MATRIX]   Skip train: {args.skip_train}", flush=True)
    print(f"[MATRIX]   Skip export: {args.skip_export}", flush=True)
    print(f"[MATRIX]   Skip replay: {args.skip_replay}", flush=True)
    print(f"[MATRIX]   Force: {args.force}", flush=True)
    print(f"[MATRIX]   Quick: {args.quick}", flush=True)
    print(f"[MATRIX]   Dry run: {args.dry_run}", flush=True)

    if args.dry_run:
        print("\nCandidate grid:\n")
        print(f"{'Name':<25} {'Tag':<60} {'pt':<6} {'cw':<6} {'tw':<6} {'cf':<6} {'wu':<4}")
        print("-" * 120)
        for c in candidates:
            print(
                f"{c.name:<25} {c.tag:<60} {c.peak_target:<6} {c.peak_shape_center_weight:<6} "
                f"{c.peak_shape_tip_weight:<6} {c.confidence_floor_weight:<6} {c.warmup_epochs:<4}"
            )
        print(f"\nTraining dirs under: {TRAINING_BASE}")
        print(f"Deploy dirs under:   {DEPLOY_BASE}")
        print(f"Debug dirs under:    {DEBUG_BASE}")
        print(f"Matrix report:       {MATRIX_REPORT_PATH}")
        print(f"Decision report:     {DECISION_REPORT_PATH}")
        return

    results: list[CandidateResult] = []

    for cand in candidates:
        print(f"\n{'#'*72}", flush=True)
        print(f"# CANDIDATE: {cand.name} ({cand.tag})", flush=True)
        print(f"{'#'*72}", flush=True)

        # Step 1: Train
        if args.skip_train:
            print(f"[MATRIX] Skipping training for {cand.tag}", flush=True)
            train_ok = cand.model_path.exists()
        else:
            train_ok = _run_training(cand, quick=args.quick, force=args.force)

        if not train_ok:
            r = CandidateResult(candidate=cand, status="skip_export")
            if not args.skip_train:
                r.status = "error"
                r.error = "Training failed"
            results.append(r)
            continue

        # Step 2: Export
        if args.skip_export:
            print(f"[MATRIX] Skipping export for {cand.tag}", flush=True)
            export_ok = cand.int8_tflite_path.exists()
        else:
            export_ok = _run_export(cand, force=args.force)

        if not export_ok:
            r = CandidateResult(candidate=cand, status="skip_replay")
            if not args.skip_export:
                r.status = "error"
                r.error = "Export failed"
            results.append(r)
            continue

        # Step 3: Replay
        if args.skip_replay:
            print(f"[MATRIX] Skipping replay for {cand.tag}", flush=True)
            r = CandidateResult(candidate=cand, status="skip_replay")
            results.append(r)
            continue

        r = _run_replay(cand, force=args.force)
        results.append(r)

        print(
            f"[MATRIX] {cand.name}: status={r.status} "
            f"MAE={r.metrics.get('accepted_mae_c', math.nan):.4f} "
            f"Acc={r.metrics.get('acceptance_rate', math.nan):.4f}",
            flush=True,
        )

    # Write reports
    _write_matrix_summary(results, quick=args.quick)
    _write_decision_report(results)

    # Print summary table
    print("\n" + "=" * 72, flush=True)
    print("MATRIX SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"{'Name':<25} {'Status':<12} {'MAE':<10} {'Acc':<10} {'Drift':<10}", flush=True)
    print("-" * 72, flush=True)
    passing = 0
    for r in results:
        mae = r.metrics.get("accepted_mae_c", math.nan)
        acc = r.metrics.get("acceptance_rate", math.nan)
        drift = r.drift.get("temperature_delta_mean", math.nan)
        mae_s = f"{mae:.4f}" if not math.isnan(mae) else "N/A"
        acc_s = f"{acc:.4f}" if not math.isnan(acc) else "N/A"
        drift_s = f"{drift:.4f}" if not math.isnan(drift) else "N/A"
        print(f"{r.candidate.name:<25} {r.status:<12} {mae_s:<10} {acc_s:<10} {drift_s:<10}", flush=True)
        if r.status == "pass":
            passing += 1
    print("-" * 72, flush=True)
    print(f"Passing: {passing}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
