#!/usr/bin/env python3
"""Inspect whether geometry_heatmap_v2 can use standard QAT cleanly.

The script is intentionally conservative: if tensorflow_model_optimization is
missing or the model has an incompatible structure, it records that the Phase 8
fallback should be quantization-noise fine-tuning instead of forcing TFMOT.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import load_geometry_heatmap_keras_model


@dataclass(frozen=True)
class FeasibilityReport:
    """Structured QAT feasibility findings."""

    tensorflow_version: str
    keras_version: str
    tfmot_available: bool
    model_path: str
    model_name: str
    input_shape: list[int | None]
    output_names: list[str]
    output_shapes: list[list[int | None]]
    custom_layer_count: int
    custom_layer_names: list[str]
    standard_qat_feasible: bool
    preferred_training_strategy: str
    notes: list[str]


def _resolve_path(repo_root: Path, path: Path) -> Path:
    """Resolve a repository-relative path."""

    return path if path.is_absolute() else repo_root / path


def _find_custom_layers(model: tf.keras.Model) -> list[str]:
    """Return any non-standard layers that would need special QAT handling."""

    custom_layers: list[str] = []
    for layer in model.layers:
        module_name = layer.__class__.__module__
        if module_name.startswith("keras.") or module_name.startswith("tensorflow.") or module_name.startswith("builtins"):
            continue
        if isinstance(layer, tf.keras.Model):
            continue
        custom_layers.append(f"{layer.name}:{layer.__class__.__name__}")
    return custom_layers


def _summarize_model(model: tf.keras.Model, *, model_path: Path) -> FeasibilityReport:
    """Build a conservative feasibility summary for the saved geometry model."""

    custom_layers = _find_custom_layers(model)
    tfmot_available = importlib.util.find_spec("tensorflow_model_optimization") is not None
    standard_qat_feasible = bool(tfmot_available and not custom_layers)
    notes = [
        "The model has three named outputs: center_heatmap, tip_heatmap, confidence.",
        "A clean TFMOT path requires the optional tensorflow_model_optimization dependency.",
    ]
    if not tfmot_available:
        notes.append("tensorflow_model_optimization is not installed in the active Poetry environment.")
    if custom_layers:
        notes.append("Custom layers would need explicit QAT wrapping: " + ", ".join(custom_layers))
    if standard_qat_feasible:
        preferred_strategy = "tfmot_quantization_aware_training"
        notes.append("Standard QAT appears structurally feasible.")
    else:
        preferred_strategy = "quantization_noise_fine_tuning"
        notes.append("Prefer quantization-noise fine-tuning with fake int8 output round-trips.")
    return FeasibilityReport(
        tensorflow_version=str(tf.__version__),
        keras_version=str(tf.keras.__version__),
        tfmot_available=tfmot_available,
        model_path=str(model_path),
        model_name=str(model.name),
        input_shape=[None if value is None else int(value) for value in model.input_shape],
        output_names=[str(name) for name in model.output_names],
        output_shapes=[
            [None if value is None else int(value) for value in list(shape)]
            for shape in model.output_shape
        ],
        custom_layer_count=len(custom_layers),
        custom_layer_names=custom_layers,
        standard_qat_feasible=standard_qat_feasible,
        preferred_training_strategy=preferred_strategy,
        notes=notes,
    )


def _write_report(report: FeasibilityReport, output_path: Path) -> None:
    """Render the feasibility findings as a short markdown report."""

    lines = [
        "# Geometry Heatmap v2 QAT Feasibility",
        "",
        f"- TensorFlow: {report.tensorflow_version}",
        f"- Keras: {report.keras_version}",
        f"- tensorflow_model_optimization available: {report.tfmot_available}",
        f"- Model path: {report.model_path}",
        f"- Model name: {report.model_name}",
        f"- Input shape: {report.input_shape}",
        f"- Output names: {', '.join(report.output_names)}",
        f"- Output shapes: {report.output_shapes}",
        f"- Custom layer count: {report.custom_layer_count}",
        f"- Custom layers: {', '.join(report.custom_layer_names) if report.custom_layer_names else 'none'}",
        f"- Standard TFMOT QAT feasible: {report.standard_qat_feasible}",
        f"- Preferred training strategy: {report.preferred_training_strategy}",
        "",
        "## Notes",
    ]
    lines.extend(f"- {note}" for note in report.notes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Inspect the base geometry_heatmap_v2 model and write the feasibility report."""

    parser = argparse.ArgumentParser(description="Check geometry_heatmap_v2 QAT feasibility")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v2/model.keras"),
        help="Base geometry heatmap model to inspect.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_v2_qat_feasibility.md"),
        help="Feasibility report destination.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = _resolve_path(repo_root, args.model_path)
    report_path = _resolve_path(repo_root, args.report_path)

    model = load_geometry_heatmap_keras_model(model_path)
    report = _summarize_model(model, model_path=model_path)
    _write_report(report, report_path)

    print(json.dumps(asdict(report), indent=2, sort_keys=True), flush=True)
    print(f"[QAT] Wrote feasibility report to {report_path}", flush=True)
    if not report.standard_qat_feasible:
        print("[QAT] Falling back to quantization-noise fine-tuning.", flush=True)


if __name__ == "__main__":
    main()
