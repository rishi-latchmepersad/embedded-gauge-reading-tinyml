"""Calibration helpers for inner dial angle-to-temperature evaluation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
from scipy.optimize import least_squares

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)

DEFAULT_COLD_ANGLE_DEGREES: float = 135.0
DEFAULT_SWEEP_DEGREES: float = 270.0
DEFAULT_MINIMUM_CELSIUS: float = -30.0
DEFAULT_MAXIMUM_CELSIUS: float = 50.0


@dataclass(frozen=True)
class GeometryTemperatureRecord:
    """One clean manifest row with derived oracle geometry temperature values."""

    image_path: str
    split: str
    source_manifest: str
    temperature_c: float
    source_width: int
    source_height: int
    dial_radius_source: float
    center_x_source: float
    center_y_source: float
    tip_x_source: float
    tip_y_source: float
    center_tip_distance_pixels: float
    angle_degrees: float
    current_temperature_c: float
    current_absolute_error_c: float


@dataclass(frozen=True)
class CalibrationCandidate:
    """One candidate angle-to-temperature mapping."""

    name: str
    kind: Literal["current", "cold_sweep", "linear", "robust_linear"]
    params: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        """Convert the candidate into a JSON-friendly dictionary."""

        return {
            "name": self.name,
            "kind": self.kind,
            "params": self.params,
        }


def load_clean_geometry_records(manifest_path: Path) -> list[GeometryTemperatureRecord]:
    """Load clean rows from the manifest and precompute oracle geometry values."""

    import csv

    records: list[GeometryTemperatureRecord] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("quality_flag") != "clean":
                continue

            center_x = float(row["center_x_source"])
            center_y = float(row["center_y_source"])
            tip_x = float(row["tip_x_source"])
            tip_y = float(row["tip_y_source"])
            angle = angle_degrees_from_center_to_tip(center_x, center_y, tip_x, tip_y)
            current_temperature = celsius_from_inner_dial_angle_degrees(angle)
            temperature = float(row["temperature_c"])

            records.append(
                GeometryTemperatureRecord(
                    image_path=row["image_path"],
                    split=row["split"],
                    source_manifest=row.get("source_manifest", ""),
                    temperature_c=temperature,
                    source_width=int(row["source_width"]),
                    source_height=int(row["source_height"]),
                    dial_radius_source=float(row["dial_radius_source"]),
                    center_x_source=center_x,
                    center_y_source=center_y,
                    tip_x_source=tip_x,
                    tip_y_source=tip_y,
                    center_tip_distance_pixels=float(row.get("center_tip_distance_pixels", "nan")),
                    angle_degrees=angle,
                    current_temperature_c=current_temperature,
                    current_absolute_error_c=abs(current_temperature - temperature),
                )
            )

    return records


def unwrap_angle_from_cold(angle_degrees: float, *, cold_angle_degrees: float = DEFAULT_COLD_ANGLE_DEGREES) -> float:
    """Convert a circular angle to a linear distance from the cold end."""

    return (angle_degrees - cold_angle_degrees) % 360.0


def predict_current_mapping_temperature(angle_degrees: float) -> float:
    """Predict temperature using the current physical mapping."""

    return celsius_from_inner_dial_angle_degrees(angle_degrees)


def fit_constrained_cold_sweep_candidate(
    records: Sequence[GeometryTemperatureRecord],
    *,
    name: str = "B_constrained_cold_sweep",
) -> CalibrationCandidate:
    """Fit the cold angle and sweep while keeping the Celsius endpoints fixed."""

    angles = np.asarray([record.angle_degrees for record in records], dtype=np.float64)
    temperatures = np.asarray([record.temperature_c for record in records], dtype=np.float64)

    def _residuals(params: np.ndarray) -> np.ndarray:
        cold_angle, sweep_degrees = params
        predictions = np.asarray(
            [
                celsius_from_inner_dial_angle_degrees(
                    float(angle),
                    cold_angle_degrees=float(cold_angle),
                    sweep_degrees=float(sweep_degrees),
                    minimum_celsius=DEFAULT_MINIMUM_CELSIUS,
                    maximum_celsius=DEFAULT_MAXIMUM_CELSIUS,
                )
                for angle in angles
            ],
            dtype=np.float64,
        )
        return predictions - temperatures

    result = least_squares(
        _residuals,
        x0=np.asarray([DEFAULT_COLD_ANGLE_DEGREES, DEFAULT_SWEEP_DEGREES], dtype=np.float64),
        bounds=(np.asarray([0.0, 100.0], dtype=np.float64), np.asarray([360.0, 400.0], dtype=np.float64)),
    )
    cold_angle, sweep_degrees = result.x
    return CalibrationCandidate(
        name=name,
        kind="cold_sweep",
        params={
            "cold_angle_degrees": float(cold_angle),
            "sweep_degrees": float(sweep_degrees),
            "minimum_celsius": DEFAULT_MINIMUM_CELSIUS,
            "maximum_celsius": DEFAULT_MAXIMUM_CELSIUS,
        },
    )


def fit_linear_temperature_candidate(
    records: Sequence[GeometryTemperatureRecord],
    *,
    name: str = "C_linear_unwrapped",
    robust: bool = False,
) -> CalibrationCandidate:
    """Fit a linear temperature model over the unwrapped angle distance."""

    x_values = np.asarray(
        [unwrap_angle_from_cold(record.angle_degrees) for record in records],
        dtype=np.float64,
    )
    y_values = np.asarray([record.temperature_c for record in records], dtype=np.float64)

    def _residuals(params: np.ndarray) -> np.ndarray:
        slope, intercept = params
        return slope * x_values + intercept - y_values

    least_squares_kwargs: dict[str, Any] = {
        "x0": np.asarray([0.3, -30.0], dtype=np.float64),
    }
    if robust:
        least_squares_kwargs["loss"] = "huber"
        least_squares_kwargs["f_scale"] = 1.5

    result = least_squares(_residuals, **least_squares_kwargs)
    slope, intercept = result.x
    return CalibrationCandidate(
        name=name,
        kind="robust_linear" if robust else "linear",
        params={
            "slope": float(slope),
            "intercept": float(intercept),
            "cold_angle_degrees": DEFAULT_COLD_ANGLE_DEGREES,
        },
    )


def predict_temperature_from_candidate(angle_degrees: float, candidate: CalibrationCandidate) -> float:
    """Predict temperature using one calibration candidate."""

    if candidate.kind == "current":
        return predict_current_mapping_temperature(angle_degrees)
    if candidate.kind == "cold_sweep":
        return celsius_from_inner_dial_angle_degrees(
            angle_degrees,
            cold_angle_degrees=candidate.params["cold_angle_degrees"],
            sweep_degrees=candidate.params["sweep_degrees"],
            minimum_celsius=candidate.params["minimum_celsius"],
            maximum_celsius=candidate.params["maximum_celsius"],
        )
    if candidate.kind in {"linear", "robust_linear"}:
        delta = unwrap_angle_from_cold(angle_degrees, cold_angle_degrees=candidate.params["cold_angle_degrees"])
        return candidate.params["slope"] * delta + candidate.params["intercept"]
    raise ValueError(f"Unknown calibration candidate kind: {candidate.kind}")


def evaluate_candidate(
    records: Sequence[GeometryTemperatureRecord],
    candidate: CalibrationCandidate,
) -> list[dict[str, Any]]:
    """Evaluate one candidate against a set of geometry records."""

    evaluated: list[dict[str, Any]] = []
    for record in records:
        predicted_temperature = predict_temperature_from_candidate(record.angle_degrees, candidate)
        predicted_error = abs(predicted_temperature - record.temperature_c)
        evaluated.append(
            {
                **asdict(record),
                "candidate_name": candidate.name,
                "candidate_kind": candidate.kind,
                "predicted_temperature_c": float(predicted_temperature),
                "absolute_error_c": float(predicted_error),
            }
        )
    return evaluated


def summarize_mae(rows: Sequence[dict[str, Any]], *, field_name: str = "absolute_error_c") -> float:
    """Compute mean absolute error for a set of evaluated rows."""

    errors = np.asarray([float(row[field_name]) for row in rows], dtype=np.float64)
    return float(np.mean(errors))


def make_candidate_current() -> CalibrationCandidate:
    """Return the current physical mapping as a named candidate."""

    return CalibrationCandidate(name="A_current_mapping", kind="current", params={})
