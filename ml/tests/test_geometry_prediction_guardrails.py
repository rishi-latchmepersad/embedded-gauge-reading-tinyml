"""Tests for geometry prediction guardrails."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from embedded_gauge_reading_tinyml.geometry_crop_dataset import SourceGeometryExample
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import HeatmapSample
from embedded_gauge_reading_tinyml.geometry_prediction_guardrails import (
    GeometryGuardrailThresholds,
    apply_geometry_guardrails,
    compute_geometry_quality_features,
    decode_heatmap_geometry_prediction,
    guarded_temperature_from_prediction,
)
from embedded_gauge_reading_tinyml.geometry_temperature_calibration import CalibrationCandidate
from embedded_gauge_reading_tinyml.heatmap_utils import make_gaussian_heatmap


def _make_sample(
    *,
    center_x: float = 112.0,
    center_y: float = 112.0,
    tip_x: float = 96.0,
    tip_y: float = 96.0,
    temperature_c: float = 0.0,
    crop_width: int = 224,
    crop_height: int = 224,
    dial_radius_source: float = 20.0,
) -> HeatmapSample:
    """Create a deterministic synthetic heatmap sample for guardrail tests."""

    example = SourceGeometryExample(
        image_path="ml/data/raw/synthetic.jpg",
        temperature_c=temperature_c,
        source_width=crop_width,
        source_height=crop_height,
        loose_crop_x1=0,
        loose_crop_y1=0,
        loose_crop_x2=crop_width,
        loose_crop_y2=crop_height,
        center_x_source=center_x,
        center_y_source=center_y,
        tip_x_source=tip_x,
        tip_y_source=tip_y,
        dial_radius_source=dial_radius_source,
        split="test",
        quality_flag="clean",
    )
    metadata = {
        "image_path": example.image_path,
        "split": example.split,
        "temperature_c": temperature_c,
        "source_manifest": "synthetic.csv",
        "source_width": crop_width,
        "source_height": crop_height,
        "quality_flag": "clean",
        "dial_radius_source": dial_radius_source,
        "crop_x1": 0,
        "crop_y1": 0,
        "crop_x2": crop_width,
        "crop_y2": crop_height,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "jitter_shift_x": 0,
        "jitter_shift_y": 0,
        "jitter_scale": 1.0,
        "jitter_aspect": 1.0,
        "center_x_norm": center_x / float(crop_width - 1),
        "center_y_norm": center_y / float(crop_height - 1),
        "tip_x_norm": tip_x / float(crop_width - 1),
        "tip_y_norm": tip_y / float(crop_height - 1),
        "center_x_224": center_x,
        "center_y_224": center_y,
        "tip_x_224": tip_x,
        "tip_y_224": tip_y,
    }
    crop_image = np.zeros((224, 224, 3), dtype=np.float32)
    center_heatmap = make_gaussian_heatmap(56, 56, center_x / 223.0, center_y / 223.0, 5.0).astype(np.float32)
    tip_heatmap = make_gaussian_heatmap(56, 56, tip_x / 223.0, tip_y / 223.0, 5.0).astype(np.float32)
    return HeatmapSample(
        example=example,
        crop_image=crop_image,
        metadata=metadata,
        center_heatmap=center_heatmap,
        tip_heatmap=tip_heatmap,
    )


def test_decode_heatmap_geometry_prediction_matches_expected_geometry() -> None:
    """A clean synthetic geometry bundle should decode back to the true points."""

    sample = _make_sample()
    calibration_candidate = CalibrationCandidate(
        name="current",
        kind="current",
        params={},
    )
    decoded = decode_heatmap_geometry_prediction(
        sample,
        sample.center_heatmap,
        sample.tip_heatmap,
        0.75,
        calibration_candidate,
    )
    assert decoded.predicted_center_x_224 == pytest.approx(sample.metadata["center_x_224"], abs=1.0)
    assert decoded.predicted_tip_x_224 == pytest.approx(sample.metadata["tip_x_224"], abs=1.0)
    assert decoded.predicted_center_y_224 == pytest.approx(sample.metadata["center_y_224"], abs=1.0)
    assert decoded.predicted_tip_y_224 == pytest.approx(sample.metadata["tip_y_224"], abs=1.0)
    features = compute_geometry_quality_features(decoded)
    assert features.center_tip_distance_ratio == pytest.approx(1.0, rel=0.15)
    assert features.min_edge_margin_px > 20.0


def test_apply_geometry_guardrails_accepts_plausible_prediction() -> None:
    """A strong, in-bounds prediction should be accepted."""

    sample = _make_sample()
    calibration_candidate = CalibrationCandidate(name="current", kind="current", params={})
    decoded = decode_heatmap_geometry_prediction(
        sample,
        sample.center_heatmap,
        sample.tip_heatmap,
        0.75,
        calibration_candidate,
    )
    result = apply_geometry_guardrails(decoded, GeometryGuardrailThresholds())
    assert result.status == "accepted"
    assert result.rejection_reasons == ()
    assert result.temperature_c == pytest.approx(decoded.predicted_temperature_c_calibrated, abs=1e-6)


def test_apply_geometry_guardrails_rejects_low_peak_prediction() -> None:
    """Low peak values or confidence should trigger rejection."""

    sample = _make_sample()
    calibration_candidate = CalibrationCandidate(name="current", kind="current", params={})
    low_tip_heatmap = (sample.tip_heatmap * 0.2).astype(np.float32)
    decoded = decode_heatmap_geometry_prediction(
        sample,
        sample.center_heatmap,
        low_tip_heatmap,
        0.30,
        calibration_candidate,
    )
    result = apply_geometry_guardrails(decoded, GeometryGuardrailThresholds())
    assert result.status == "rejected"
    assert "tip_peak_too_low" in result.rejection_reasons
    assert "confidence_too_low" in result.rejection_reasons


def test_apply_geometry_guardrails_clamps_mildly_out_of_range_temperature() -> None:
    """Temperatures just outside the physical range should be clamped, not hidden."""

    sample = _make_sample()
    calibration_candidate = CalibrationCandidate(
        name="shifted_linear",
        kind="linear",
        params={
            "slope": 0.0,
            "intercept": 53.0,
            "cold_angle_degrees": 135.0,
        },
    )
    decoded = decode_heatmap_geometry_prediction(
        sample,
        sample.center_heatmap,
        sample.tip_heatmap,
        0.75,
        calibration_candidate,
    )
    result = guarded_temperature_from_prediction(
        sample,
        sample.center_heatmap,
        sample.tip_heatmap,
        0.75,
        calibration_candidate,
        thresholds=GeometryGuardrailThresholds(),
    )
    assert decoded.predicted_temperature_c_calibrated == pytest.approx(53.0, abs=1e-6)
    assert result.status == "clamped"
    assert result.temperature_c == pytest.approx(50.0, abs=1e-6)
    assert result.raw_temperature_c == pytest.approx(53.0, abs=1e-6)
    assert result.rejection_reasons == ("temperature_clamped_to_physical_range",)


def test_apply_geometry_guardrails_rejects_edge_predictions() -> None:
    """Points too close to the edge should be rejected."""

    sample = _make_sample()
    calibration_candidate = CalibrationCandidate(name="current", kind="current", params={})
    decoded = decode_heatmap_geometry_prediction(
        sample,
        sample.center_heatmap,
        sample.tip_heatmap,
        0.75,
        calibration_candidate,
    )
    edge_decoded = replace(
        decoded,
        predicted_center_x_224=3.0,
        predicted_center_y_224=3.0,
        predicted_tip_x_224=23.0,
        predicted_tip_y_224=3.0,
    )
    result = apply_geometry_guardrails(edge_decoded, GeometryGuardrailThresholds())
    assert result.status == "rejected"
    assert "predicted_point_near_edge" in result.rejection_reasons
