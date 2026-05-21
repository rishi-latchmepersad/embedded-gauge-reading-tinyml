"""Regression tests for the firmware-style polar preprocessing helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from embedded_gauge_reading_tinyml.firmware_preprocessing import (
    POLAR_VOTE_BINS,
    build_firmware_polar_vote_tensor,
    build_training_style_polar_vote_float32,
    build_training_style_polar_vote_tensor,
    decode_circular_vote_logits,
    firmware_training_crop_box,
    load_capture_image,
    probe_tensor,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.polar_vote_v28 import build_polar_vote_v28_model


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
APP_AI_C: Path = REPO_ROOT / "firmware" / "stm32" / "n657" / "Appli" / "Src" / "app_ai.c"
CAPTURE_IMAGE: Path = REPO_ROOT / "ml" / "data" / "captured_images" / "capture_0c.png"
V28_SAMPLE_IMAGE: Path = REPO_ROOT / "ml" / "data" / "captured_images" / "capture_0073.png"
V28_WEIGHTS: Path = (
    REPO_ROOT
    / "ml"
    / "artifacts"
    / "training"
    / "polar_vote_circular_v28"
    / "best_weights.weights.h5"
)
V28_CROP_BOXES: Path = REPO_ROOT / "ml" / "data" / "rectified_crop_boxes_v5_all.csv"


def test_firmware_training_crop_box_matches_shared_geometry() -> None:
    """The fixed crop should stay anchored to the firmware training geometry."""
    crop_box = firmware_training_crop_box(224, 224)

    assert crop_box == (23.0, 57.0, 178.0, 180.0)


def test_firmware_polar_vote_tensor_matches_board_shape() -> None:
    """The firmware preprocessing path should produce a 224x224x7 int8 tensor."""
    source_image, source_kind = load_capture_image(CAPTURE_IMAGE)
    tensor = build_firmware_polar_vote_tensor(source_image)
    tensor_probe = probe_tensor("firmware_input", tensor)

    assert source_kind == "rgb"
    assert tensor.dtype == np.int8
    assert tensor.shape == (224, 224, 7)
    assert tensor_probe.byte_length == 224 * 224 * 7
    assert tensor_probe.crc32_hex.startswith("0x")


def test_training_style_polar_vote_tensor_matches_board_shape() -> None:
    """The offline parity tensor should use the same 224x224x7 layout."""
    source_image, source_kind = load_capture_image(CAPTURE_IMAGE)
    tensor = build_training_style_polar_vote_tensor(source_image)
    tensor_probe = probe_tensor("training_input", tensor)

    assert source_kind == "rgb"
    assert tensor.dtype == np.int8
    assert tensor.shape == (224, 224, 7)
    assert tensor_probe.byte_length == 224 * 224 * 7
    assert tensor_probe.crc32_hex.startswith("0x")


def test_training_style_polar_vote_float32_matches_board_shape() -> None:
    """The exact offline helper should build the float32 7-channel tensor."""
    source_image, source_kind = load_capture_image(CAPTURE_IMAGE)
    tensor = build_training_style_polar_vote_float32(source_image)
    tensor_probe = probe_tensor("training_input_float32", tensor)

    assert source_kind == "rgb"
    assert tensor.dtype == np.float32
    assert tensor.shape == (224, 224, 7)
    assert tensor_probe.byte_length == 224 * 224 * 7 * 4
    assert tensor_probe.crc32_hex.startswith("0x")


def test_exact_v28_offline_recipe_reproduces_saved_prediction() -> None:
    """The offline V28 recipe should match the saved hard-case prediction."""
    gauge_spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    source_image, _source_kind = load_capture_image(V28_SAMPLE_IMAGE)
    crop_box: tuple[float, float, float, float] | None = None
    with V28_CROP_BOXES.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["image_path"] == "ml/data/captured_images/capture_0073.png":
                crop_box = (
                    float(row["x0"]),
                    float(row["y0"]),
                    float(row["x1"]),
                    float(row["y1"]),
                )
                break
    assert crop_box is not None

    tensor = build_training_style_polar_vote_float32(
        source_image,
        crop_box_xyxy=crop_box,
        center_search_px=5,
        gauge_spec=gauge_spec,
    )
    model = build_polar_vote_v28_model()
    model.load_weights(V28_WEIGHTS)
    logits = model.predict(tensor[None, ...], verbose=0)[0]
    prediction = decode_circular_vote_logits(logits, gauge_spec)

    assert prediction == pytest.approx(45.951546, abs=1e-3)


def test_exact_v28_offline_recipe_needs_center_search() -> None:
    """The hard-case V28 recipe should depend on the 3x3 center sweep."""
    gauge_spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    source_image, _source_kind = load_capture_image(V28_SAMPLE_IMAGE)
    crop_box: tuple[float, float, float, float] | None = None
    with V28_CROP_BOXES.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["image_path"] == "ml/data/captured_images/capture_0073.png":
                crop_box = (
                    float(row["x0"]),
                    float(row["y0"]),
                    float(row["x1"]),
                    float(row["y1"]),
                )
                break
    assert crop_box is not None

    tensor = build_training_style_polar_vote_float32(
        source_image,
        crop_box_xyxy=crop_box,
        center_search_px=0,
        gauge_spec=gauge_spec,
    )
    model = build_polar_vote_v28_model()
    model.load_weights(V28_WEIGHTS)
    logits = model.predict(tensor[None, ...], verbose=0)[0]
    prediction = decode_circular_vote_logits(logits, gauge_spec)

    assert abs(prediction - 45.951546) > 10.0


def test_v28_scratch_buffers_live_in_npusram6() -> None:
    """The large exact V28 scratch buffers should stay out of main RAM."""
    app_ai_source = APP_AI_C.read_text(encoding="utf-8")

    assert "resized_rgb[224U * 224U * 3U] __attribute__((section(\".npusram6\")))" in app_ai_source
    assert "polar_luma[224U * 224U] __attribute__((section(\".npusram6\")))" in app_ai_source


def test_circular_vote_decode_tracks_the_midpoint_of_the_sweep() -> None:
    """A sharp peak at the sweep midpoint should decode to about 10C."""
    gauge_spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    logits = np.full(POLAR_VOTE_BINS, -128, dtype=np.int8)
    logits[168] = 127

    prediction = decode_circular_vote_logits(logits, gauge_spec)

    assert prediction == pytest.approx(10.0, abs=0.75)
