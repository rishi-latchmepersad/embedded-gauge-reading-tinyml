"""Tests for the keypoint-gated reader cascade helpers."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pytest

import embedded_gauge_reading_tinyml.geometry_cascade as geometry_cascade


@dataclass
class _SequenceModel:
    """A tiny callable model stub that returns a fixed sequence of outputs."""

    outputs: list[object]
    calls: int = 0

    def __call__(self, inputs: object, training: bool = False) -> object:
        del inputs, training
        if self.calls >= len(self.outputs):
            raise AssertionError("Model was called more times than expected.")
        output = self.outputs[self.calls]
        self.calls += 1
        return output


def _patch_identity_resize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the TensorFlow resize helper with a fast identity path for tests."""
    monkeypatch.setattr(
        geometry_cascade,
        "resize_with_pad_rgb",
        lambda image, crop_box_xyxy, image_size=224: np.asarray(image, dtype=np.uint8),
    )


def _make_heatmaps(*, peak: bool) -> np.ndarray:
    """Build a tiny two-keypoint heatmap tensor for confidence tests."""
    heatmaps = np.full((4, 4, 2), 0.2, dtype=np.float32)
    if peak:
        heatmaps[1, 2, 0] = 4.0
        heatmaps[2, 1, 1] = 4.0
    return heatmaps


def test_heatmap_confidence_prefers_sharp_peaks() -> None:
    """A peaked heatmap should score higher than a flat one."""
    low = geometry_cascade.heatmap_confidence(_make_heatmaps(peak=False))
    high = geometry_cascade.heatmap_confidence(_make_heatmaps(peak=True))

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_run_geometry_cascade_keeps_first_pass_when_confident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confident localizer should skip the second crop."""
    _patch_identity_resize(monkeypatch)
    model = _SequenceModel(
        outputs=[
            {
                "gauge_value": np.array([[17.0]], dtype=np.float32),
                "keypoint_heatmaps": _make_heatmaps(peak=True)[None, ...],
                "keypoint_coords": np.array(
                    [[[110.0, 110.0], [130.0, 110.0]]], dtype=np.float32
                ),
            }
        ]
    )
    source_image = np.zeros((224, 224, 3), dtype=np.uint8)

    result = geometry_cascade.run_geometry_cascade(
        model=model,  # type: ignore[arg-type]
        source_image=source_image,
        base_crop_box_xyxy=(0.0, 0.0, 224.0, 224.0),
        image_height=224,
        image_width=224,
        confidence_threshold=0.5,
        recrop_scale=0.75,
        min_recrop_size=64.0,
    )

    assert result.used_second_pass is False
    assert result.second_pass is None
    assert result.final_value == pytest.approx(17.0)
    assert model.calls == 1


def test_run_geometry_cascade_uses_reader_on_second_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A low-confidence first pass should trigger a tighter second crop."""
    _patch_identity_resize(monkeypatch)
    localizer = _SequenceModel(
        outputs=[
            {
                "gauge_value": np.array([[11.0]], dtype=np.float32),
                "keypoint_heatmaps": _make_heatmaps(peak=False)[None, ...],
                "keypoint_coords": np.array(
                    [[[100.0, 100.0], [124.0, 100.0]]], dtype=np.float32
                ),
            },
            {
                "gauge_value": np.array([[22.0]], dtype=np.float32),
                "keypoint_heatmaps": _make_heatmaps(peak=True)[None, ...],
                "keypoint_coords": np.array(
                    [[[108.0, 108.0], [140.0, 108.0]]], dtype=np.float32
                ),
            },
        ]
    )
    reader = _SequenceModel(
        outputs=[
            np.array([[31.0]], dtype=np.float32),
            np.array([[44.0]], dtype=np.float32),
        ]
    )
    source_image = np.zeros((224, 224, 3), dtype=np.uint8)

    result = geometry_cascade.run_geometry_cascade(
        model=localizer,  # type: ignore[arg-type]
        reader_model=reader,  # type: ignore[arg-type]
        source_image=source_image,
        base_crop_box_xyxy=(0.0, 0.0, 224.0, 224.0),
        image_height=224,
        image_width=224,
        confidence_threshold=0.5,
        recrop_scale=0.75,
        min_recrop_size=64.0,
    )

    assert result.used_second_pass is True
    assert result.second_pass is not None
    assert result.first_pass.value == pytest.approx(31.0)
    assert result.second_pass.value == pytest.approx(44.0)
    assert result.final_value == pytest.approx(44.0)
    assert localizer.calls == 2
    assert reader.calls == 2
