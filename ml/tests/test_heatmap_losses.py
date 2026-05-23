"""Tests for heatmap losses and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from embedded_gauge_reading_tinyml.heatmap_losses import (
    center_priority_heatmap_loss,
    combined_heatmap_loss,
    mean_predicted_heatmap_peak,
    softargmax_coordinate_loss,
    softargmax_coordinate_mae,
    weighted_center_heatmap_loss,
    weighted_heatmap_bce_loss,
    weighted_heatmap_mse_loss,
    weighted_tip_heatmap_loss,
    tip_priority_heatmap_loss,
)
from embedded_gauge_reading_tinyml.heatmap_utils import make_gaussian_heatmap


def _make_batch_heatmap(
    *,
    x_normalized: float,
    y_normalized: float,
    sigma_pixels: float = 2.5,
) -> np.ndarray:
    """Create a single-sample heatmap batch with a singleton channel dimension."""

    heatmap = make_gaussian_heatmap(56, 56, x_normalized, y_normalized, sigma_pixels)
    return heatmap[np.newaxis, ..., np.newaxis].astype(np.float32)


def test_weighted_heatmap_mse_is_zero_for_identical_heatmaps() -> None:
    """Matching heatmaps should produce zero weighted MSE."""

    y_true = _make_batch_heatmap(x_normalized=0.5, y_normalized=0.5)
    loss = weighted_heatmap_mse_loss(y_true, y_true).numpy()
    assert loss == pytest.approx(0.0, abs=1e-7)


def test_weighted_heatmap_mse_penalizes_peak_shift() -> None:
    """A shifted peak should produce a non-trivial weighted MSE."""

    y_true = _make_batch_heatmap(x_normalized=0.5, y_normalized=0.5)
    y_pred = _make_batch_heatmap(x_normalized=0.55, y_normalized=0.5)
    loss = weighted_heatmap_mse_loss(y_true, y_pred).numpy()
    assert loss > 0.001


def test_weighted_heatmap_bce_is_finite() -> None:
    """Weighted BCE should return a finite scalar for soft targets."""

    y_true = _make_batch_heatmap(x_normalized=0.35, y_normalized=0.65)
    y_pred = _make_batch_heatmap(x_normalized=0.36, y_normalized=0.64)
    loss = weighted_heatmap_bce_loss(y_true, y_pred).numpy()
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_softargmax_coordinate_loss_is_zero_for_identical_heatmaps() -> None:
    """Softargmax coordinate loss should vanish when heatmaps match."""

    y_true = _make_batch_heatmap(x_normalized=0.25, y_normalized=0.75)
    loss = softargmax_coordinate_loss(y_true, y_true).numpy()
    assert loss == pytest.approx(0.0, abs=1e-7)


def test_softargmax_coordinate_loss_increases_for_shifted_peak() -> None:
    """Softargmax coordinate loss should detect a moved peak."""

    y_true = _make_batch_heatmap(x_normalized=0.25, y_normalized=0.75)
    y_pred = _make_batch_heatmap(x_normalized=0.30, y_normalized=0.75)
    loss = softargmax_coordinate_loss(y_true, y_pred).numpy()
    assert loss > 0.5


def test_softargmax_coordinate_mae_tracks_shift() -> None:
    """Coordinate MAE should be small for identical maps and larger for shifts."""

    y_true = _make_batch_heatmap(x_normalized=0.40, y_normalized=0.20)
    y_pred = _make_batch_heatmap(x_normalized=0.44, y_normalized=0.20)
    mae = softargmax_coordinate_mae(y_true, y_pred).numpy()
    assert mae > 0.5


def test_combined_heatmap_loss_is_positive_for_shifted_peak() -> None:
    """The combined heatmap loss should be positive when the peak is wrong."""

    y_true = _make_batch_heatmap(x_normalized=0.60, y_normalized=0.40)
    y_pred = _make_batch_heatmap(x_normalized=0.64, y_normalized=0.40)
    loss = combined_heatmap_loss(y_true, y_pred).numpy()
    assert loss > 0.5


def test_center_priority_heatmap_loss_is_zero_for_identical_heatmaps() -> None:
    """The center-priority loss should vanish on identical targets."""

    y_true = _make_batch_heatmap(x_normalized=0.55, y_normalized=0.45)
    loss = center_priority_heatmap_loss(y_true, y_true).numpy()
    assert loss == pytest.approx(0.0, abs=1e-7)


def test_tip_priority_heatmap_loss_is_positive_for_shifted_peak() -> None:
    """The tip-priority loss should respond to a peak shift."""

    y_true = _make_batch_heatmap(x_normalized=0.40, y_normalized=0.60)
    y_pred = _make_batch_heatmap(x_normalized=0.45, y_normalized=0.60)
    loss = tip_priority_heatmap_loss(y_true, y_pred).numpy()
    assert loss > 0.5


def test_weighted_center_heatmap_loss_is_positive_for_shifted_peak() -> None:
    """The v2 center loss should still penalize a displaced peak."""

    y_true = _make_batch_heatmap(x_normalized=0.50, y_normalized=0.50)
    y_pred = _make_batch_heatmap(x_normalized=0.54, y_normalized=0.50)
    loss = weighted_center_heatmap_loss(y_true, y_pred).numpy()
    assert loss > 0.5


def test_weighted_tip_heatmap_loss_is_zero_for_identical_heatmaps() -> None:
    """The v2 tip loss should vanish on identical heatmaps."""

    y_true = _make_batch_heatmap(x_normalized=0.35, y_normalized=0.65)
    loss = weighted_tip_heatmap_loss(y_true, y_true).numpy()
    assert loss == pytest.approx(0.0, abs=1e-7)


def test_mean_predicted_heatmap_peak_reports_peak_value() -> None:
    """Peak metric should reflect the maximum heatmap activation."""

    y_pred = _make_batch_heatmap(x_normalized=0.50, y_normalized=0.50)
    peak = mean_predicted_heatmap_peak(y_pred, y_pred).numpy()
    assert peak > 0.9
