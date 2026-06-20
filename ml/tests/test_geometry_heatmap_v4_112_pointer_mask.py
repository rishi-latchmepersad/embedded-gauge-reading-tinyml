"""Tests for the pointer-mask supervision used by the v4.112 geometry trainer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_trainer_module():
    """Load the training script as a module without executing it as a job."""

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "ml" / "scripts" / "train_geometry_heatmap_v4_112_quant_native.py"
    spec = importlib.util.spec_from_file_location("train_geometry_heatmap_v4_112_quant_native", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pointer_mask_target_is_finite_and_has_expected_shape() -> None:
    """The generated mask target should stay finite and stay on-grid."""

    trainer = _load_trainer_module()
    pointer_mask = trainer._make_pointer_mask_targets(  # noqa: SLF001
        np.asarray([112.0], dtype=np.float32),
        np.asarray([112.0], dtype=np.float32),
        np.asarray([168.0], dtype=np.float32),
        np.asarray([112.0], dtype=np.float32),
        heatmap_size=112,
        sigma_px=1.6,
    )

    assert pointer_mask.shape == (1, 112, 112, 1)
    assert np.isfinite(pointer_mask).all()
    assert float(pointer_mask.max()) > 0.9


def test_pointer_mask_target_collapses_to_zero_for_degenerate_segment() -> None:
    """A zero-length segment should not invent a fake pointer mask."""

    trainer = _load_trainer_module()
    pointer_mask = trainer._make_pointer_mask_targets(  # noqa: SLF001
        np.asarray([112.0], dtype=np.float32),
        np.asarray([112.0], dtype=np.float32),
        np.asarray([112.0], dtype=np.float32),
        np.asarray([112.0], dtype=np.float32),
        heatmap_size=112,
        sigma_px=1.6,
    )

    assert pointer_mask.shape == (1, 112, 112, 1)
    assert np.isfinite(pointer_mask).all()
    assert float(pointer_mask.max()) == 0.0
