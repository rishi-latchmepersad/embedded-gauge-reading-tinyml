"""Smoke tests for the center-detector distillation trainer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "ml" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_qat_obb_simcc_combined import CenterSimCCDistillTrainer  # noqa: E402
from train_qat_obb_simcc_combined import (  # noqa: E402
    CombinedGeometrySequence,
    CombinedSample,
    IMAGE_SIZE,
    _load_teacher_model,
    _select_combined_sample,
)
from embedded_gauge_reading_tinyml.firmware_preprocessing import (  # noqa: E402
    firmware_training_crop_box,
)
from embedded_gauge_reading_tinyml.obb_simcc_tf_models import (  # noqa: E402
    build_mobilenetv2_center_simcc_model,
)


def _one_hot_batch(indices: list[int], *, num_bins: int) -> tf.Tensor:
    """Build a small batch of one-hot SimCC targets."""

    vectors = [tf.one_hot(index, num_bins, dtype=tf.float32) for index in indices]
    return tf.stack(vectors, axis=0)


def test_center_simcc_distill_trainer_runs_one_step() -> None:
    """The custom KD trainer should execute one train step end to end."""

    student = build_mobilenetv2_center_simcc_model(
        image_shape=(224, 224, 3),
        alpha=0.35,
        pretrained=False,
        backbone_trainable=False,
        num_bins=112,
        spatial_channels=32,
        head_units=48,
        head_dropout=0.1,
    )
    trainer = CenterSimCCDistillTrainer(
        student,
        teacher_model=None,
        include_temperature_head=False,
    )
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    batch_size = 2
    x = tf.zeros((batch_size, 224, 224, 3), dtype=tf.float32)
    y = {
        "center_xy": tf.zeros((batch_size, 2), dtype=tf.float32),
        "center_x_simcc": _one_hot_batch([10, 11], num_bins=112),
        "center_y_simcc": _one_hot_batch([12, 13], num_bins=112),
        "tip_x_simcc": _one_hot_batch([14, 15], num_bins=112),
        "tip_y_simcc": _one_hot_batch([16, 17], num_bins=112),
    }
    w = {key: tf.ones((batch_size,), dtype=tf.float32) for key in y}

    dataset = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(batch_size)
    history = trainer.fit(dataset, epochs=1, steps_per_epoch=1, verbose=0)

    assert "loss" in history.history
    assert "center_xy_mae_px" in history.history


def test_combined_geometry_sequence_crops_and_pads_training_images_to_224() -> None:
    """The training sequence should crop, pad, and relabel into 224x224 space."""

    image_path = PROJECT_ROOT / "tmp" / "center_simcc_crop_pad_test.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    source = np.zeros((200, 400, 3), dtype=np.uint8)
    source[:, :, 0] = 32
    source[:, :, 1] = 96
    source[:, :, 2] = 160
    Image.fromarray(source, mode="RGB").save(image_path)

    try:
        sample = CombinedSample(
            image_path=image_path,
            source_width=400,
            source_height=200,
            crop_box_xyxy=(0.0, 0.0, 400.0, 200.0),
            center_xy=(0.25, 0.25),
            center_weight=1.0,
            tip_xy=(0.75, 0.75),
            tip_weight=1.0,
            temperature_c=9.0,
            temperature_weight=1.0,
            source_kinds=("test",),
        )
        sequence = CombinedGeometrySequence(
            [sample],
            batch_size=1,
            include_temperature_head=True,
            augment=False,
            seed=123,
        )

        batch_x, batch_y, batch_w = sequence[0]

        assert batch_x.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 3)
        assert batch_x.dtype == np.float32
        assert np.isfinite(batch_x).all()
        assert np.allclose(batch_x[0, 0, 0], -1.0)
        assert batch_y["center_xy"].shape == (1, 2)
        np.testing.assert_allclose(
            batch_y["center_xy"][0],
            np.array([0.25, 0.375], dtype=np.float32),
            atol=1e-6,
        )
        assert batch_w["center_xy"].shape == (1,)
    finally:
        image_path.unlink(missing_ok=True)


def test_combined_samples_use_the_shared_firmware_crop_box() -> None:
    """All grouped training rows should use the same firmware crop geometry."""

    pxl_entry = {
        "image_path": "ml/data/captured_images/fake_pxl_row.png",
        "annotations": [
            {
                "source_kind": "pxl_geometry",
                "source_row_index": 1,
                "source_row": {
                    "image_path": "ml/data/captured_images/fake_pxl_row.png",
                    "source_width": 400,
                    "source_height": 200,
                    "loose_crop_x1": 10.0,
                    "loose_crop_y1": 12.0,
                    "loose_crop_x2": 390.0,
                    "loose_crop_y2": 188.0,
                    "center_x_source": 120.0,
                    "center_y_source": 80.0,
                    "tip_x_source": 280.0,
                    "tip_y_source": 110.0,
                    "temperature_c": 9.0,
                    "true_angle_degrees": 297.0,
                },
            }
        ],
    }
    temperature_entry = {
        "image_path": "ml/data/captured_images/capture_0007.png",
        "annotations": [
            {
                "source_kind": "temperature_only",
                "source_row_index": 1,
                "source_row": {
                    "image_path": "ml/data/captured_images/capture_0007.png",
                    "source_width": 224,
                    "source_height": 224,
                    "value": 18.0,
                    "true_angle_degrees": 297.0,
                },
            }
        ],
    }

    pxl_sample = _select_combined_sample(pxl_entry, include_temperature_head=True)
    temperature_sample = _select_combined_sample(
        temperature_entry,
        include_temperature_head=True,
    )

    assert pxl_sample is not None
    assert temperature_sample is not None
    assert pxl_sample.crop_box_xyxy == firmware_training_crop_box(400, 200)
    assert temperature_sample.crop_box_xyxy == firmware_training_crop_box(224, 224)


def test_load_teacher_model_rejects_legacy_320_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stale 320x320 teacher should fail fast with a clear geometry error."""

    teacher_path = tmp_path / "legacy_teacher.keras"
    teacher_path.write_bytes(b"placeholder")

    class LegacyTeacher:
        """Minimal stand-in for a loaded Keras model with old input geometry."""

        input_shape = (None, 320, 320, 3)

    monkeypatch.setattr(
        "train_qat_obb_simcc_combined.keras.models.load_model",
        lambda *args, **kwargs: LegacyTeacher(),
    )

    with pytest.raises(ValueError, match="224x224"):
        _load_teacher_model(teacher_path)
