"""Tests for the combined center-detector + SimCC gauge model builder."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from embedded_gauge_reading_tinyml.obb_simcc_tf_models import (
    build_mobilenetv2_center_simcc_model,
)


def _prediction_tensors(outputs: object) -> list[tf.Tensor]:
    """Normalize list- or dict-shaped predictions into a tensor list."""

    if isinstance(outputs, dict):
        return [tf.convert_to_tensor(outputs[name]) for name in outputs]
    return [tf.convert_to_tensor(output) for output in outputs]  # type: ignore[arg-type]


def test_build_mobilenetv2_center_simcc_model_outputs_expected_heads() -> None:
    """The center-detector model should emit a center head plus SimCC heads."""

    model = build_mobilenetv2_center_simcc_model(
        image_shape=(224, 224, 3),
        alpha=0.35,
        pretrained=False,
        backbone_trainable=False,
        num_bins=112,
        spatial_channels=64,
        head_units=96,
        head_dropout=0.15,
    )

    assert model.name == "mobilenetv2_center_simcc_gauge"
    assert set(model.output_names) == {
        "center_xy",
        "center_x_simcc",
        "center_y_simcc",
        "tip_x_simcc",
        "tip_y_simcc",
    }
    assert model.get_layer("center_xy").__class__.__name__ == "Dense"
    assert tuple(model.get_layer("center_xy").output.shape) == (None, 2)
    assert model.get_layer("center_x_simcc").__class__.__name__ == "Softmax"
    assert tuple(model.get_layer("center_x_simcc").output.shape) == (None, 112)
    assert model.get_layer("tip_y_simcc").__class__.__name__ == "Softmax"


def test_build_mobilenetv2_center_simcc_model_is_qat_cloneable() -> None:
    """The center-detector model should survive tfmot quantization cloning."""

    model = build_mobilenetv2_center_simcc_model(
        image_shape=(224, 224, 3),
        alpha=0.35,
        pretrained=False,
        backbone_trainable=False,
        num_bins=112,
        spatial_channels=64,
        head_units=96,
        head_dropout=0.15,
    )

    qat_model = tfmot.quantization.keras.quantize_model(model)
    batch = tf.zeros((2, 224, 224, 3), dtype=tf.float32)
    outputs = qat_model(batch)

    if isinstance(outputs, dict):
        assert tuple(outputs["center_xy"].shape) == (2, 2)
        assert tuple(outputs["center_x_simcc"].shape) == (2, 112)
        assert tuple(outputs["center_y_simcc"].shape) == (2, 112)
        assert tuple(outputs["tip_x_simcc"].shape) == (2, 112)
        assert tuple(outputs["tip_y_simcc"].shape) == (2, 112)
    else:
        tensors = _prediction_tensors(outputs)
        assert [tuple(output.shape) for output in tensors] == [
            (2, 2),
            (2, 112),
            (2, 112),
            (2, 112),
            (2, 112),
        ]
