"""Tests for the TFLite-safe deployable SimCC model builders."""

from __future__ import annotations

from embedded_gauge_reading_tinyml.models_deploy import build_spatial_simcc_gauge_model


def test_spatial_simcc_gauge_model_has_spatial_heads() -> None:
    """The spatial SimCC deploy model should keep a 14x14 trunk and 1D heads."""

    model = build_spatial_simcc_gauge_model(
        input_shape=(224, 224, 3),
        alpha=0.35,
        num_bins=112,
        spatial_channels=64,
        pretrained=False,
    )

    assert [tuple(output.shape) for output in model.outputs] == [
        (None, 112),
        (None, 112),
        (None, 112),
        (None, 112),
        (None, 1),
    ]
    assert model.get_layer("spatial_trunk_up_1").__class__.__name__ == "UpSampling2D"
    assert model.get_layer("spatial_trunk_conv_1").__class__.__name__ == "Conv2D"
    assert model.get_layer("center_x_conv_1").__class__.__name__ == "Conv2D"
    assert model.get_layer("center_y_conv_1").__class__.__name__ == "Conv2D"
    assert model.get_layer("tip_x_conv_1").__class__.__name__ == "Conv2D"
    assert model.get_layer("tip_y_conv_1").__class__.__name__ == "Conv2D"
