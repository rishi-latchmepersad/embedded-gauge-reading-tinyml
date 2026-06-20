"""Tests for geometry model builders."""

from __future__ import annotations

from embedded_gauge_reading_tinyml.models_geometry import (
    _build_mobilenetv2_geometry_heatmap_decoder,
    build_mobilenetv2_geometry_heatmap_v4_112,
    build_heatmap_angle_model,
)


def test_shared_112_heatmap_decoder_has_full_resolution_refine_block() -> None:
    """The shared 112x112 decoder should include the full-resolution refine layer."""

    model = _build_mobilenetv2_geometry_heatmap_decoder(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=True,
        heatmap_size=112,
        pretrained=False,
    )

    assert [tuple(output.shape) for output in model.outputs] == [
        (None, 112, 112, 1),
        (None, 112, 112, 1),
        (None, 1),
    ]
    refine_layer = model.get_layer("geometry_decoder_refine_fullres")
    assert refine_layer is not None
    assert getattr(refine_layer, "filters", None) == 32
    assert model.get_layer("geometry_decoder_up_4").interpolation == "bilinear"


def test_v4_112_geometry_heatmap_has_full_resolution_refine_block() -> None:
    """The v4 112x112 geometry head should sharpen the final spatial tensor."""

    model = build_mobilenetv2_geometry_heatmap_v4_112(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=True,
        include_aux_coords=False,
        aux_head_type="none",
        decoder_width_multiplier=1.0,
        pretrained=False,
    )

    assert [tuple(output.shape) for output in model.outputs] == [
        (None, 112, 112, 1),
        (None, 112, 112, 1),
        (None, 1),
    ]
    refine_layer = model.get_layer("geometry_decoder_refine_fullres")
    assert refine_layer is not None
    assert getattr(refine_layer, "filters", None) == 32
    assert model.get_layer("geometry_decoder_up_4").interpolation == "bilinear"


def test_v4_112_geometry_heatmap_can_use_transpose_final_upsample() -> None:
    """The v4 112x112 geometry head should support a learnable final resize."""

    model = build_mobilenetv2_geometry_heatmap_v4_112(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=True,
        include_aux_coords=False,
        aux_head_type="none",
        decoder_width_multiplier=1.0,
        decoder_upsample_mode="transpose",
        pretrained=False,
    )

    assert [tuple(output.shape) for output in model.outputs] == [
        (None, 112, 112, 1),
        (None, 112, 112, 1),
        (None, 1),
    ]
    refine_layer = model.get_layer("geometry_decoder_refine_fullres")
    assert refine_layer is not None
    assert getattr(refine_layer, "filters", None) == 32
    assert model.get_layer("geometry_decoder_up_4").__class__.__name__ == "Conv2DTranspose"


def test_v4_112_geometry_heatmap_can_use_hybrid_residual_final_upsample() -> None:
    """The v4 112x112 geometry head should support a stable learnable residual upsample."""

    model = build_mobilenetv2_geometry_heatmap_v4_112(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=True,
        include_aux_coords=False,
        aux_head_type="none",
        decoder_width_multiplier=1.0,
        decoder_upsample_mode="hybrid_residual",
        pretrained=False,
    )

    assert [tuple(output.shape) for output in model.outputs] == [
        (None, 112, 112, 1),
        (None, 112, 112, 1),
        (None, 1),
    ]
    refine_layer = model.get_layer("geometry_decoder_refine_fullres")
    assert refine_layer is not None
    assert getattr(refine_layer, "filters", None) == 32
    assert model.get_layer("geometry_decoder_up_4").__class__.__name__ == "Add"
    assert model.get_layer("geometry_decoder_up_4_bilinear").__class__.__name__ == "UpSampling2D"
    assert model.get_layer("geometry_decoder_up_4_residual").__class__.__name__ == "Conv2DTranspose"


def test_v4_112_geometry_heatmap_can_use_multiscale_fusion() -> None:
    """The v4 112x112 geometry head should support UNet-style multi-scale skip fusion."""

    model = build_mobilenetv2_geometry_heatmap_v4_112(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=True,
        include_aux_coords=False,
        aux_head_type="axis_simcc",
        decoder_width_multiplier=1.5,
        decoder_upsample_mode="hybrid_residual",
        decoder_multiscale_fusion=True,
        pretrained=False,
    )

    assert [tuple(output.shape) for output in model.outputs] == [
        (None, 112, 112, 1),
        (None, 112, 112, 1),
        (None, 1),
        (None, 4, 112),
    ]
    assert model.get_layer("geometry_decoder_fuse_14_add").__class__.__name__ == "Add"
    assert model.get_layer("geometry_decoder_fuse_28_add").__class__.__name__ == "Add"
    assert model.get_layer("geometry_decoder_fuse_56_add").__class__.__name__ == "Add"


def test_v4_112_geometry_heatmap_can_use_fullres_residual_sharpening() -> None:
    """The v4 112x112 geometry head should support a zero-start full-resolution residual branch."""

    model = build_mobilenetv2_geometry_heatmap_v4_112(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=True,
        include_aux_coords=False,
        aux_head_type="none",
        decoder_width_multiplier=1.0,
        decoder_upsample_mode="hybrid_residual",
        decoder_fullres_residual_block=True,
        pretrained=False,
    )

    assert [tuple(output.shape) for output in model.outputs] == [
        (None, 112, 112, 1),
        (None, 112, 112, 1),
        (None, 1),
    ]
    assert model.get_layer("geometry_decoder_refine_fullres_residual").__class__.__name__ == "Add"
    assert model.get_layer("geometry_decoder_refine_fullres_residual_conv_1").__class__.__name__ == "Conv2D"
    assert model.get_layer("geometry_decoder_refine_fullres_residual_conv_2").__class__.__name__ == "Conv2D"


def test_angle_heatmap_model_uses_bilinear_final_upsample() -> None:
    """The angle model should share the same bilinear final upsample behavior."""

    model = build_heatmap_angle_model(
        input_shape=(224, 224, 3),
        alpha=0.35,
        backbone_frozen=True,
        heatmap_size=112,
    )

    assert [tuple(output.shape) for output in model.outputs] == [
        (None, 112, 112, 1),
        (None, 112, 112, 1),
        (None, 1),
    ]
    assert model.get_layer("angle_decoder_up_4").interpolation == "bilinear"
    assert model.get_layer("angle_decoder_refine_fullres") is not None
