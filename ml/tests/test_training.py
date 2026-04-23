"""Unit tests for training utilities and training orchestration."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest
import tensorflow as tf

import embedded_gauge_reading_tinyml.training as training
from embedded_gauge_reading_tinyml.models import (
    build_compact_interval_model,
    build_compact_geometry_model,
    build_mobilenetv2_direction_model,
    build_mobilenetv2_geometry_uncertainty_model,
    build_mobilenetv2_obb_model,
    build_mobilenetv2_rectifier_model,
    build_mobilenetv2_regression_model,
)
from embedded_gauge_reading_tinyml.dataset import EllipseLabel, PointLabel, Sample
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec
from embedded_gauge_reading_tinyml.labels import LabelSummary


def _make_spec(gauge_id: str = "littlegood_home_temp_gauge_c") -> GaugeSpec:
    """Build a deterministic GaugeSpec for tests."""
    return GaugeSpec(
        gauge_id=gauge_id,
        min_angle_rad=0.0,
        sweep_rad=np.pi,
        min_value=-30.0,
        max_value=50.0,
    )


def _make_sample(
    image_path: Path,
    *,
    tip_x: float,
    tip_y: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    dial_cx: float = 100.0,
    dial_cy: float = 120.0,
    dial_rx: float = 20.0,
    dial_ry: float = 10.0,
) -> Sample:
    """Create a synthetic Sample with dial + center + tip labels."""
    dial: EllipseLabel = EllipseLabel(
        cx=dial_cx,
        cy=dial_cy,
        rx=dial_rx,
        ry=dial_ry,
        rotation=0.0,
        label="temp_dial",
    )
    center: PointLabel = PointLabel(x=center_x, y=center_y, label="temp_center")
    tip: PointLabel = PointLabel(x=tip_x, y=tip_y, label="temp_tip")
    return Sample(image_path=image_path, dial=dial, center=center, tip=tip)


def _write_test_jpeg(path: Path, *, h: int = 16, w: int = 16) -> None:
    """Write a tiny JPEG image to disk for input-pipeline tests."""
    image_np: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
    image_np[..., 0] = 64
    image_np[..., 1] = 128
    image_np[..., 2] = 192
    encoded: tf.Tensor = tf.io.encode_jpeg(tf.constant(image_np, dtype=tf.uint8))
    tf.io.write_file(str(path), encoded)


def _write_test_png(path: Path, *, h: int = 16, w: int = 16) -> None:
    """Write a tiny PNG image to disk for manifest-loader tests."""
    image_np: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
    image_np[..., 0] = 192
    image_np[..., 1] = 96
    image_np[..., 2] = 32
    encoded: tf.Tensor = tf.io.encode_png(tf.constant(image_np, dtype=tf.uint8))
    tf.io.write_file(str(path), encoded)


def _write_manifest_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    """Write a tiny image/value manifest for split-routing tests."""
    lines = ["image_path,value"]
    for image_path, value in rows:
        lines.append(f"{image_path},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_split_config_accepts_valid_values() -> None:
    """_validate_split_config should not raise for valid fractions."""
    config: training.TrainConfig = training.TrainConfig(
        val_fraction=0.2, test_fraction=0.2
    )
    training._validate_split_config(config)


def test_train_config_defaults_match_strong_mobilenetv2_preset() -> None:
    """TrainConfig defaults should match the strongest MobileNetV2 preset."""
    config: training.TrainConfig = training.TrainConfig()

    assert config.gauge_id == "littlegood_home_temp_gauge_c"
    assert config.image_height == 224
    assert config.image_width == 224
    assert config.batch_size == 8
    assert config.epochs == 40
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.seed == 21
    assert config.strict_labels is False
    assert config.model_family == "mobilenet_v2"
    assert config.mobilenet_pretrained is True
    assert config.mobilenet_backbone_trainable is True
    assert config.mobilenet_warmup_epochs == 8
    assert config.mobilenet_alpha == pytest.approx(1.0)
    assert config.mobilenet_head_units == 128
    assert config.mobilenet_head_dropout == pytest.approx(0.2)
    assert config.geometry_uncertainty_loss_weight == pytest.approx(0.25)
    assert config.geometry_uncertainty_low_quantile == pytest.approx(0.10)
    assert config.geometry_uncertainty_high_quantile == pytest.approx(0.90)
    assert config.test_manifest is None
    assert config.init_model_path is None


def test_build_mobilenetv2_tiny_regression_model_is_smaller() -> None:
    """The tiny MobileNetV2 variant should keep the same output contract."""
    standard = build_mobilenetv2_regression_model(
        image_height=224,
        image_width=224,
        pretrained=False,
        backbone_trainable=False,
    )
    tiny = build_mobilenetv2_regression_model(
        image_height=224,
        image_width=224,
        pretrained=False,
        backbone_trainable=False,
        alpha=0.35,
        head_units=64,
        head_dropout=0.15,
    )

    assert tiny.name == "mobilenetv2_gauge_regressor_a035_h064"
    assert tiny.output_shape == (None, 1)
    assert tiny.count_params() < standard.count_params()


def test_build_compact_interval_model_outputs_scalar_and_bins() -> None:
    """Compact interval CNN should emit both gauge_value and interval logits."""
    model = build_compact_interval_model(
        image_height=32,
        image_width=32,
        value_min=-30.0,
        value_max=50.0,
        bin_width=10.0,
    )
    batch: tf.Tensor = tf.zeros((2, 32, 32, 3), dtype=tf.float32)
    outputs = model(batch)

    assert model.name == "compact_interval_gauge_regressor"
    assert outputs["gauge_value"].shape == (2, 1)
    assert outputs["interval_logits"].shape == (2, 8)


def test_build_compact_geometry_model_outputs_geometry_tensors() -> None:
    """Compact geometry CNN should emit scalar, heatmaps, and keypoint coords."""
    model = build_compact_geometry_model(
        image_height=32,
        image_width=32,
        value_min=-30.0,
        value_max=50.0,
        min_angle_rad=0.0,
        sweep_rad=np.pi,
        heatmap_size=16,
    )
    batch: tf.Tensor = tf.zeros((2, 32, 32, 3), dtype=tf.float32)
    outputs = model(batch)

    assert model.name == "compact_geometry_gauge_regressor"
    assert outputs["gauge_value"].shape == (2, 1)
    assert outputs["keypoint_heatmaps"].shape == (2, 16, 16, 2)
    assert outputs["keypoint_coords"].shape == (2, 2, 2)


def test_build_mobilenetv2_geometry_uncertainty_model_outputs_bounds() -> None:
    """Geometry-uncertainty CNN should emit scalar, heatmaps, coords, and bounds."""
    model = build_mobilenetv2_geometry_uncertainty_model(
        image_height=32,
        image_width=32,
        value_min=-30.0,
        value_max=50.0,
        min_angle_rad=0.0,
        sweep_rad=np.pi,
        heatmap_size=16,
        pretrained=False,
        backbone_trainable=False,
    )
    batch: tf.Tensor = tf.zeros((2, 32, 32, 3), dtype=tf.float32)
    outputs = model(batch)

    assert model.name == "mobilenetv2_geometry_uncertainty_regressor"
    assert outputs["gauge_value"].shape == (2, 1)
    assert outputs["keypoint_heatmaps"].shape == (2, 16, 16, 2)
    assert outputs["keypoint_coords"].shape == (2, 2, 2)
    assert outputs["gauge_value_lower"].shape == (2, 1)
    assert outputs["gauge_value_upper"].shape == (2, 1)


def test_build_mobilenetv2_rectifier_model_outputs_box() -> None:
    """Rectifier CNN should emit a normalized crop-box prediction."""
    model = build_mobilenetv2_rectifier_model(
        image_height=32,
        image_width=32,
        pretrained=False,
        backbone_trainable=False,
    )
    batch: tf.Tensor = tf.zeros((2, 32, 32, 3), dtype=tf.float32)
    outputs = model(batch)

    assert model.name == "mobilenetv2_rectifier_regressor"
    assert outputs["rectifier_box"].shape == (2, 4)


def test_build_mobilenetv2_obb_model_outputs_params() -> None:
    """OBB localizer should emit a compact oriented-box parameter vector."""
    model = build_mobilenetv2_obb_model(
        image_height=32,
        image_width=32,
        pretrained=False,
        backbone_trainable=False,
    )
    batch: tf.Tensor = tf.zeros((2, 32, 32, 3), dtype=tf.float32)
    outputs = model(batch)

    assert model.name == "mobilenetv2_obb_regressor"
    assert outputs["obb_params"].shape == (2, 6)


def test_load_crop_with_obb_target_wraps_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OBB crop loaders should return dict targets keyed by obb_params."""

    def _fake_loader(
        image_path: tf.Tensor,
        value: tf.Tensor,
        crop_box_xyxy: tf.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        _ = (image_path, value, crop_box_xyxy, image_height, image_width)
        return tf.zeros((4, 4, 3), dtype=tf.float32), tf.constant([0.5], dtype=tf.float32)

    monkeypatch.setattr(
        training, "_load_crop_and_preprocess_image", _fake_loader
    )

    image, target = training._load_crop_with_obb_target(
        tf.constant("a.jpg"),
        tf.constant(10.0),
        tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=tf.float32),
        tf.constant([0.0, 0.0, 8.0, 8.0], dtype=tf.float32),
        32,
        32,
    )
    image_w, target_w, weight = training._load_crop_with_obb_weight(
        tf.constant("a.jpg"),
        tf.constant(10.0),
        tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=tf.float32),
        tf.constant([0.0, 0.0, 8.0, 8.0], dtype=tf.float32),
        32,
        32,
        tf.constant(0.25, dtype=tf.float32),
    )

    assert image.shape == (4, 4, 3)
    assert target["obb_params"].shape == (6,)
    assert image_w.shape == (4, 4, 3)
    assert target_w["obb_params"].shape == (6,)
    assert weight["obb_params"].shape == ()
    assert float(weight["obb_params"]) == pytest.approx(0.25)


def test_train_happy_path_with_mobilenetv2_obb_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train should route through the MobileNetV2 OBB family."""
    spec: GaugeSpec = _make_spec()

    label_summary: LabelSummary = LabelSummary(
        total_samples=10,
        in_sweep=9,
        out_of_sweep=1,
        min_fraction=0.0,
        max_fraction=1.0,
    )

    examples: list[training.TrainingExample] = [
        training.TrainingExample("a.jpg", 1.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("b.jpg", 2.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("c.jpg", 3.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("d.jpg", 4.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
    ]
    split: training.DatasetSplit = training.DatasetSplit(
        train_examples=examples[:2],
        val_examples=examples[2:3],
        test_examples=examples[3:],
    )

    class _FakeHistory:
        """Small stand-in for keras.callbacks.History."""

        def __init__(self) -> None:
            self.history: dict[str, list[float]] = {"loss": [1.0], "val_loss": [1.2]}

    class _FakeModel:
        """Small stand-in for a compiled OBB localizer."""

        def compile(self, **kwargs: Any) -> None:
            _ = kwargs

        def fit(
            self,
            train_ds: Any,
            validation_data: Any,
            epochs: int,
            callbacks: list[Any],
            verbose: int,
        ) -> _FakeHistory:
            assert train_ds is not None
            assert validation_data is not None
            assert epochs == 1
            assert len(callbacks) == 2
            assert verbose == 2
            return _FakeHistory()

        def evaluate(
            self, test_ds: Any, return_dict: bool, verbose: int
        ) -> dict[str, float]:
            assert test_ds is not None
            assert return_dict is True
            assert verbose == 0
            return {"loss": 1.0, "mae": 0.5, "rmse": 0.75}

    fake_model = _FakeModel()

    def _build_tf_dataset_stub(
        examples: list[training.TrainingExample],
        config: training.TrainConfig,
        training: bool,
        target_kind: str = "value",
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert target_kind == "obb"
        return {"n": len(examples), "train": training, "target_kind": target_kind}

    monkeypatch.setattr(training, "load_gauge_specs", lambda: {spec.gauge_id: spec})
    monkeypatch.setattr(
        training, "load_dataset", lambda labelled_dir, raw_dir: [object()] * 10
    )
    monkeypatch.setattr(
        training, "summarize_label_sweep", lambda samples, spec: label_summary
    )
    monkeypatch.setattr(
        training,
        "_build_training_examples",
        lambda samples, spec, **kwargs: (examples, 1),
    )
    monkeypatch.setattr(training, "_split_examples", lambda examples, config: split)
    monkeypatch.setattr(training, "_build_tf_dataset", _build_tf_dataset_stub)
    monkeypatch.setattr(
        training,
        "build_mobilenetv2_obb_model",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        training,
        "_compute_mean_baseline_mae",
        lambda train_examples, test_examples: 2.5,
    )

    result: training.TrainingResult = training.train(
        training.TrainConfig(
            epochs=1,
            model_family="mobilenet_v2_obb",
            mobilenet_backbone_trainable=False,
            mobilenet_warmup_epochs=0,
        )
    )

    assert result.model is fake_model
    assert result.label_summary == label_summary
    assert result.dropped_out_of_sweep == 1
    assert result.baseline_test_mae == pytest.approx(2.5)
    assert result.test_metrics == {
        "loss": 1.0,
        "mae": 0.5,
        "rmse": 0.75,
        "baseline_mae_mean_predictor": 2.5,
    }


def test_train_happy_path_with_mobilenetv2_geometry_uncertainty_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train should route through the geometry-uncertainty MobileNetV2 family."""
    spec: GaugeSpec = _make_spec()

    label_summary: LabelSummary = LabelSummary(
        total_samples=10,
        in_sweep=9,
        out_of_sweep=1,
        min_fraction=0.0,
        max_fraction=1.0,
    )

    examples: list[training.TrainingExample] = [
        training.TrainingExample("a.jpg", 1.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("b.jpg", 2.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("c.jpg", 3.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("d.jpg", 4.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
    ]
    split: training.DatasetSplit = training.DatasetSplit(
        train_examples=examples[:2],
        val_examples=examples[2:3],
        test_examples=examples[3:],
    )

    class _FakeHistory:
        """Small stand-in for keras.callbacks.History."""

        def __init__(self) -> None:
            self.history: dict[str, list[float]] = {"loss": [1.0], "val_loss": [1.2]}

    class _FakeModel:
        """Small stand-in for a compiled geometry-uncertainty model."""

        def compile(self, **kwargs: Any) -> None:
            _ = kwargs

        def fit(
            self,
            train_ds: Any,
            validation_data: Any,
            epochs: int,
            callbacks: list[Any],
            verbose: int,
        ) -> _FakeHistory:
            assert train_ds is not None
            assert validation_data is not None
            assert epochs == 1
            assert len(callbacks) == 2
            assert verbose == 2
            return _FakeHistory()

        def evaluate(
            self, test_ds: Any, return_dict: bool, verbose: int
        ) -> dict[str, float]:
            assert test_ds is not None
            assert return_dict is True
            assert verbose == 0
            return {"loss": 1.0, "mae": 0.5, "rmse": 0.75}

    fake_model = _FakeModel()

    def _build_tf_dataset_stub(
        examples: list[training.TrainingExample],
        config: training.TrainConfig,
        training: bool,
        target_kind: str = "value",
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert target_kind == "geometry_uncertainty"
        assert kwargs.get("value_range") == (-30.0, 50.0)
        return {"n": len(examples), "train": training, "target_kind": target_kind}

    monkeypatch.setattr(training, "load_gauge_specs", lambda: {spec.gauge_id: spec})
    monkeypatch.setattr(
        training, "load_dataset", lambda labelled_dir, raw_dir: [object()] * 10
    )
    monkeypatch.setattr(
        training, "summarize_label_sweep", lambda samples, spec: label_summary
    )
    monkeypatch.setattr(
        training,
        "_build_training_examples",
        lambda samples, spec, **kwargs: (examples, 1),
    )
    monkeypatch.setattr(training, "_split_examples", lambda examples, config: split)
    monkeypatch.setattr(training, "_build_tf_dataset", _build_tf_dataset_stub)
    monkeypatch.setattr(
        training,
        "build_mobilenetv2_geometry_uncertainty_model",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        training,
        "_compute_mean_baseline_mae",
        lambda train_examples, test_examples: 2.5,
    )

    result: training.TrainingResult = training.train(
        training.TrainConfig(epochs=1, model_family="mobilenet_v2_geometry_uncertainty")
    )

    assert result.model is fake_model
    assert result.label_summary == label_summary
    assert result.dropped_out_of_sweep == 1
    assert result.baseline_test_mae == pytest.approx(2.5)
    assert result.test_metrics == {
        "loss": 1.0,
        "mae": 0.5,
        "rmse": 0.75,
        "baseline_mae_mean_predictor": 2.5,
    }


def test_train_happy_path_with_mobilenetv2_rectifier_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train should route through the MobileNetV2 rectifier family."""
    spec: GaugeSpec = _make_spec()

    label_summary: LabelSummary = LabelSummary(
        total_samples=10,
        in_sweep=9,
        out_of_sweep=1,
        min_fraction=0.0,
        max_fraction=1.0,
    )

    examples: list[training.TrainingExample] = [
        training.TrainingExample("a.jpg", 1.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("b.jpg", 2.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("c.jpg", 3.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("d.jpg", 4.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
    ]
    split: training.DatasetSplit = training.DatasetSplit(
        train_examples=examples[:2],
        val_examples=examples[2:3],
        test_examples=examples[3:],
    )

    class _FakeHistory:
        """Small stand-in for keras.callbacks.History."""

        def __init__(self) -> None:
            self.history: dict[str, list[float]] = {"loss": [1.0], "val_loss": [1.2]}

    class _FakeModel:
        """Small stand-in for a compiled rectifier model."""

        def compile(self, **kwargs: Any) -> None:
            _ = kwargs

        def fit(
            self,
            train_ds: Any,
            validation_data: Any,
            epochs: int,
            callbacks: list[Any],
            verbose: int,
        ) -> _FakeHistory:
            assert train_ds is not None
            assert validation_data is not None
            assert epochs == 1
            assert len(callbacks) == 2
            assert verbose == 2
            return _FakeHistory()

        def evaluate(
            self, test_ds: Any, return_dict: bool, verbose: int
        ) -> dict[str, float]:
            assert test_ds is not None
            assert return_dict is True
            assert verbose == 0
            return {"loss": 1.0, "mae": 0.5, "rmse": 0.75}

    fake_model = _FakeModel()

    def _build_tf_dataset_stub(
        examples: list[training.TrainingExample],
        config: training.TrainConfig,
        training: bool,
        target_kind: str = "value",
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert target_kind == "rectifier_box"
        value_range = kwargs.get("value_range")
        assert value_range is None or (
            isinstance(value_range, tuple) and len(value_range) == 2
        )
        return {"n": len(examples), "train": training, "target_kind": target_kind}

    monkeypatch.setattr(training, "load_gauge_specs", lambda: {spec.gauge_id: spec})
    monkeypatch.setattr(
        training, "load_dataset", lambda labelled_dir, raw_dir: [object()] * 10
    )
    monkeypatch.setattr(
        training, "summarize_label_sweep", lambda samples, spec: label_summary
    )
    monkeypatch.setattr(
        training,
        "_build_training_examples",
        lambda samples, spec, **kwargs: (examples, 1),
    )
    monkeypatch.setattr(training, "_split_examples", lambda examples, config: split)
    monkeypatch.setattr(training, "_build_tf_dataset", _build_tf_dataset_stub)
    monkeypatch.setattr(
        training,
        "build_mobilenetv2_rectifier_model",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        training,
        "_compute_mean_baseline_mae",
        lambda train_examples, test_examples: 2.5,
    )

    result: training.TrainingResult = training.train(
        training.TrainConfig(epochs=1, model_family="mobilenet_v2_rectifier")
    )

    assert result.model is fake_model
    assert result.label_summary == label_summary
    assert result.dropped_out_of_sweep == 1
    assert result.baseline_test_mae == pytest.approx(2.5)
    assert result.test_metrics == {
        "loss": 1.0,
        "mae": 0.5,
        "rmse": 0.75,
        "baseline_mae_mean_predictor": 2.5,
    }


def test_build_mobilenetv2_direction_model_outputs_unit_vector() -> None:
    """Direction CNN should emit a 2D normalized needle vector."""
    model = build_mobilenetv2_direction_model(
        image_height=224,
        image_width=224,
        pretrained=False,
        backbone_trainable=False,
    )

    assert model.name == "mobilenetv2_needle_direction_regressor"
    assert model.output_shape == (None, 2)


def test_load_hard_case_examples_reads_manifest_and_uses_full_image_crop(
    tmp_path: Path,
) -> None:
    """Hard-case manifest rows should load as scalar training examples."""
    image_path: Path = tmp_path / "capture_0c_preview.png"
    _write_test_png(image_path)

    manifest_path: Path = tmp_path / "hard_cases.csv"
    manifest_path.write_text(
        "image_path,value\n"
        f"{image_path},0\n",
        encoding="utf-8",
    )

    examples = training._load_hard_case_examples(
        manifest_path,
        image_height=224,
        image_width=224,
    )

    assert len(examples) == 1
    assert examples[0].image_path == str(image_path)
    assert examples[0].value == pytest.approx(0.0)
    assert examples[0].crop_box_xyxy == pytest.approx((0.0, 0.0, 224.0, 224.0))


@pytest.mark.parametrize(
    ("config", "expected_error"),
    [
        (
            training.TrainConfig(val_fraction=0.0, test_fraction=0.2),
            "val_fraction must be in (0, 1).",
        ),
        (
            training.TrainConfig(val_fraction=0.2, test_fraction=1.0),
            "test_fraction must be in (0, 1).",
        ),
        (
            training.TrainConfig(val_fraction=0.6, test_fraction=0.4),
            "val_fraction + test_fraction must be < 1.0.",
        ),
    ],
)
def test_validate_split_config_rejects_invalid_values(
    config: training.TrainConfig,
    expected_error: str,
) -> None:
    """_validate_split_config should reject invalid split configurations."""
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        training._validate_split_config(config)


def test_compute_crop_box_uses_ellipse_and_padding(tmp_path: Path) -> None:
    """_compute_crop_box should create padded xyxy crop around the dial."""
    sample: Sample = _make_sample(
        tmp_path / "a.jpg",
        tip_x=1.0,
        tip_y=0.0,
        dial_cx=100.0,
        dial_cy=200.0,
        dial_rx=20.0,
        dial_ry=10.0,
    )
    crop: tuple[float, float, float, float] = training._compute_crop_box(
        sample, pad_ratio=0.1
    )
    assert crop == pytest.approx((78.0, 189.0, 122.0, 211.0))


def test_build_training_examples_filters_out_of_sweep(tmp_path: Path) -> None:
    """_build_training_examples should drop out-of-sweep labels."""
    spec: GaugeSpec = _make_spec()

    in_sweep: Sample = _make_sample(tmp_path / "in.jpg", tip_x=1.0, tip_y=0.0)
    out_of_sweep: Sample = _make_sample(tmp_path / "out.jpg", tip_x=0.0, tip_y=-1.0)

    examples, dropped = training._build_training_examples(
        [in_sweep, out_of_sweep],
        spec,
        strict_labels=True,
        crop_pad_ratio=0.1,
    )

    assert dropped == 1
    assert len(examples) == 1
    assert examples[0].image_path.endswith("in.jpg")
    assert examples[0].value == pytest.approx(-30.0)
    assert examples[0].needle_unit_xy == pytest.approx((1.0, 0.0))


def test_split_examples_raises_for_too_few_examples() -> None:
    """_split_examples should fail when fewer than 3 examples are provided."""
    config: training.TrainConfig = training.TrainConfig()
    with pytest.raises(ValueError, match="Need at least 3 examples"):
        training._split_examples([], config)


def test_split_examples_is_deterministic_and_disjoint() -> None:
    """_split_examples should be reproducible and non-overlapping."""
    examples: list[training.TrainingExample] = [
            training.TrainingExample(
                image_path=f"img_{i:03d}.jpg",
                value=float(i),
                crop_box_xyxy=(0.0, 0.0, 10.0, 10.0),
                needle_unit_xy=(1.0, 0.0),
            )
        for i in range(20)
    ]
    config: training.TrainConfig = training.TrainConfig(
        seed=123, val_fraction=0.2, test_fraction=0.2
    )

    split_a: training.DatasetSplit = training._split_examples(examples, config)
    split_b: training.DatasetSplit = training._split_examples(examples, config)

    assert split_a == split_b

    train_paths = {e.image_path for e in split_a.train_examples}
    val_paths = {e.image_path for e in split_a.val_examples}
    test_paths = {e.image_path for e in split_a.test_examples}

    assert len(train_paths | val_paths | test_paths) == 20
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_split_examples_honors_pinned_val_and_test_manifests(
    tmp_path: Path,
) -> None:
    """Pinned val/test manifests should be removed from the training pool."""
    examples: list[training.TrainingExample] = []
    for i in range(6):
        image_path = str(tmp_path / f"img_{i:03d}.jpg")
        examples.append(
            training.TrainingExample(
                image_path=image_path,
                value=float(i),
                crop_box_xyxy=(0.0, 0.0, 10.0, 10.0),
                needle_unit_xy=(1.0, 0.0),
            )
        )

    val_manifest = tmp_path / "val.csv"
    test_manifest = tmp_path / "test.csv"
    _write_manifest_csv(
        val_manifest,
        [
            (examples[1].image_path, 1.0),
            (examples[2].image_path, 2.0),
        ],
    )
    _write_manifest_csv(
        test_manifest,
        [
            (examples[3].image_path, 3.0),
            (examples[4].image_path, 4.0),
        ],
    )

    config = training.TrainConfig(
        seed=123,
        val_fraction=0.2,
        test_fraction=0.2,
        val_manifest=str(val_manifest),
        test_manifest=str(test_manifest),
    )

    split: training.DatasetSplit = training._split_examples(examples, config)

    assert [e.image_path for e in split.val_examples] == [
        examples[1].image_path,
        examples[2].image_path,
    ]
    assert [e.image_path for e in split.test_examples] == [
        examples[3].image_path,
        examples[4].image_path,
    ]
    assert [e.image_path for e in split.train_examples] == [
        examples[0].image_path,
        examples[5].image_path,
    ]


def test_crop_image_with_xyxy_clips_and_returns_nonempty() -> None:
    """_crop_image_with_xyxy should clip OOB boxes and always return valid crop."""
    image: tf.Tensor = tf.zeros((10, 10, 3), dtype=tf.float32)

    full_crop: tf.Tensor = training._crop_image_with_xyxy(
        image,
        tf.constant([-5.0, -3.0, 20.0, 25.0], dtype=tf.float32),
    )
    assert full_crop.shape == (10, 10, 3)

    small_crop: tf.Tensor = training._crop_image_with_xyxy(
        image,
        tf.constant([4.2, 3.1, 6.7, 7.9], dtype=tf.float32),
    )
    assert small_crop.shape == (5, 3, 3)


def test_load_crop_and_preprocess_image_outputs_normalized_tensor(
    tmp_path: Path,
) -> None:
    """_load_crop_and_preprocess_image should return resized [0,1] float image."""
    image_path: Path = tmp_path / "img.jpg"
    _write_test_jpeg(image_path, h=12, w=10)

    image, target = training._load_crop_and_preprocess_image(
        tf.constant(str(image_path)),
        tf.constant(12.5, dtype=tf.float32),
        tf.constant([2.0, 2.0, 9.0, 10.0], dtype=tf.float32),
        image_height=20,
        image_width=18,
    )

    assert image.shape == (20, 18, 3)
    assert image.dtype == tf.float32
    assert target.dtype == tf.float32
    assert float(tf.reduce_min(image).numpy()) >= 0.0
    assert float(tf.reduce_max(image).numpy()) <= 1.0
    assert float(target.numpy()) == pytest.approx(12.5)


def test_build_tf_dataset_returns_batched_tensors(tmp_path: Path) -> None:
    """_build_tf_dataset should emit batched (image, value, weight) tensors."""
    examples: list[training.TrainingExample] = []
    for i in range(3):
        path: Path = tmp_path / f"img_{i}.jpg"
        _write_test_jpeg(path, h=8, w=8)
        examples.append(
            training.TrainingExample(
                image_path=str(path),
                value=float(i),
                crop_box_xyxy=(1.0, 1.0, 7.0, 7.0),
                needle_unit_xy=(1.0, 0.0),
            )
        )

    config: training.TrainConfig = training.TrainConfig(
        image_height=16,
        image_width=16,
        batch_size=2,
        augment_training=False,
    )

    dataset: tf.data.Dataset = training._build_tf_dataset(
        examples, config, training=False
    )
    batches = list(dataset.as_numpy_iterator())

    assert len(batches) == 2
    images, targets, weights = batches[0]
    assert images.shape == (2, 16, 16, 3)
    assert targets.shape == (2,)
    assert weights.shape == (2,)
    assert images.dtype == np.float32
    assert targets.dtype == np.float32
    assert weights.dtype == np.float32


def test_compute_edge_weights_emphasizes_extremes() -> None:
    """Edge-focused weights should favor samples near the gauge limits."""
    examples: list[training.TrainingExample] = [
        training.TrainingExample(
            "low.jpg",
            -30.0,
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0),
            value_norm=0.0,
        ),
        training.TrainingExample(
            "mid.jpg",
            10.0,
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0),
            value_norm=0.5,
        ),
        training.TrainingExample(
            "high.jpg",
            50.0,
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0),
            value_norm=1.0,
        ),
    ]

    weights = training._compute_edge_weights(examples, strength=1.0)

    assert weights[0] > weights[1]
    assert weights[2] > weights[1]
    assert weights[0] == pytest.approx(weights[2])


def test_build_regression_model_output_shape() -> None:
    """build_regression_model should output one scalar per input image."""
    model = training.build_regression_model(32, 32)
    batch: tf.Tensor = tf.zeros((2, 32, 32, 3), dtype=tf.float32)
    preds: tf.Tensor = model(batch)

    assert model.output_shape == (None, 1)
    assert preds.shape == (2, 1)


def test_compute_mean_baseline_mae() -> None:
    """_compute_mean_baseline_mae should match manual mean-predictor MAE."""
    train_examples: list[training.TrainingExample] = [
        training.TrainingExample("a.jpg", 0.0, (0.0, 0.0, 1.0, 1.0), (1.0, 0.0)),
        training.TrainingExample("b.jpg", 2.0, (0.0, 0.0, 1.0, 1.0), (1.0, 0.0)),
    ]
    test_examples: list[training.TrainingExample] = [
        training.TrainingExample("c.jpg", 1.0, (0.0, 0.0, 1.0, 1.0), (1.0, 0.0)),
        training.TrainingExample("d.jpg", 3.0, (0.0, 0.0, 1.0, 1.0), (1.0, 0.0)),
    ]

    mae: float = training._compute_mean_baseline_mae(train_examples, test_examples)
    assert mae == pytest.approx(1.0)


def test_train_raises_for_unknown_gauge_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """train should fail fast when gauge_id is not in loaded specs."""
    known_spec: GaugeSpec = _make_spec("known_gauge")
    monkeypatch.setattr(
        training, "load_gauge_specs", lambda: {"known_gauge": known_spec}
    )

    with pytest.raises(ValueError, match="Unknown gauge_id"):
        training.train(training.TrainConfig(gauge_id="missing_gauge"))


def test_train_raises_when_no_samples_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """train should fail when dataset loading returns an empty list."""
    spec: GaugeSpec = _make_spec()
    monkeypatch.setattr(training, "load_gauge_specs", lambda: {spec.gauge_id: spec})
    monkeypatch.setattr(training, "load_dataset", lambda labelled_dir, raw_dir: [])

    with pytest.raises(ValueError, match="No samples found"):
        training.train(training.TrainConfig())


def test_train_happy_path_with_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """train should return TrainingResult and include baseline metric."""
    spec: GaugeSpec = _make_spec()

    label_summary: LabelSummary = LabelSummary(
        total_samples=10,
        in_sweep=9,
        out_of_sweep=1,
        min_fraction=0.0,
        max_fraction=1.0,
    )

    examples: list[training.TrainingExample] = [
        training.TrainingExample("a.jpg", 1.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("b.jpg", 2.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("c.jpg", 3.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("d.jpg", 4.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
    ]
    split: training.DatasetSplit = training.DatasetSplit(
        train_examples=examples[:2],
        val_examples=examples[2:3],
        test_examples=examples[3:],
    )

    class _FakeHistory:
        """Small stand-in for keras.callbacks.History."""

        def __init__(self) -> None:
            self.history: dict[str, list[float]] = {"loss": [1.0], "val_loss": [1.2]}

    class _FakeModel:
        """Small stand-in for compiled model in orchestration test."""

        def compile(self, **kwargs: Any) -> None:
            _ = kwargs

        def fit(
            self,
            train_ds: Any,
            validation_data: Any,
            epochs: int,
            callbacks: list[Any],
            verbose: int,
        ) -> _FakeHistory:
            assert train_ds is not None
            assert validation_data is not None
            assert epochs == 1
            assert len(callbacks) == 2
            assert verbose == 2
            return _FakeHistory()

        def evaluate(
            self, test_ds: Any, return_dict: bool, verbose: int
        ) -> dict[str, float]:
            assert test_ds is not None
            assert return_dict is True
            assert verbose == 0
            return {"loss": 1.0, "mae": 0.5, "rmse": 0.75}

    fake_model = _FakeModel()

    monkeypatch.setattr(training, "load_gauge_specs", lambda: {spec.gauge_id: spec})
    monkeypatch.setattr(
        training, "load_dataset", lambda labelled_dir, raw_dir: [object()] * 10
    )
    monkeypatch.setattr(
        training, "summarize_label_sweep", lambda samples, spec: label_summary
    )
    monkeypatch.setattr(
        training,
        "_build_training_examples",
        lambda samples, spec, **kwargs: (examples, 1),
    )
    monkeypatch.setattr(training, "_split_examples", lambda examples, config: split)
    monkeypatch.setattr(
        training,
        "_build_tf_dataset",
        lambda examples, config, training, target_kind="value", **kwargs: {
            "n": len(examples),
            "train": training,
            "target_kind": target_kind,
        },
    )
    monkeypatch.setattr(
        training,
        "build_regression_model",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        training,
        "build_mobilenetv2_regression_model",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        training,
        "_compute_mean_baseline_mae",
        lambda train_examples, test_examples: 2.5,
    )

    result: training.TrainingResult = training.train(
        training.TrainConfig(epochs=1, model_family="compact")
    )

    assert result.model is fake_model
    assert result.label_summary == label_summary
    assert result.dropped_out_of_sweep == 1
    assert result.baseline_test_mae == pytest.approx(2.5)
    assert result.test_metrics == {
        "loss": 1.0,
        "mae": 0.5,
        "rmse": 0.75,
        "baseline_mae_mean_predictor": 2.5,
    }


def test_train_happy_path_with_compact_interval_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train should route through the compact interval model family."""
    spec: GaugeSpec = _make_spec()

    label_summary: LabelSummary = LabelSummary(
        total_samples=10,
        in_sweep=9,
        out_of_sweep=1,
        min_fraction=0.0,
        max_fraction=1.0,
    )

    examples: list[training.TrainingExample] = [
        training.TrainingExample("a.jpg", 1.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("b.jpg", 2.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("c.jpg", 3.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("d.jpg", 4.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
    ]
    split: training.DatasetSplit = training.DatasetSplit(
        train_examples=examples[:2],
        val_examples=examples[2:3],
        test_examples=examples[3:],
    )

    class _FakeHistory:
        """Small stand-in for keras.callbacks.History."""

        def __init__(self) -> None:
            self.history: dict[str, list[float]] = {"loss": [1.0], "val_loss": [1.2]}

    class _FakeModel:
        """Small stand-in for a compiled interval model."""

        def compile(self, **kwargs: Any) -> None:
            _ = kwargs

        def fit(
            self,
            train_ds: Any,
            validation_data: Any,
            epochs: int,
            callbacks: list[Any],
            verbose: int,
        ) -> _FakeHistory:
            assert train_ds is not None
            assert validation_data is not None
            assert epochs == 1
            assert len(callbacks) == 2
            assert verbose == 2
            return _FakeHistory()

        def evaluate(
            self, test_ds: Any, return_dict: bool, verbose: int
        ) -> dict[str, float]:
            assert test_ds is not None
            assert return_dict is True
            assert verbose == 0
            return {"loss": 1.0, "mae": 0.5, "rmse": 0.75}

    fake_model = _FakeModel()

    monkeypatch.setattr(training, "load_gauge_specs", lambda: {spec.gauge_id: spec})
    monkeypatch.setattr(
        training, "load_dataset", lambda labelled_dir, raw_dir: [object()] * 10
    )
    monkeypatch.setattr(
        training, "summarize_label_sweep", lambda samples, spec: label_summary
    )
    monkeypatch.setattr(
        training,
        "_build_training_examples",
        lambda samples, spec, **kwargs: (examples, 1),
    )
    monkeypatch.setattr(training, "_split_examples", lambda examples, config: split)
    monkeypatch.setattr(
        training,
        "_build_tf_dataset",
        lambda examples, config, training, target_kind="value", **kwargs: {
            "n": len(examples),
            "train": training,
            "target_kind": target_kind,
        },
    )
    monkeypatch.setattr(
        training,
        "build_compact_interval_model",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        training,
        "_compute_mean_baseline_mae",
        lambda train_examples, test_examples: 2.5,
    )

    result: training.TrainingResult = training.train(
        training.TrainConfig(epochs=1, model_family="compact_interval")
    )

    assert result.model is fake_model
    assert result.label_summary == label_summary
    assert result.dropped_out_of_sweep == 1
    assert result.baseline_test_mae == pytest.approx(2.5)
    assert result.test_metrics == {
        "loss": 1.0,
        "mae": 0.5,
        "rmse": 0.75,
        "baseline_mae_mean_predictor": 2.5,
    }


def test_train_happy_path_with_compact_geometry_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train should route through the compact geometry model family."""
    spec: GaugeSpec = _make_spec()

    label_summary: LabelSummary = LabelSummary(
        total_samples=10,
        in_sweep=9,
        out_of_sweep=1,
        min_fraction=0.0,
        max_fraction=1.0,
    )

    examples: list[training.TrainingExample] = [
        training.TrainingExample("a.jpg", 1.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("b.jpg", 2.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("c.jpg", 3.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("d.jpg", 4.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
    ]
    split: training.DatasetSplit = training.DatasetSplit(
        train_examples=examples[:2],
        val_examples=examples[2:3],
        test_examples=examples[3:],
    )

    class _FakeHistory:
        """Small stand-in for keras.callbacks.History."""

        def __init__(self) -> None:
            self.history: dict[str, list[float]] = {"loss": [1.0], "val_loss": [1.2]}

    class _FakeModel:
        """Small stand-in for a compiled compact geometry model."""

        def compile(self, **kwargs: Any) -> None:
            _ = kwargs

        def fit(
            self,
            train_ds: Any,
            validation_data: Any,
            epochs: int,
            callbacks: list[Any],
            verbose: int,
        ) -> _FakeHistory:
            assert train_ds is not None
            assert validation_data is not None
            assert epochs == 1
            assert len(callbacks) == 2
            assert verbose == 2
            return _FakeHistory()

        def evaluate(
            self, test_ds: Any, return_dict: bool, verbose: int
        ) -> dict[str, float]:
            assert test_ds is not None
            assert return_dict is True
            assert verbose == 0
            return {"loss": 1.0, "mae": 0.5, "rmse": 0.75}

    fake_model = _FakeModel()

    def _build_tf_dataset_stub(
        examples: list[training.TrainingExample],
        config: training.TrainConfig,
        training: bool,
        target_kind: str = "value",
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert target_kind == "geometry"
        assert kwargs.get("value_range") == (-30.0, 50.0)
        return {"n": len(examples), "train": training, "target_kind": target_kind}

    monkeypatch.setattr(training, "load_gauge_specs", lambda: {spec.gauge_id: spec})
    monkeypatch.setattr(
        training, "load_dataset", lambda labelled_dir, raw_dir: [object()] * 10
    )
    monkeypatch.setattr(
        training, "summarize_label_sweep", lambda samples, spec: label_summary
    )
    monkeypatch.setattr(
        training,
        "_build_training_examples",
        lambda samples, spec, **kwargs: (examples, 1),
    )
    monkeypatch.setattr(training, "_split_examples", lambda examples, config: split)
    monkeypatch.setattr(training, "_build_tf_dataset", _build_tf_dataset_stub)
    monkeypatch.setattr(
        training,
        "build_compact_geometry_model",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        training,
        "_compute_mean_baseline_mae",
        lambda train_examples, test_examples: 2.5,
    )

    result: training.TrainingResult = training.train(
        training.TrainConfig(epochs=1, model_family="compact_geometry")
    )

    assert result.model is fake_model
    assert result.label_summary == label_summary
    assert result.dropped_out_of_sweep == 1
    assert result.baseline_test_mae == pytest.approx(2.5)
    assert result.test_metrics == {
        "loss": 1.0,
        "mae": 0.5,
        "rmse": 0.75,
        "baseline_mae_mean_predictor": 2.5,
    }


def test_train_can_warm_start_from_existing_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """train should load an initial model path when warm-starting fine-tune runs."""
    spec: GaugeSpec = _make_spec()

    label_summary: LabelSummary = LabelSummary(
        total_samples=10,
        in_sweep=9,
        out_of_sweep=1,
        min_fraction=0.0,
        max_fraction=1.0,
    )

    examples: list[training.TrainingExample] = [
        training.TrainingExample("a.jpg", 1.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("b.jpg", 2.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("c.jpg", 3.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
        training.TrainingExample("d.jpg", 4.0, (0.0, 0.0, 8.0, 8.0), (1.0, 0.0)),
    ]
    split: training.DatasetSplit = training.DatasetSplit(
        train_examples=examples[:2],
        val_examples=examples[2:3],
        test_examples=examples[3:],
    )

    class _FakeHistory:
        """Small stand-in for keras.callbacks.History."""

        def __init__(self) -> None:
            self.history: dict[str, list[float]] = {"loss": [1.0], "val_loss": [1.2]}

    class _FakeModel:
        """Small stand-in for a warm-started Keras model."""

        def __init__(self) -> None:
            self.trainable = False

        def compile(self, **kwargs: Any) -> None:
            _ = kwargs

        def fit(
            self,
            train_ds: Any,
            validation_data: Any,
            epochs: int,
            callbacks: list[Any],
            verbose: int,
        ) -> _FakeHistory:
            assert train_ds is not None
            assert validation_data is not None
            assert epochs == 1
            assert len(callbacks) == 2
            assert verbose == 2
            return _FakeHistory()

        def evaluate(
            self, test_ds: Any, return_dict: bool, verbose: int
        ) -> dict[str, float]:
            assert test_ds is not None
            assert return_dict is True
            assert verbose == 0
            return {"loss": 1.0, "mae": 0.5, "rmse": 0.75}

    fake_model = _FakeModel()
    init_model_path: Path = tmp_path / "existing.keras"
    init_model_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(training, "load_gauge_specs", lambda: {spec.gauge_id: spec})
    monkeypatch.setattr(
        training, "load_dataset", lambda labelled_dir, raw_dir: [object()] * 10
    )
    monkeypatch.setattr(
        training, "summarize_label_sweep", lambda samples, spec: label_summary
    )
    monkeypatch.setattr(
        training,
        "_build_training_examples",
        lambda samples, spec, **kwargs: (examples, 1),
    )
    monkeypatch.setattr(training, "_split_examples", lambda examples, config: split)
    monkeypatch.setattr(
        training,
        "_build_tf_dataset",
        lambda examples, config, training, target_kind="value", **kwargs: {
            "n": len(examples),
            "train": training,
            "target_kind": target_kind,
        },
    )
    monkeypatch.setattr(
        training.keras.models,
        "load_model",
        lambda path, compile=False, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        training,
        "build_regression_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not build")),
    )
    monkeypatch.setattr(
        training,
        "build_mobilenetv2_regression_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not build")),
    )
    monkeypatch.setattr(
        training,
        "_compute_mean_baseline_mae",
        lambda train_examples, test_examples: 2.5,
    )

    result: training.TrainingResult = training.train(
        training.TrainConfig(
            epochs=1,
            model_family="mobilenet_v2",
            init_model_path=str(init_model_path),
        )
    )

    assert result.model is fake_model
    assert result.label_summary == label_summary
    assert result.dropped_out_of_sweep == 1
    assert result.baseline_test_mae == pytest.approx(2.5)
