"""Unit tests for training utilities and training orchestration."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest
import tensorflow as tf

import embedded_gauge_reading_tinyml.training as training
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


def test_validate_split_config_accepts_valid_values() -> None:
    """_validate_split_config should not raise for valid fractions."""
    config: training.TrainConfig = training.TrainConfig(
        val_fraction=0.2, test_fraction=0.2
    )
    training._validate_split_config(config)


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
    """_build_tf_dataset should emit batched (image, value) tensors."""
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
    images, targets = batches[0]
    assert images.shape == (2, 16, 16, 3)
    assert targets.shape == (2,)
    assert images.dtype == np.float32
    assert targets.dtype == np.float32


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
        lambda samples, spec, strict_labels, crop_pad_ratio: (examples, 1),
    )
    monkeypatch.setattr(training, "_split_examples", lambda examples, config: split)
    monkeypatch.setattr(
        training,
        "_build_tf_dataset",
        lambda examples, config, training: {"n": len(examples), "train": training},
    )
    monkeypatch.setattr(
        training,
        "build_regression_model",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        training,
        "_compute_mean_baseline_mae",
        lambda train_examples, test_examples: 2.5,
    )

    result: training.TrainingResult = training.train(training.TrainConfig(epochs=1))

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
