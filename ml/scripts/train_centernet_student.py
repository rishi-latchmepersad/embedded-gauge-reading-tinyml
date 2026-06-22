#!/usr/bin/env python3
"""Train a MobileNetV2 CenterNet student via knowledge distillation + QAT.

Phase 2 of the center detector pipeline:
  1. Loads the trained ResNet-50 parent model (teacher).
  2. Builds a MobileNetV2 student with <2.5 MB activations.
  3. Trains with KD loss (teacher heatmap + offset as soft targets).
  4. Applies QAT fine-tuning for INT8 deployment.
  5. Matches firmware preprocessing (224x224 colour, letterbox pad).
  6. Target: center MAE < 5px at source resolution.

Usage:
    cd ml
    poetry run python scripts/train_centernet_student.py \
        --teacher artifacts/training/centernet_parent_*/final_model.keras
    poetry run python scripts/train_centernet_student.py \
        --teacher path/to/teacher.keras --qat --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
_SRC_DIR: Path = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from embedded_gauge_reading_tinyml.centernet import (
    CenterNetConfig,
    centernet_kd_loss,
    build_centernet_mobilenetv2_student,
    build_centernet_tf_dataset,
    decode_centernet_batch,
    load_geometry_manifest,
)

ML_ROOT: Path = _PROJECT_ROOT
DATA_DIR: Path = ML_ROOT / "data"
ARTIFACTS_DIR: Path = ML_ROOT / "artifacts"
DEFAULT_MANIFEST: Path = DATA_DIR / "geometry_heatmap_v12_all_data_manifest.csv"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class StudentTrainConfig:
    """Configuration for CenterNet student training with KD + QAT."""

    # Teacher.
    teacher_model_path: Path | None = None

    # Data.
    manifest_path: Path = field(default_factory=lambda: DEFAULT_MANIFEST)
    # Matches firmware: 224×224 colour, letterbox-padded.
    input_height: int = 224
    input_width: int = 224
    heatmap_height: int = 56
    heatmap_width: int = 56

    # Student architecture.
    alpha: float = 0.35

    # Training phases.
    pretrain_epochs: int = 20
    kd_epochs: int = 40
    qat_epochs: int = 20

    batch_size: int = 16
    initial_lr: float = 1e-3
    kd_lr: float = 5e-4
    qat_lr: float = 1e-4
    min_lr: float = 1e-7

    # KD loss config.
    temperature: float = 4.0
    kd_heatmap_weight: float = 0.5
    kd_offset_weight: float = 0.1
    hard_heatmap_weight: float = 0.5
    hard_offset_weight: float = 1.0

    # Board preprocessing.
    board_style_prob: float = 0.4

    # QAT.
    apply_qat: bool = True

    # Augmentation.
    augment: bool = True
    sigma_pixels: float = 2.0

    # Scheduling.
    lr_patience: int = 5
    lr_factor: float = 0.5
    early_stop_patience: int = 15

    # Logging.
    output_dir: Path = field(
        default_factory=lambda: ARTIFACTS_DIR
        / "training"
        / f"centernet_student_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )

    gpu_memory_growth: bool = True
    mixed_precision: bool = False


# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------


def _setup_gpu(config: StudentTrainConfig) -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        if config.gpu_memory_growth:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU: {gpu.name}")
    if config.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def _build_datasets(
    config: StudentTrainConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    all_rows = load_geometry_manifest(config.manifest_path, splits=("train", "val"))
    train_rows = [r for r in all_rows if r.split == "train"]
    val_rows = [r for r in all_rows if r.split == "val"]
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    train_ds = build_centernet_tf_dataset(
        train_rows,
        input_height=config.input_height,
        input_width=config.input_width,
        heatmap_height=config.heatmap_height,
        heatmap_width=config.heatmap_width,
        sigma_pixels=config.sigma_pixels,
        batch_size=config.batch_size,
        shuffle=True,
        augment=config.augment,
    )
    val_ds = build_centernet_tf_dataset(
        val_rows,
        input_height=config.input_height,
        input_width=config.input_width,
        heatmap_height=config.heatmap_height,
        heatmap_width=config.heatmap_width,
        sigma_pixels=config.sigma_pixels,
        batch_size=config.batch_size,
        shuffle=False,
        augment=False,
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _build_student_model(config: StudentTrainConfig) -> keras.Model:
    centernet_cfg = CenterNetConfig(
        input_height=config.input_height,
        input_width=config.input_width,
        heatmap_height=config.heatmap_height,
        heatmap_width=config.heatmap_width,
        backbone="mobilenetv2",
        alpha=config.alpha,
        backbone_trainable=True,
        use_skip_connections=True,
        decoder_filters=64,
        head_filters=16,
    )
    return build_centernet_mobilenetv2_student(centernet_cfg)


def _load_teacher_model(teacher_path: Path) -> keras.Model:
    teacher = keras.models.load_model(str(teacher_path), compile=False)
    teacher.trainable = False
    print(f"Teacher loaded: {teacher.name} ({teacher.count_params():,} params)")
    return teacher


# ---------------------------------------------------------------------------
# KD Trainer (custom loop for teacher outputs)
# ---------------------------------------------------------------------------


class KDTrainer:
    """Custom training loop for KD since teacher outputs are extra inputs."""

    def __init__(
        self,
        student: keras.Model,
        teacher: keras.Model,
        config: StudentTrainConfig,
    ):
        self.student = student
        self.teacher = teacher
        self.config = config
        self.optimizer: keras.optimizers.Optimizer | None = None
        self.train_loss_metric = keras.metrics.Mean(name="train_kd_loss")
        self.val_loss_metric = keras.metrics.Mean(name="val_kd_loss")

    def compile(self, optimizer: keras.optimizers.Optimizer) -> None:
        self.optimizer = optimizer
        # Expose optimizer on model for Keras callbacks (ReduceLROnPlateau etc.).
        self.student.optimizer = optimizer

    @tf.function
    def train_step(
        self,
        images: tf.Tensor,
        target: tf.Tensor,
        use_teacher: bool = True,
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            student_out = self.student(images, training=True)  # (B, H, W, 3)

            teacher_out = None
            if use_teacher:
                # Resize images to teacher input size (384×384).
                teacher_h = self.config.input_height * 384 // 224  # 384
                teacher_w = self.config.input_width * 384 // 224
                images_teacher = tf.image.resize(
                    images, (teacher_h, teacher_w), method="bilinear"
                )
                teacher_out_raw = self.teacher(images_teacher, training=False)
                # Resize teacher output to student output size.
                sh = tf.shape(student_out)
                teacher_out = tf.image.resize(
                    teacher_out_raw, (sh[1], sh[2]), method="bilinear"
                )

            loss = centernet_kd_loss(
                student_out=student_out,
                target=target,
                teacher_out=teacher_out,
                temperature=self.config.temperature,
                kd_heatmap_weight=self.config.kd_heatmap_weight if use_teacher else 0.0,
                kd_offset_weight=self.config.kd_offset_weight if use_teacher else 0.0,
                hard_heatmap_weight=self.config.hard_heatmap_weight,
                hard_offset_weight=self.config.hard_offset_weight,
            )

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.student.trainable_variables)
        )
        self.train_loss_metric.update_state(loss)
        return loss

    @tf.function
    def test_step(
        self,
        images: tf.Tensor,
        target: tf.Tensor,
        use_teacher: bool = True,
    ) -> tf.Tensor:
        student_out = self.student(images, training=False)

        teacher_out = None
        if use_teacher:
            teacher_h = self.config.input_height * 384 // 224
            teacher_w = self.config.input_width * 384 // 224
            images_teacher = tf.image.resize(
                images, (teacher_h, teacher_w), method="bilinear"
            )
            teacher_out_raw = self.teacher(images_teacher, training=False)
            sh = tf.shape(student_out)
            teacher_out = tf.image.resize(
                teacher_out_raw, (sh[1], sh[2]), method="bilinear"
            )

        loss = centernet_kd_loss(
            student_out=student_out,
            target=target,
            teacher_out=teacher_out,
            temperature=self.config.temperature,
            kd_heatmap_weight=self.config.kd_heatmap_weight if use_teacher else 0.0,
            kd_offset_weight=self.config.kd_offset_weight if use_teacher else 0.0,
            hard_heatmap_weight=self.config.hard_heatmap_weight,
            hard_offset_weight=self.config.hard_offset_weight,
        )
        self.val_loss_metric.update_state(loss)
        return loss

    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int,
        use_teacher: bool = True,
        callbacks: list[keras.callbacks.Callback] | None = None,
    ) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {"loss": [], "val_loss": []}

        if callbacks:
            for cb in callbacks:
                cb.set_model(self.student)
                cb.on_train_begin()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            self.train_loss_metric.reset_state()
            for images, target in train_ds:
                self.train_step(images, target, use_teacher=use_teacher)
            train_loss = self.train_loss_metric.result().numpy()

            self.val_loss_metric.reset_state()
            for images, target in val_ds:
                self.test_step(images, target, use_teacher=use_teacher)
            val_loss = self.val_loss_metric.result().numpy()

            history["loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            lr = float(self.optimizer.learning_rate.numpy()) if hasattr(self.optimizer, "learning_rate") else 0
            print(f"  loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, lr: {lr:.2e}")

            if callbacks:
                logs = {"loss": train_loss, "val_loss": val_loss}
                for cb in callbacks:
                    cb.on_epoch_end(epoch, logs)

        if callbacks:
            for cb in callbacks:
                cb.on_train_end()

        return history


# ---------------------------------------------------------------------------
# QAT
# ---------------------------------------------------------------------------


def _apply_qat(model: keras.Model) -> keras.Model:
    try:
        import tensorflow_model_optimization as tfmot
        qat_model = tfmot.quantization.keras.quantize_model(model)
        print("QAT model created.")
        return qat_model
    except ImportError:
        print("WARNING: tfmot not available. Skipping QAT.")
        return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_student(
    student: keras.Model,
    val_rows: list,
    config: StudentTrainConfig,
) -> dict[str, float]:
    val_ds = build_centernet_tf_dataset(
        val_rows,
        input_height=config.input_height,
        input_width=config.input_width,
        heatmap_height=config.heatmap_height,
        heatmap_width=config.heatmap_width,
        sigma_pixels=config.sigma_pixels,
        batch_size=config.batch_size,
        shuffle=False,
        augment=False,
    )

    pixel_errors: list[float] = []
    output_stride = config.input_height // config.heatmap_height

    for batch_idx, (images, target) in enumerate(val_ds):
        pred = student.predict(images, verbose=0)
        hm_pred = pred[..., 0:1]
        off_pred = pred[..., 1:3]
        hm_gt_np = target.numpy()[..., 0:1]
        off_gt_np = target.numpy()[..., 1:3]

        detections = decode_centernet_batch(
            hm_pred, off_pred, topk=1, min_score=0.1
        )

        for i, dets in enumerate(detections):
            if not dets:
                pixel_errors.append(float(config.heatmap_height))
                continue
            cx_hm, cy_hm, _ = dets[0]
            gt_hm_np_i = hm_gt_np[i].squeeze()
            gt_cy, gt_cx = np.unravel_index(np.argmax(gt_hm_np_i), gt_hm_np_i.shape)
            gt_cx += off_gt_np[i, int(gt_cy), int(gt_cx), 0]
            gt_cy += off_gt_np[i, int(gt_cy), int(gt_cx), 1]
            hm_error = np.sqrt((cx_hm - gt_cx) ** 2 + (cy_hm - gt_cy) ** 2)
            pixel_errors.append(float(hm_error))

        if (batch_idx + 1) * config.batch_size >= 200:
            break

    results: dict[str, float] = {
        "center_mae_hm_pixels": float(np.mean(pixel_errors)),
        "center_mae_source_pixels": float(np.mean(pixel_errors)) * output_stride,
        "center_mae_n_pixels": len(pixel_errors),
    }
    print(f"\nStudent validation:")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")
    return results


# ---------------------------------------------------------------------------
# Activations estimate
# ---------------------------------------------------------------------------


def _estimate_activation_memory(model: keras.Model) -> float:
    total = 0
    for layer in model.layers:
        shape = getattr(layer, "output_shape", None)
        if shape and isinstance(shape, tuple) and all(d is not None for d in shape[1:]):
            elements = int(np.prod(shape[1:]))
            total += elements
    return total * 4 / (1024 * 1024)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train_student(config: StudentTrainConfig) -> keras.Model:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_gpu(config)

    with open(config.output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    train_ds, val_ds = _build_datasets(config)

    student = _build_student_model(config)
    student.summary()

    act_mb = _estimate_activation_memory(student)
    print(f"\nEstimated activation memory: {act_mb:.1f} MB")
    if act_mb > 2.5:
        print("WARNING: Activations exceed 2.5 MB target!")

    if config.teacher_model_path is None:
        raise ValueError("--teacher path is required for KD training")
    teacher = _load_teacher_model(Path(config.teacher_model_path))

    def _make_callbacks(prefix: str) -> list[keras.callbacks.Callback]:
        return [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=config.lr_factor,
                patience=config.lr_patience, min_lr=config.min_lr, verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.early_stop_patience,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.CSVLogger(
                str(config.output_dir / f"{prefix}_log.csv")
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(config.output_dir / f"best_{prefix}.weights.h5"),
                monitor="val_loss", save_best_only=True,
                save_weights_only=True, verbose=1,
            ),
        ]

    # ---- Phase 1: Pretrain (hard loss only) ----
    print("\n" + "=" * 60)
    print("Phase 1: Pretrain (hard loss only)")
    print("=" * 60)
    pretrain_trainer = KDTrainer(student, teacher, config)
    pretrain_trainer.compile(
        keras.optimizers.Adam(learning_rate=config.initial_lr)
    )
    pretrain_trainer.fit(
        train_ds, val_ds, epochs=config.pretrain_epochs,
        use_teacher=False,  # No KD during pretrain.
        callbacks=_make_callbacks("pretrain"),
    )

    # ---- Phase 2: Knowledge Distillation ----
    print("\n" + "=" * 60)
    print("Phase 2: Knowledge Distillation")
    print("=" * 60)
    kd_trainer = KDTrainer(student, teacher, config)
    kd_trainer.compile(
        keras.optimizers.Adam(learning_rate=config.kd_lr)
    )
    kd_trainer.fit(
        train_ds, val_ds, epochs=config.kd_epochs,
        use_teacher=True,
        callbacks=_make_callbacks("kd"),
    )
    student.save(config.output_dir / "student_kd.keras")

    # ---- Phase 3: QAT fine-tuning ----
    if config.apply_qat:
        print("\n" + "=" * 60)
        print("Phase 3: QAT Fine-tuning")
        print("=" * 60)

        qat_student = _apply_qat(student)

        qat_train_ds = build_centernet_tf_dataset(
            [r for r in load_geometry_manifest(
                config.manifest_path, splits=("train",)
            ) if r.split == "train"],
            input_height=config.input_height,
            input_width=config.input_width,
            heatmap_height=config.heatmap_height,
            heatmap_width=config.heatmap_width,
            sigma_pixels=config.sigma_pixels,
            batch_size=max(4, config.batch_size // 4),
            shuffle=True,
            augment=True,
        )

        qat_trainer = KDTrainer(qat_student, teacher, config)
        qat_trainer.compile(
            keras.optimizers.Adam(
                learning_rate=config.qat_lr, clipnorm=1.0,
            )
        )
        qat_trainer.fit(
            qat_train_ds, val_ds, epochs=config.qat_epochs,
            use_teacher=True,
            callbacks=_make_callbacks("qat"),
        )
        qat_student.save(config.output_dir / "student_qat.keras")
        student = qat_student

    # ---- Evaluation ----
    val_rows = load_geometry_manifest(config.manifest_path, splits=("val",))
    results = _evaluate_student(student, val_rows, config)
    with open(config.output_dir / "val_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return student


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 CenterNet student via KD + QAT."
    )
    parser.add_argument(
        "--teacher", type=Path, required=True,
        help="Path to trained CenterNet teacher model (.keras).",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
    )
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--kd-epochs", type=int, default=40)
    parser.add_argument("--qat-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--no-qat", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()

    config = StudentTrainConfig(
        teacher_model_path=args.teacher,
        manifest_path=args.manifest,
        pretrain_epochs=args.pretrain_epochs,
        kd_epochs=args.kd_epochs,
        qat_epochs=args.qat_epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        apply_qat=not args.no_qat,
    )
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    train_student(config)


if __name__ == "__main__":
    _main()
