"""Quantization-aware fine-tune the strongest scalar MobileNetV2 model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

import keras
import numpy as np
import tensorflow as tf

# Add `ml/src` to sys.path so this script works from the ml/ directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.dataset import load_dataset  # noqa: E402
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs  # noqa: E402
from embedded_gauge_reading_tinyml.labels import summarize_label_sweep  # noqa: E402
from embedded_gauge_reading_tinyml.training import (  # noqa: E402
    LABELLED_DIR,
    RAW_DIR,
    DatasetSplit,
    TrainConfig,
    TrainingExample,
    _build_tf_dataset,
    _build_training_examples,
    _compile_regression_model,
    _configure_training_runtime,
    _log_dataset_state,
    _log_model_choice,
    _log_runtime_state,
    _load_hard_case_examples,
    _make_training_callbacks,
    _split_examples,
    _validate_split_config,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the QAT experiment."""
    parser = argparse.ArgumentParser(
        description="Quantization-aware fine-tune the best scalar MobileNetV2 model."
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "training"
        / "scalar_full_finetune_from_best_board30_piecewise_calibrated"
        / "model.keras",
        help="Path to the saved scalar MobileNetV2 model to quantize and fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--hard-case-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "hard_cases_plus_board30.csv",
        help="CSV manifest of extra hard board captures.",
    )
    parser.add_argument(
        "--hard-case-repeat",
        type=int,
        default=8,
        help="Repeat count for each hard-case row.",
    )
    parser.add_argument(
        "--edge-focus-strength",
        type=float,
        default=1.5,
        help="Additional weight placed on extreme gauge values (0 disables).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the MobileNetV2 backbone and fine-tune only the head.",
    )
    parser.add_argument(
        "--no-augment-training",
        action="store_true",
        help="Disable training-time image augmentation for the QAT run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="gpu",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="scalar_qat_from_best_board30",
    )
    return parser.parse_args()


def _freeze_mobilenet_backbone(model: keras.Model) -> None:
    """Freeze the nested MobileNetV2 backbone while leaving the head trainable."""
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower():
            layer.trainable = False
            print(f"[QAT] Frozen backbone layer '{layer.name}'.", flush=True)


def _load_base_model(model_path: Path) -> keras.Model:
    """Load the source scalar model that will be quantized."""
    print(f"[QAT] Loading base model from {model_path}...", flush=True)
    custom_objects: dict[str, Any] = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
    }
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )
    print(f"[QAT] Loaded model '{model.name}'.", flush=True)
    return model


def _load_hard_case_training_examples(
    manifest_path: Path,
    *,
    image_height: int,
    image_width: int,
    repeat: int,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> list[TrainingExample]:
    """Load and upweight extra hard board captures for QAT fine-tuning."""
    hard_cases: list[TrainingExample] = _load_hard_case_examples(
        manifest_path,
        image_height=image_height,
        image_width=image_width,
        value_range=value_range,
    )
    repeated_cases: list[TrainingExample] = hard_cases * max(repeat, 0)
    print(
        "[QAT] Hard-case examples: "
        f"base={len(hard_cases)} repeat={repeat} added={len(repeated_cases)}",
        flush=True,
    )
    return repeated_cases


def _start_training_heartbeat() -> tuple[threading.Event, threading.Thread]:
    """Start a background heartbeat so long GPU epochs stay visible."""
    stop_event = threading.Event()

    def _run() -> None:
        """Print a periodic pulse until training completes."""
        start = time.monotonic()
        tick = 0
        while not stop_event.wait(30.0):
            tick += 1
            elapsed_s = time.monotonic() - start
            print(
                "[QAT] Training heartbeat: "
                f"elapsed={elapsed_s:.0f}s tick={tick} still running on GPU...",
                flush=True,
            )

    thread = threading.Thread(
        target=_run,
        name="qat-training-heartbeat",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def main() -> None:
    """Quantize the base MobileNetV2 regressor and fine-tune it on labeled data."""
    args = _parse_args()
    run_dir: Path = args.artifacts_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        augment_training=not args.no_augment_training,
        hard_case_manifest=str(args.hard_case_manifest),
        hard_case_repeat=args.hard_case_repeat,
        edge_focus_strength=args.edge_focus_strength,
        model_family="mobilenet_v2",
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=not args.freeze_backbone,
        mobilenet_warmup_epochs=0,
        init_model_path=str(args.base_model),
    )

    print(
        "[QAT] Fine-tune job: "
        f"base_model={args.base_model} "
        f"epochs={config.epochs} "
        f"learning_rate={config.learning_rate} "
        f"device={config.device}",
        flush=True,
    )

    print("[QAT] Validating training config...", flush=True)
    _validate_split_config(config)
    print("[QAT] Config validated.", flush=True)
    print("[QAT] Configuring TensorFlow runtime...", flush=True)
    _configure_training_runtime(config)
    print("[QAT] TensorFlow runtime configured.", flush=True)
    print("[QAT] Logging runtime state...", flush=True)
    _log_runtime_state(config)
    print("[QAT] Runtime state logged.", flush=True)

    keras.utils.set_random_seed(config.seed)
    np.random.seed(config.seed)

    specs = load_gauge_specs()
    spec = specs[config.gauge_id]
    print(f"[QAT] Loaded gauge spec for '{config.gauge_id}'.", flush=True)

    print("[QAT] Loading labelled dataset...", flush=True)
    samples = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
    print(f"[QAT] Loaded {len(samples)} labelled samples.", flush=True)

    print("[QAT] Summarizing label sweep...", flush=True)
    label_summary = summarize_label_sweep(samples, spec)
    print("[QAT] Label sweep summary complete.", flush=True)

    print("[QAT] Building training examples...", flush=True)
    examples, dropped_out_of_sweep = _build_training_examples(
        samples,
        spec,
        strict_labels=config.strict_labels,
        crop_pad_ratio=config.crop_pad_ratio,
    )
    if len(examples) < 3:
        raise ValueError(
            "Not enough valid examples after filtering invalid sweep labels."
        )
    print(
        "[QAT] Training examples ready: "
        f"examples={len(examples)} dropped_out_of_sweep={dropped_out_of_sweep}",
        flush=True,
    )

    print("[QAT] Splitting dataset...", flush=True)
    split: DatasetSplit = _split_examples(examples, config)

    hard_case_manifest_path = Path(config.hard_case_manifest) if config.hard_case_manifest else None
    if hard_case_manifest_path is not None:
        if not hard_case_manifest_path.is_absolute():
            hard_case_manifest_path = PROJECT_ROOT / hard_case_manifest_path
        hard_case_examples = _load_hard_case_training_examples(
            hard_case_manifest_path,
            image_height=config.image_height,
            image_width=config.image_width,
            repeat=config.hard_case_repeat,
            value_range=(spec.min_value, spec.max_value),
        )
        if hard_case_examples:
            split = DatasetSplit(
                train_examples=split.train_examples + hard_case_examples,
                val_examples=split.val_examples,
                test_examples=split.test_examples,
            )
            print(
                "[QAT] Hard-case fine-tuning examples loaded: "
                f"added={len(hard_case_examples)}",
                flush=True,
            )

    _log_dataset_state(
        config,
        label_summary=label_summary,
        split=split,
        dropped_out_of_sweep=dropped_out_of_sweep,
    )

    _log_model_choice(config)
    base_model = _load_base_model(args.base_model)
    print("[QAT] Quantizing base model to int8...", flush=True)
    base_model.quantize("int8")
    qat_model = base_model
    if args.freeze_backbone:
        _freeze_mobilenet_backbone(qat_model)
    qat_model.trainable = True
    print(
        "[QAT] Quantized model ready: "
        f"name={qat_model.name} params={int(qat_model.count_params()):,}",
        flush=True,
    )

    train_ds = _build_tf_dataset(
        split.train_examples,
        config,
        training=True,
        target_kind="value",
    )
    val_ds = _build_tf_dataset(
        split.val_examples,
        config,
        training=False,
        target_kind="value",
    )
    test_ds = _build_tf_dataset(
        split.test_examples,
        config,
        training=False,
        target_kind="value",
    )
    print("[QAT] TensorFlow datasets built.", flush=True)

    _compile_regression_model(qat_model, learning_rate=config.learning_rate)
    callbacks = _make_training_callbacks()
    print("[QAT] Starting quantization-aware fine-tune...", flush=True)
    stop_event, _heartbeat = _start_training_heartbeat()
    try:
        history = qat_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=2,
        )
    finally:
        stop_event.set()

    raw_metrics: dict[str, float] = qat_model.evaluate(
        test_ds,
        return_dict=True,
        verbose=0,
    )
    test_metrics: dict[str, float] = {k: float(v) for k, v in raw_metrics.items()}

    model_path: Path = run_dir / "model.keras"
    qat_model.save(model_path)
    history_path: Path = run_dir / "history.json"
    metrics_path: Path = run_dir / "metrics.json"
    history_path.write_text(
        json.dumps({key: [float(v) for v in values] for key, values in history.history.items()}, indent=2),
        encoding="utf-8",
    )
    metrics_payload: dict[str, Any] = {
        "config": asdict(config),
        "label_summary": asdict(label_summary),
        "test_metrics": test_metrics,
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"[QAT] Run directory: {run_dir}", flush=True)
    print(f"[QAT] Model saved: {model_path}", flush=True)
    print(f"[QAT] Label summary: {label_summary}", flush=True)
    print(f"[QAT] Test metrics: {test_metrics}", flush=True)


if __name__ == "__main__":
    main()
