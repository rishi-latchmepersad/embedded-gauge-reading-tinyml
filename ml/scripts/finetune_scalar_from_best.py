"""Fine-tune the strongest scalar MobileNetV2 model on hard board captures."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import threading
import time
from pathlib import Path
import json
import sys
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
    TrainingExample,
    TrainConfig,
    _build_tf_dataset,
    _build_training_examples,
    _compile_regression_model,
    _compute_mean_baseline_mae,
    _configure_training_runtime,
    _log_dataset_state,
    _log_model_choice,
    _log_runtime_state,
    _make_training_callbacks,
    _load_hard_case_examples,
    _split_examples,
    _validate_split_config,
)
from embedded_gauge_reading_tinyml.presets import (  # noqa: E402
    DEFAULT_EDGE_FOCUS_STRENGTH,
    DEFAULT_KEYPOINT_HEATMAP_SIZE,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the head-only fine-tune job."""
    parser = argparse.ArgumentParser(
        description="Fine-tune the best scalar MobileNetV2 model on hard board captures."
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training" / "wsl_mnv2_finetune_seed21" / "model.keras",
        help="Path to the saved scalar MobileNetV2 model to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--hard-case-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "hard_cases.csv",
        help="CSV manifest of extra hard board captures.",
    )
    parser.add_argument(
        "--hard-case-repeat",
        type=int,
        default=4,
        help="Repeat count for each hard-case row.",
    )
    parser.add_argument(
        "--edge-focus-strength",
        type=float,
        default=DEFAULT_EDGE_FOCUS_STRENGTH,
        help="Additional weight placed on extreme gauge values (0 disables).",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        action="store_true",
        help="Leave the MobileNetV2 backbone trainable instead of freezing it.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="gpu",
    )
    parser.add_argument(
        "--no-gpu-memory-growth",
        action="store_true",
        help="Disable TensorFlow GPU memory growth if GPU startup is flaky.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="scalar_head_finetune_from_best",
    )
    return parser.parse_args()


def _load_base_model(model_path: Path) -> keras.Model:
    """Load the previously best scalar MobileNetV2 model artifact."""
    print(f"[FT] Loading base model from {model_path}...", flush=True)
    custom_objects: dict[str, Any] = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"[FT] Loaded model '{model.name}'.", flush=True)
    return model


def _freeze_mobilenet_backbone(model: keras.Model) -> None:
    """Freeze the nested MobileNetV2 backbone while leaving the head trainable."""
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower():
            layer.trainable = False
            print(f"[FT] Frozen backbone layer '{layer.name}'.", flush=True)


def _load_hard_case_training_examples(
    manifest_path: Path,
    *,
    image_height: int,
    image_width: int,
    repeat: int,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> list[TrainingExample]:
    """Load and upweight extra hard board captures for head-only fine-tuning."""
    hard_cases: list[TrainingExample] = _load_hard_case_examples(
        manifest_path,
        image_height=image_height,
        image_width=image_width,
        value_range=value_range,
    )
    repeated_cases: list[TrainingExample] = hard_cases * max(repeat, 0)
    print(
        "[FT] Hard-case examples: "
        f"base={len(hard_cases)} repeat={repeat} added={len(repeated_cases)}",
        flush=True,
    )
    return repeated_cases


def _predict_hard_cases(
    model: keras.Model,
    examples: list[TrainingExample],
    *,
    image_height: int,
    image_width: int,
) -> list[dict[str, float | str]]:
    """Run in-memory inference on the labeled hard-case frames."""
    predictions: list[dict[str, float | str]] = []
    for example in examples:
        image_path = Path(example.image_path)
        image = tf.keras.utils.load_img(
            image_path,
            target_size=(image_height, image_width),
        )
        image_array = tf.keras.utils.img_to_array(image).astype(np.float32)
        batch = tf.convert_to_tensor(image_array[None, ...] / 255.0, dtype=tf.float32)
        pred_value = float(model.predict(batch, verbose=0)[0][0])
        abs_error = abs(pred_value - example.value)
        predictions.append(
            {
                "image_path": str(image_path),
                "label_value": float(example.value),
                "prediction": pred_value,
                "abs_error": abs_error,
            }
        )
        print(
            "[FT] Hard-case eval: "
            f"{image_path.name} true={example.value:.4f} "
            f"pred={pred_value:.4f} abs_err={abs_error:.4f}",
            flush=True,
        )

    if predictions:
        mean_abs_error = float(np.mean([item["abs_error"] for item in predictions]))
        print(f"[FT] Hard-case mean abs error: {mean_abs_error:.4f}", flush=True)
    return predictions


def _start_training_heartbeat() -> tuple[threading.Event, threading.Thread]:
    """Start a background heartbeat so long GPU epochs are visibly alive."""
    stop_event = threading.Event()

    def _run() -> None:
        """Print a periodic pulse until training completes."""
        start = time.monotonic()
        tick = 0
        while not stop_event.wait(30.0):
            tick += 1
            elapsed_s = time.monotonic() - start
            print(
                "[FT] Training heartbeat: "
                f"elapsed={elapsed_s:.0f}s tick={tick} still running on GPU...",
                flush=True,
            )

    thread = threading.Thread(target=_run, name="training-heartbeat", daemon=True)
    thread.start()
    return stop_event, thread


def main() -> None:
    """Fine-tune the best scalar MobileNetV2 model and save the adapted artifact."""
    args = _parse_args()
    run_dir: Path = args.artifacts_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        gpu_memory_growth=not args.no_gpu_memory_growth,
        hard_case_manifest=str(args.hard_case_manifest),
        hard_case_repeat=args.hard_case_repeat,
        edge_focus_strength=args.edge_focus_strength,
        model_family="mobilenet_v2",
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=True,
        mobilenet_warmup_epochs=0,
    )

    print(
        "[FT] Fine-tune job: "
        f"base_model={args.base_model} "
        f"epochs={config.epochs} "
        f"learning_rate={config.learning_rate} "
        f"device={config.device}",
        flush=True,
    )

    print("[FT] Validating training config...", flush=True)
    _validate_split_config(config)
    print("[FT] Config validated.", flush=True)
    print("[FT] Configuring TensorFlow runtime...", flush=True)
    _configure_training_runtime(config)
    print("[FT] TensorFlow runtime configured.", flush=True)
    print("[FT] Logging runtime state...", flush=True)
    _log_runtime_state(config)
    print("[FT] Runtime state logged.", flush=True)

    keras.utils.set_random_seed(config.seed)
    np.random.seed(config.seed)

    specs = load_gauge_specs()
    spec = specs[config.gauge_id]
    print(f"[FT] Loaded gauge spec for '{config.gauge_id}'.", flush=True)

    print("[FT] Loading labelled dataset...", flush=True)
    samples = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
    print(f"[FT] Loaded {len(samples)} labelled samples.", flush=True)

    print("[FT] Summarizing label sweep...", flush=True)
    label_summary = summarize_label_sweep(samples, spec)
    print("[FT] Label sweep summary complete.", flush=True)

    print("[FT] Building training examples...", flush=True)
    examples, dropped_out_of_sweep = _build_training_examples(
        samples,
        spec,
        image_height=config.image_height,
        image_width=config.image_width,
        keypoint_heatmap_size=DEFAULT_KEYPOINT_HEATMAP_SIZE,
        strict_labels=config.strict_labels,
        crop_pad_ratio=config.crop_pad_ratio,
    )
    print(
        "[FT] Training examples ready: "
        f"examples={len(examples)} dropped_out_of_sweep={dropped_out_of_sweep}",
        flush=True,
    )

    print("[FT] Splitting dataset...", flush=True)
    split: DatasetSplit = _split_examples(examples, config)
    if config.hard_case_manifest and config.hard_case_repeat > 0:
        hard_case_examples = _load_hard_case_training_examples(
            Path(config.hard_case_manifest),
            image_height=config.image_height,
            image_width=config.image_width,
            repeat=config.hard_case_repeat,
            value_range=(spec.min_value, spec.max_value),
        )
        split = DatasetSplit(
            train_examples=split.train_examples + hard_case_examples,
            val_examples=split.val_examples,
            test_examples=split.test_examples,
        )

    _log_dataset_state(
        config,
        label_summary=label_summary,
        split=split,
        dropped_out_of_sweep=dropped_out_of_sweep,
    )

    _log_model_choice(config)
    model = _load_base_model(args.base_model)
    if not args.no_freeze_backbone:
        _freeze_mobilenet_backbone(model)

    _compile_regression_model(model, learning_rate=config.learning_rate)
    train_ds = _build_tf_dataset(split.train_examples, config, training=True)
    val_ds = _build_tf_dataset(split.val_examples, config, training=False)
    test_ds = _build_tf_dataset(split.test_examples, config, training=False)
    print("[FT] TensorFlow datasets built.", flush=True)

    callbacks = _make_training_callbacks()
    print("[FT] Starting fine-tune fit...", flush=True)
    heartbeat_stop, heartbeat_thread = _start_training_heartbeat()
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=1,
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2.0)

    hard_case_predictions: list[dict[str, float | str]] = []
    if config.hard_case_manifest:
        print("[FT] Running in-memory hard-case evaluation...", flush=True)
        hard_case_examples = _load_hard_case_examples(
            Path(config.hard_case_manifest),
            image_height=config.image_height,
            image_width=config.image_width,
            value_range=(spec.min_value, spec.max_value),
        )
        hard_case_predictions = _predict_hard_cases(
            model,
            hard_case_examples,
            image_height=config.image_height,
            image_width=config.image_width,
        )

    raw_metrics: dict[str, float] = model.evaluate(test_ds, return_dict=True, verbose=0)
    test_metrics: dict[str, float] = {k: float(v) for k, v in raw_metrics.items()}
    baseline_test_mae: float = _compute_mean_baseline_mae(
        split.train_examples,
        split.test_examples,
    )
    test_metrics["baseline_mae_mean_predictor"] = baseline_test_mae

    model_path: Path = run_dir / "model.keras"
    model.save(model_path)
    history_path: Path = run_dir / "history.json"
    metrics_path: Path = run_dir / "metrics.json"

    history_payload: dict[str, list[float]] = {
        key: [float(v) for v in values] for key, values in history.history.items()
    }
    metrics_payload: dict[str, Any] = {
        "config": asdict(config),
        "label_summary": asdict(label_summary),
        "test_metrics": test_metrics,
        "hard_case_predictions": hard_case_predictions,
        "model_path": str(model_path),
    }

    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Model saved: {model_path}", flush=True)
    print(f"Label summary: {label_summary}", flush=True)
    print(f"Test metrics: {test_metrics}", flush=True)


if __name__ == "__main__":
    main()
