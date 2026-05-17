#!/usr/bin/env python3
"""Train MobileNetV2 direction model with board-style augmentation and hard-case injection."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import keras
import tensorflow as tf
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.labels import summarize_label_sweep
from embedded_gauge_reading_tinyml.models import build_mobilenetv2_direction_model
from embedded_gauge_reading_tinyml.training import (
    DatasetSplit,
    TrainConfig,
    _build_tf_dataset,
    _build_training_examples,
    _compile_direction_model,
    _configure_training_runtime,
    _load_hard_case_examples,
    _log_dataset_state,
    _make_training_callbacks,
    _split_examples,
)

def _parse_args():
    p = argparse.ArgumentParser(description='Train MobileNetV2 direction model.')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=21)
    p.add_argument('--hard-case-manifest', type=Path, default=PROJECT_ROOT / 'data' / 'hard_cases_plus_board30.csv')
    p.add_argument('--hard-case-repeat', type=int, default=8)
    p.add_argument('--board-style-augment-prob', type=float, default=0.5)
    p.add_argument('--artifacts-dir', type=Path, default=PROJECT_ROOT / 'artifacts' / 'training')
    p.add_argument('--run-name', type=str, default='mobilenetv2_direction_board_augment')
    return p.parse_args()

def main():
    args = _parse_args()
    run_dir = args.artifacts_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'[DIRECTION] Run directory: {run_dir}', flush=True)

    _configure_training_runtime(TrainConfig(device='gpu', gpu_memory_growth=True))

    config = TrainConfig(
        model_family='mobilenet_v2_direction',
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        augment_training=True,
        board_style_augment_prob=args.board_style_augment_prob,
        device='gpu',
        gpu_memory_growth=True,
        mobilenet_alpha=1.0,
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=True,
        mobilenet_head_units=128,
        mobilenet_head_dropout=0.2,
    )

    spec = load_gauge_specs(config.gauge_id)
    samples = load_dataset()
    label_summary = summarize_label_sweep(samples, spec)

    examples, dropped = _build_training_examples(
        samples, spec,
        image_height=config.image_height,
        image_width=config.image_width,
        strict_labels=config.strict_labels,
        crop_pad_ratio=config.crop_pad_ratio,
    )
    print(f'[DIRECTION] Built {len(examples)} training examples', flush=True)

    split = _split_examples(examples, config)

    hard_path = Path(args.hard_case_manifest)
    if hard_path.exists():
        hard_examples = _load_hard_case_examples(
            hard_path,
            image_height=config.image_height,
            image_width=config.image_width,
            repeat=args.hard_case_repeat,
            value_range=(spec.min_value, spec.max_value),
        )
        if hard_examples:
            split = DatasetSplit(
                train_examples=split.train_examples + hard_examples,
                val_examples=split.val_examples,
                test_examples=split.test_examples,
            )
            print(f'[DIRECTION] Added {len(hard_examples)} hard-case repeats', flush=True)

    _log_dataset_state(config, label_summary=label_summary, split=split, dropped_out_of_sweep=dropped)

    model = build_mobilenetv2_direction_model(
        config.image_height,
        config.image_width,
        pretrained=config.mobilenet_pretrained,
        backbone_trainable=config.mobilenet_backbone_trainable,
        alpha=config.mobilenet_alpha,
        head_units=config.mobilenet_head_units,
        head_dropout=config.mobilenet_head_dropout,
    )
    model.trainable = True
    total_params = int(model.count_params())
    print(f'[DIRECTION] Model params: {total_params}', flush=True)

    train_ds = _build_tf_dataset(split.train_examples, config, training=True, target_kind='needle_unit_xy')
    val_ds = _build_tf_dataset(split.val_examples, config, training=False, target_kind='needle_unit_xy')
    test_ds = _build_tf_dataset(split.test_examples, config, training=False, target_kind='needle_unit_xy')

    _compile_direction_model(model, learning_rate=config.learning_rate, spec=spec)
    callbacks = _make_training_callbacks()

    print('[DIRECTION] Starting training...', flush=True)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
    print(f'[DIRECTION] Test metrics: {test_metrics}', flush=True)

    model_path = run_dir / 'model.keras'
    model.save(model_path)
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (run_dir / 'history.json').write_text(json.dumps(hist_dict, indent=2), encoding='utf-8')
    (run_dir / 'metrics.json').write_text(
        json.dumps({'test_metrics': test_metrics}, indent=2), encoding='utf-8'
    )
    print(f'[DIRECTION] Saved model to {model_path}', flush=True)

if __name__ == '__main__':
    main()
