#!/usr/bin/env python3
"""Train MobileNetV2 direction model with board-style augmentation, evaluated on hard cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import train, TrainConfig

def main():
    config = TrainConfig(
        model_family='mobilenet_v2_direction',
        epochs=40,
        learning_rate=1e-4,
        batch_size=8,
        seed=21,
        augment_training=True,
        board_style_augment_prob=0.5,
        hard_case_eval_manifest='data/hard_cases_plus_board30.csv',
        device='cpu',
        gpu_memory_growth=False,
        mobilenet_alpha=1.0,
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=True,
        mobilenet_head_units=128,
        mobilenet_head_dropout=0.2,
    )

    result = train(config)
    print(f"Test MAE (C): {result.test_metrics}")

    run_dir = Path('artifacts/training/mobilenetv2_direction_board_augment')
    run_dir.mkdir(parents=True, exist_ok=True)
    result.model.save(run_dir / "model.keras")
    hist_dict = {k: [float(v) for v in vals] for k, vals in result.history.history.items()}
    (run_dir / "history.json").write_text(json.dumps(hist_dict, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(result.test_metrics, indent=2), encoding="utf-8")
    print(f"Saved artifacts to {run_dir}")

if __name__ == '__main__':
    main()
