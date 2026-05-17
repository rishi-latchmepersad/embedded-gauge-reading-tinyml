#!/usr/bin/env python3
"""Smoke test: 1-epoch direction model run."""

from __future__ import annotations

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
        epochs=1,
        learning_rate=1e-4,
        batch_size=4,
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
    print(f"Test metrics: {result.test_metrics}")
    print("Smoke test PASSED")

if __name__ == '__main__':
    main()
