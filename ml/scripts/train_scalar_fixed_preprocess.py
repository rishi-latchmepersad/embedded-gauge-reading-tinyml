#!/usr/bin/env python3
"""Train MobileNetV2 scalar model with FIXED preprocessing.

The key fix: _build_mobilenetv2_backbone now uses Rescaling(2.0, -1.0) 
instead of Rescaling(1.0/127.5, -1.0), because the training pipeline 
normalizes images to [0,1] not [0,255].

This makes pretrained ImageNet weights actually useful.
"""

from __future__ import annotations

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
        model_family='mobilenet_v2',
        epochs=20,
        learning_rate=3e-4,  # higher LR since pretrained weights are now useful
        batch_size=16,
        seed=21,
        augment_training=True,
        board_style_augment_prob=0.5,
        device='cpu',
        gpu_memory_growth=False,
        mobilenet_alpha=1.0,
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=False,  # freeze backbone first
        mobilenet_head_units=128,
        mobilenet_head_dropout=0.2,
        edge_focus_strength=2.5,
        hard_case_manifest='data/hard_cases_plus_board30.csv',
        hard_case_repeat=16,
    )

    result = train(config)
    print(f"Test metrics: {result.test_metrics}")

    run_dir = Path('artifacts/training/scalar_fixed_preprocess_v1')
    run_dir.mkdir(parents=True, exist_ok=True)
    result.model.save(run_dir / 'model.keras')
    hist_dict = {k: [float(v) for v in vals] for k, vals in result.history.history.items()}
    (run_dir / 'history.json').write_text(json.dumps(hist_dict, indent=2), encoding='utf-8')
    (run_dir / 'metrics.json').write_text(json.dumps(result.test_metrics, indent=2), encoding='utf-8')
    print(f'Saved artifacts to {run_dir}')

if __name__ == '__main__':
    main()
