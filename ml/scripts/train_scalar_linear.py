#!/usr/bin/env python3
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
        epochs=40,
        learning_rate=1e-4,
        batch_size=8,
        seed=21,
        augment_training=True,
        board_style_augment_prob=0.5,
        device='cpu',
        gpu_memory_growth=False,
        mobilenet_alpha=1.0,
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=True,
        mobilenet_head_units=128,
        mobilenet_head_dropout=0.2,
        linear_output=True,
        range_aware_sampling=True,
        cold_tail_fraction=0.15,
        hot_tail_fraction=0.15,
        hard_case_manifest='data/hard_cases_plus_board30.csv',
        hard_case_repeat=5,
        val_manifest='data/hard_cases_plus_board30_valid_with_new6.csv',
        test_manifest='data/board_rectified_probe_20260422.csv',
    )

    result = train(config)
    print(f'Test metrics: {result.test_metrics}')

    run_dir = Path('artifacts/training/scalar_linear_no_sigmoid')
    run_dir.mkdir(parents=True, exist_ok=True)
    result.model.save(run_dir / 'model.keras')
    hist_dict = {k: [float(v) for v in vals] for k, vals in result.history.history.items()}
    (run_dir / 'history.json').write_text(json.dumps(hist_dict, indent=2), encoding='utf-8')
    (run_dir / 'metrics.json').write_text(json.dumps(result.test_metrics, indent=2), encoding='utf-8')
    print(f'Saved artifacts to {run_dir}')

if __name__ == '__main__':
    main()