#!/usr/bin/env python3
"""Careful fine-tune from gpu5_recover with light hard-case repetition.

Strategy:
- init from gpu5_recover (best model so far: 5.52C MAE hard cases)
- hard_case_repeat=4 (light, not 32x)
- backbone_trainable=False via patched training.py
- learning_rate=5e-7 (very low)
- epochs=10 (few)
- batch_size=16
- AUG_HEAVY=1 for stronger photometric augmentation
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["AUG_HEAVY"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import train, TrainConfig

def main():
    init_model = Path('artifacts/training/no_cal_hardpush_gpu5_recover/model.keras')
    if not init_model.exists():
        raise FileNotFoundError(f'init model not found at {init_model}')

    config = TrainConfig(
        model_family='mobilenet_v2',
        epochs=10,
        learning_rate=5e-7,
        batch_size=16,
        seed=21,
        augment_training=True,
        board_style_augment_prob=0.3,
        device='cpu',
        gpu_memory_growth=False,
        mobilenet_alpha=1.0,
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=False,
        mobilenet_warmup_epochs=0,
        mobilenet_head_units=128,
        mobilenet_head_dropout=0.2,
        edge_focus_strength=2.0,
        hard_case_manifest='data/unified_training_manifest_v1.csv',
        hard_case_repeat=4,
        val_manifest='data/hard_cases_plus_board30_valid_with_new6.csv',
        test_manifest='data/board_rectified_probe_20260422.csv',
        init_model_path=str(init_model),
        monotonic_pair_strength=0.0,
        interpolation_pair_strength=0.0,
    )
    result = train(config)
    print(f"Test metrics: {result.test_metrics}")
    run_dir = Path('artifacts/training/scalar_careful_finetune_v1')
    run_dir.mkdir(parents=True, exist_ok=True)
    result.model.save(run_dir / 'model.keras')
    hist_dict = {k: [float(v) for v in vals] for k, vals in result.history.history.items()}
    (run_dir / 'history.json').write_text(json.dumps(hist_dict, indent=2), encoding='utf-8')
    (run_dir / 'metrics.json').write_text(json.dumps(result.test_metrics, indent=2), encoding='utf-8')
    print(f'Saved artifacts to {run_dir}')

if __name__ == '__main__':
    main()
