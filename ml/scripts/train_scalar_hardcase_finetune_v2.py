#!/usr/bin/env python3
"""Fine-tune from the best pretrained scalar model with aggressive hard-case repetition."""

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
    # Start from the gpu5_recover model which was already fine-tuned from gpu3
    init_model = Path('artifacts/training/no_cal_hardpush_gpu5_recover/model.keras')
    if not init_model.exists():
        print(f'WARNING: init model not found at {init_model}, training from scratch')
        init_model = None
    else:
        print(f'Initializing from {init_model}')

    config = TrainConfig(
        model_family='mobilenet_v2',
        epochs=30,
        learning_rate=1e-6,
        batch_size=16,
        seed=21,
        augment_training=True,
        board_style_augment_prob=0.5,
        device='cpu',
        gpu_memory_growth=False,
        mobilenet_alpha=1.0,
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=True,  # Fine-tune full model
        mobilenet_warmup_epochs=0,
        mobilenet_head_units=128,
        mobilenet_head_dropout=0.2,
        edge_focus_strength=2.5,
        hard_case_manifest='data/hard_cases_plus_board30.csv',
        hard_case_repeat=32,
        init_model_path=str(init_model) if init_model and init_model.exists() else None,
    )

    result = train(config)
    print(f"Test metrics: {result.test_metrics}")

    run_dir = Path('artifacts/training/scalar_hardcase_finetune_v2')
    run_dir.mkdir(parents=True, exist_ok=True)
    result.model.save(run_dir / 'model.keras')
    hist_dict = {k: [float(v) for v in vals] for k, vals in result.history.history.items()}
    (run_dir / 'history.json').write_text(json.dumps(hist_dict, indent=2), encoding='utf-8')
    (run_dir / 'metrics.json').write_text(json.dumps(result.test_metrics, indent=2), encoding='utf-8')
    print(f'Saved artifacts to {run_dir}')

if __name__ == '__main__':
    main()
