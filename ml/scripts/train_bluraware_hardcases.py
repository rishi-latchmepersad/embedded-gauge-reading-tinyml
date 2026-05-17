#!/usr/bin/env python3
"""Train the blur-aware MobileNetV2 reader that performed best on hard cases."""

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
        model_family='mobilenet_v2_bluraware_reader',
        epochs=18,
        learning_rate=5e-6,
        batch_size=4,
        seed=21,
        augment_training=True,
        board_style_augment_prob=0.5,
        device='gpu',
        gpu_memory_growth=True,
        mobilenet_alpha=0.35,
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=True,
        mobilenet_warmup_epochs=4,
        mobilenet_unfreeze_last_n=12,
        mobilenet_freeze_batchnorm=True,
        mobilenet_head_units=64,
        mobilenet_head_dropout=0.15,
        # The winning reader used a linear head; sigmoid saturation hurt extremes.
        linear_output=True,
        hard_case_manifest='data/hard_cases_plus_board30_valid_with_new5.csv',
        hard_case_repeat=4,
        precomputed_crop_boxes_path='data/rectified_crop_boxes_v5_all.csv',
        edge_focus_strength=1.0,
    )

    result = train(config)
    print(f"Test metrics: {result.test_metrics}")

    run_dir = Path('artifacts/training/bluraware_hardcases_v1')
    run_dir.mkdir(parents=True, exist_ok=True)
    result.model.save(run_dir / "model.keras")
    hist_dict = {k: [float(v) for v in vals] for k, vals in result.history.history.items()}
    (run_dir / "history.json").write_text(json.dumps(hist_dict, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(result.test_metrics, indent=2), encoding="utf-8")
    print(f"Saved artifacts to {run_dir}")

if __name__ == '__main__':
    main()
