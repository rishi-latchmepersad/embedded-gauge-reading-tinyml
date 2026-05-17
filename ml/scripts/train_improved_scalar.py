"""Train improved scalar model with range-aware sampling and hard case emphasis."""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import train, TrainConfig

def main():
    config = TrainConfig(
        model_family='mobilenet_v2',
        epochs=50,
        learning_rate=1e-4,
        batch_size=8,
        seed=42,
        augment_training=True,
        board_style_augment_prob=0.5,
        hard_case_eval_manifest='data/hard_cases_plus_board30.csv',
        range_aware_sampling=True,
        cold_tail_fraction=0.15,
        hot_tail_fraction=0.15,
        oversampling_factor=3.0,
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=True,
        mobilenet_alpha=1.0,
        mobilenet_head_units=128,
        mobilenet_head_dropout=0.2,
        # Keep the regression head linear so the model can cover the extremes.
        linear_output=True,
        device='cpu',
        val_fraction=0.15,
        test_fraction=0.15,
    )
    
    result = train(config)
    
    run_dir = PROJECT_ROOT / 'artifacts' / 'training' / 'improved_scalar_v1'
    run_dir.mkdir(parents=True, exist_ok=True)
    result.model.save(run_dir / 'model.keras')
    
    with open(run_dir / 'history.json', 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in result.history.history.items()}, f, indent=2)
    
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(result.test_metrics, f, indent=2)
    
    print(f"\n[TRAIN] Saved to {run_dir}")
    print(f"[TRAIN] Test metrics: {result.test_metrics}")

if __name__ == '__main__':
    main()

