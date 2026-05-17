#!/usr/bin/env python3
"""Train a fraction-first MobileNetV2 reader on the full live-board mix.

This recipe keeps the same broad full-range data mix we used for the strongest
scalar CNN, but shifts the training objective so the network learns normalized
sweep fraction first and only converts that fraction back to Celsius at the end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import TrainConfig, train


def main() -> None:
    """Run the fraction-first retrain and save the resulting artifacts."""
    config = TrainConfig(
        model_family="mobilenet_v2_fraction",
        epochs=18,
        learning_rate=5e-6,
        batch_size=4,
        seed=21,
        augment_training=True,
        board_style_augment_prob=0.5,
        device="gpu",
        gpu_memory_growth=True,
        mobilenet_alpha=0.35,
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=True,
        mobilenet_warmup_epochs=2,
        mobilenet_unfreeze_last_n=12,
        mobilenet_freeze_batchnorm=True,
        mobilenet_head_units=64,
        mobilenet_head_dropout=0.15,
        hard_case_manifest="data/unified_training_manifest_v1.csv",
        hard_case_repeat=1,
        val_manifest="data/hard_cases_plus_board30_valid_with_new6.csv",
        test_manifest="data/board_rectified_probe_20260422.csv",
        precomputed_crop_boxes_path="data/rectified_crop_boxes_v5_all.csv",
        range_aware_sampling=True,
        cold_tail_fraction=0.20,
        hot_tail_fraction=0.20,
        oversampling_factor=2.0,
        sweep_fraction_loss_weight=1.0,
        edge_focus_strength=2.0,
        init_model_path="artifacts/training/cnn_full_range_live_mix_v1/full_range_live_mix_v1/model.keras",
    )

    result = train(config)
    print(f"Test metrics: {result.test_metrics}")

    run_dir = Path("artifacts/training/fraction_first_full_range_v1")
    run_dir.mkdir(parents=True, exist_ok=True)
    result.model.save(run_dir / "model.keras")
    hist_dict = {
        key: [float(value) for value in values]
        for key, values in result.history.history.items()
    }
    (run_dir / "history.json").write_text(
        json.dumps(hist_dict, indent=2),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result.test_metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Saved artifacts to {run_dir}")


if __name__ == "__main__":
    main()
