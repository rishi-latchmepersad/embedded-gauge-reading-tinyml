"""Train a fraction-first MobileNetV2 regressor on the full hard-case mix.

This wrapper keeps the sweep-fraction objective as the primary target, but it
uses the actual hard-case pool instead of a generic full-range manifest so the
model sees the cold and hot ends during training instead of only at eval time.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import TrainConfig, train  # noqa: E402


def main() -> None:
    """Configure and launch the fraction-first hard-case training run."""
    artifacts_dir = PROJECT_ROOT / "artifacts" / "training" / "fraction_first_hardcases_full_range_v1"
    config = TrainConfig(
        gauge_id="littlegood_home_temp_gauge_c",
        image_height=224,
        image_width=224,
        batch_size=4,
        epochs=24,
        learning_rate=5e-6,
        seed=21,
        val_fraction=0.15,
        test_fraction=0.15,
        crop_pad_ratio=0.25,
        augment_training=True,
        board_style_augment_prob=0.5,
        device="gpu",
        gpu_memory_growth=True,
        mixed_precision=False,
        edge_focus_strength=2.0,
        model_family="mobilenet_v2_fraction",
        mobilenet_pretrained=True,
        mobilenet_backbone_trainable=True,
        mobilenet_alpha=0.35,
        mobilenet_head_units=64,
        mobilenet_head_dropout=0.15,
        mobilenet_warmup_epochs=2,
        mobilenet_unfreeze_last_n=12,
        mobilenet_freeze_batchnorm=True,
        hard_case_manifest="data/hard_cases_plus_board30.csv",
        hard_case_repeat=3,
        range_aware_sampling=True,
        cold_tail_fraction=0.20,
        hot_tail_fraction=0.20,
        oversampling_factor=2.0,
        sweep_fraction_loss_weight=1.5,
        precomputed_crop_boxes_path="data/rectified_crop_boxes_v5_all.csv",
        init_model_path=(
            PROJECT_ROOT
            / "artifacts"
            / "training"
            / "cnn_full_range_live_mix_v1"
            / "full_range_live_mix_v1"
            / "model.keras"
        ).as_posix(),
    )
    print("[WRAPPER] Starting fraction-first hard-case full-range training.")
    print(f"[WRAPPER] Config: {json.dumps(asdict(config), indent=2, sort_keys=True)}")
    result = train(config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.keras"
    history_path = artifacts_dir / "history.json"
    metrics_path = artifacts_dir / "metrics.json"
    result.model.save(model_path)
    history_payload = {
        key: [float(v) for v in values]
        for key, values in result.history.history.items()
    }
    history_path.write_text(json.dumps(history_payload, indent=2, sort_keys=True))
    metrics_payload: dict[str, Any] = {
        "config": asdict(config),
        "label_summary": asdict(result.label_summary),
        "test_metrics": result.test_metrics,
        "baseline_test_mae": float(result.baseline_test_mae),
        "dropped_out_of_sweep": int(result.dropped_out_of_sweep),
        "model_path": model_path.as_posix(),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
    print(f"[WRAPPER] Artifacts saved under: {artifacts_dir}")
    print(f"[WRAPPER] Model saved: {model_path}")
    print(f"[WRAPPER] Label summary: {result.label_summary}")
    print(f"[WRAPPER] Test metrics: {result.test_metrics}")


if __name__ == "__main__":
    main()
