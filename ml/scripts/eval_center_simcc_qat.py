#!/usr/bin/env python3
"""Evaluate a saved center-detector + SimCC model on the grouped manifest.

This script reuses the training pipeline's manifest parsing and batch
construction so we can score a saved float or QAT model without rerunning the
full training job.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from embedded_gauge_reading_tinyml.quantize_compat import quantize_load_scope  # noqa: E402

DEFAULT_BATCH_SIZE = 16
DEFAULT_TEST_FRACTION = 0.10
DEFAULT_VAL_FRACTION = 0.15
MANIFEST_PATH = PROJECT_ROOT / "data" / "labelled_captured_images.json"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a saved center-detector + SimCC model on a holdout split."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Which holdout split to score after the manifest is partitioned.",
    )
    return parser.parse_args()


def _split_samples(
    samples: list[object],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[object], list[object], list[object]]:
    """Split manifest rows into train, validation, and test partitions."""

    split_labels = np.array(
        [1 if getattr(sample, "tip_xy", None) is not None else 0 for sample in samples],
        dtype=np.int32,
    )
    sample_indices = np.arange(len(samples), dtype=np.int32)
    train_val_indices, test_indices = train_test_split(
        sample_indices,
        test_size=max(test_fraction, 0.05),
        random_state=seed,
        shuffle=True,
        stratify=split_labels if len(np.unique(split_labels)) > 1 else None,
    )
    relative_val_fraction = val_fraction / max(1.0 - test_fraction, 1e-6)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=max(relative_val_fraction, 0.05),
        random_state=seed + 1,
        shuffle=True,
        stratify=split_labels[train_val_indices]
        if len(np.unique(split_labels[train_val_indices])) > 1
        else None,
    )
    return (
        [samples[int(index)] for index in train_indices],
        [samples[int(index)] for index in val_indices],
        [samples[int(index)] for index in test_indices],
    )


def main() -> None:
    """Load a saved model and score it on the requested holdout split."""

    args = _parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be in (0, 1).")
    if not (0.0 < args.test_fraction < 1.0):
        raise ValueError("--test-fraction must be in (0, 1).")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be < 1.0.")

    with quantize_load_scope():
        model = tf.keras.models.load_model(
            args.model_path,
            compile=False,
            safe_mode=False,
        )

    os.environ.setdefault("TF_SKIP_EXPLICIT_GPU_CONFIG", "1")
    training = importlib.import_module("train_qat_obb_simcc_combined")
    load_combined_samples = training.load_combined_samples
    CombinedGeometrySequence = training.CombinedGeometrySequence
    evaluate_sequence = training._evaluate_sequence

    samples = load_combined_samples(args.manifest, include_temperature_head=True)
    if not samples:
        raise ValueError(f"No usable samples were found in {args.manifest}.")

    _train_samples, val_samples, test_samples = _split_samples(
        samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    scored_samples = val_samples if args.split == "val" else test_samples
    sequence = CombinedGeometrySequence(
        scored_samples,
        batch_size=args.batch_size,
        include_temperature_head=True,
        augment=False,
        seed=args.seed,
    )

    report = evaluate_sequence(
        model,
        sequence,
        include_temperature_head="gauge_value" in model.output_names,
    )
    payload = {
        "model_path": args.model_path.as_posix(),
        "manifest": args.manifest.as_posix(),
        "split": args.split,
        "sample_count": len(scored_samples),
        "metrics": report,
        "output_names": list(model.output_names),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
