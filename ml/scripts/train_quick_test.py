"""Quick training script for canonical baseline - minimal version for testing.

This is a simplified version that trains quickly to verify the pipeline works.
For full training, use run_training.py with proper configuration.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add ml/src to path
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import keras
from embedded_gauge_reading_tinyml.board_crop_compare import (
    load_rgb_image,
    resize_with_pad_rgb,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def preprocess_image(image_path: str, target_size: tuple[int, int] = (224, 224)):
    """Load and preprocess an image."""
    img = load_rgb_image(image_path)
    img_resized = resize_with_pad_rgb(img, target_size[1], target_size[0])
    return img_resized.astype(np.float32) / 255.0


def load_data(split_path: Path, repo_root: Path):
    """Load a data split."""
    df = pd.read_csv(split_path)

    images = []
    values = []

    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = repo_root / img_path

        try:
            img = preprocess_image(str(img_path))
            images.append(img)
            values.append(float(row["value"]))
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")

    return np.array(images), np.array(values)


def build_simple_model(input_shape=(224, 224, 3)):
    """Build a simple CNN model for quick testing."""
    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, 3, activation="relu", input_shape=input_shape),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="huber",
        metrics=["mae"],
    )

    return model


def main():
    """Run quick training."""
    repo_root = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")

    # Load data
    logger.info("Loading training data...")
    X_train, y_train = load_data(
        repo_root / "ml/data/splits/canonical_split_v1_train.csv", repo_root
    )
    logger.info(f"Loaded {len(X_train)} training samples")

    logger.info("Loading validation data...")
    X_val, y_val = load_data(
        repo_root / "ml/data/splits/canonical_split_v1_val.csv", repo_root
    )
    logger.info(f"Loaded {len(X_val)} validation samples")

    logger.info("Loading test data...")
    X_test, y_test = load_data(
        repo_root / "ml/data/splits/canonical_split_v1_test.csv", repo_root
    )
    logger.info(f"Loaded {len(X_test)} test samples")

    # Build model
    logger.info("Building model...")
    model = build_simple_model()
    model.summary()

    # Train
    logger.info("Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=16,
        verbose=1,
    )

    # Evaluate
    logger.info("Evaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test, return_dict=True, verbose=1)
    logger.info(f"Test MAE: {test_metrics['mae']:.4f}")

    # Save
    output_dir = repo_root / "ml/artifacts/canonical_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "model.keras")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history.history, f, indent=2)

    # Save predictions
    predictions = model.predict(X_test, verbose=0).flatten()
    results_df = pd.DataFrame(
        {
            "true": y_test,
            "predicted": predictions,
            "abs_error": np.abs(predictions - y_test),
        }
    )
    results_df.to_csv(output_dir / "test_predictions.csv", index=False)

    logger.info(f"Saved artifacts to {output_dir}")
    logger.info("Training complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
