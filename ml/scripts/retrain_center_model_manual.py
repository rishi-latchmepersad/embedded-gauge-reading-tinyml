"""Retrain center detector with manual annotations for board captures + PXL photos."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


PROJECT_ROOT = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")
MANUAL_DIR = PROJECT_ROOT / "ml" / "data" / "center_training_manual"
PXL_DIR = PROJECT_ROOT / "ml" / "data" / "center_training_crops"
ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts" / "deployment"

INPUT_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4


def build_mobilenetv2_center(input_shape=(224, 224, 3)):
    base = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        alpha=1.0,
    )
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    x = base.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(2, activation="sigmoid", name="center_xy")(x)

    model = keras.Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def load_dataset(manual_dir, pxl_dir):
    entries = []
    
    # Load manual board captures
    with open(manual_dir / "metadata.json") as f:
        manual_entries = json.load(f)
    for e in manual_entries:
        # Clip out-of-range values
        cx = np.clip(e["center_x"], 0.0, 1.0)
        cy = np.clip(e["center_y"], 0.0, 1.0)
        entries.append({
            "path": str(manual_dir / "images" / e["image"]),
            "cx": cx,
            "cy": cy,
            "source": "manual",
        })
    
    # Load PXL photos
    with open(pxl_dir / "metadata.json") as f:
        pxl_entries = json.load(f)
    for e in pxl_entries:
        # Only include PXL entries with valid labels
        if e["center_x_norm"] < 0 or e["center_x_norm"] > 1 or e["center_y_norm"] < 0 or e["center_y_norm"] > 1:
            continue
        # Filter out entries with repaired OBB labels (those near 0.5,0.5)
        # These were repaired because rim estimator gave outliers
        # For PXL entries, keep all since they're actual measurements
        entries.append({
            "path": str(pxl_dir / e["image_path"]),
            "cx": e["center_x_norm"],
            "cy": e["center_y_norm"],
            "source": "pxl",
        })
    
    random.shuffle(entries)
    return entries


def data_generator(entries, batch_size, augment=False):
    """Keras-compatible data generator."""
    indices = list(range(len(entries)))
    random.shuffle(indices)
    
    while True:
        random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            images = []
            labels = []
            
            for idx in batch_indices:
                e = entries[idx]
                img = cv2.imread(e["path"])
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if augment and random.random() < 0.5:
                    # Slight jitter to simulate board variations
                    dx = random.randint(-4, 4)
                    dy = random.randint(-4, 4)
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    img = cv2.warpAffine(img, M, (INPUT_SIZE, INPUT_SIZE),
                                         borderValue=(128, 128, 128))
                    # Adjust label accordingly
                    cx = e["cx"] + dx / INPUT_SIZE
                    cy = e["cy"] + dy / INPUT_SIZE
                else:
                    cx, cy = e["cx"], e["cy"]
                
                images.append(img.astype(np.float32))
                labels.append([cx, cy])
            
            if len(images) == 0:
                continue
            
            yield np.array(images), np.array(labels)


def main():
    print("Loading datasets...")
    entries = load_dataset(MANUAL_DIR, PXL_DIR)
    print(f"Total entries: {len(entries)}")
    
    # Split train/val (80/20)
    split = int(0.8 * len(entries))
    train_entries = entries[:split]
    val_entries = entries[split:]
    
    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}")
    
    # Count sources
    train_manual = sum(1 for e in train_entries if e["source"] == "manual")
    train_pxl = sum(1 for e in train_entries if e["source"] == "pxl")
    val_manual = sum(1 for e in val_entries if e["source"] == "manual")
    val_pxl = sum(1 for e in val_entries if e["source"] == "pxl")
    print(f"Train: {train_manual} manual + {train_pxl} pxl")
    print(f"Val: {val_manual} manual + {val_pxl} pxl")
    
    # Build model
    model = build_mobilenetv2_center()
    model.summary()
    
    # Generators
    train_gen = data_generator(train_entries, BATCH_SIZE, augment=True)
    val_gen = data_generator(val_entries, BATCH_SIZE, augment=False)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(ARTIFACTS / "center_model_v5_manual.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        train_gen,
        steps_per_epoch=max(1, len(train_entries) // BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, len(val_entries) // BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Evaluate
    val_gen_eval = data_generator(val_entries, BATCH_SIZE, augment=False)
    val_images = []
    val_labels = []
    for _ in range(len(val_entries) // BATCH_SIZE + 1):
        try:
            x, y = next(val_gen_eval)
            val_images.extend(x)
            val_labels.extend(y)
        except StopIteration:
            break
    
    val_images = np.array(val_images[:len(val_entries)])
    val_labels = np.array(val_labels[:len(val_entries)])
    
    preds = model.predict(val_images, verbose=0)
    errors_px = np.sqrt((preds[:, 0] - val_labels[:, 0])**2 + (preds[:, 1] - val_labels[:, 1])**2) * INPUT_SIZE
    val_mae_px = np.mean(errors_px)
    print(f"\nVal MAE (px): {val_mae_px:.4f}")
    
    # Export to TFLite int8
    print("\nExporting to int8 TFLite...")
    model = keras.models.load_model(str(ARTIFACTS / "center_model_v5_manual.keras"))
    
    def representative_dataset():
        for i in range(min(100, len(train_entries))):
            e = train_entries[i]
            img = cv2.imread(e["path"])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            yield [img[None, ...]]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    out_dir = ARTIFACTS / "center_model_v5_manual_int8"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_int8.tflite", "wb") as f:
        f.write(tflite_model)
    
    print(f"Exported to {out_dir / 'model_int8.tflite'}")
    print(f"Final val MAE: {val_mae_px:.4f} px")


if __name__ == "__main__":
    main()
