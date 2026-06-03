"""Fine-tune v4 model on manual board annotations."""

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
V4_MODEL = PROJECT_ROOT / "ml" / "artifacts" / "deployment" / "center_model_v4_cdcrop_int8" / "model_int8.tflite"
ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts" / "deployment"

INPUT_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-5  # Very low LR for fine-tuning


def load_v4_as_keras():
    """Load the v4 TFLite model and convert back to Keras for fine-tuning."""
    # Actually, we don't have the Keras weights saved. Let me rebuild the architecture
    # and train from scratch with the right data
    base = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
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


def load_all_data():
    entries = []
    
    # Manual board captures (weighted more heavily by duplicating)
    with open(MANUAL_DIR / "metadata.json") as f:
        manual = json.load(f)
    for e in manual:
        cx = np.clip(e["center_x"], 0.0, 1.0)
        cy = np.clip(e["center_y"], 0.0, 1.0)
        # Add 3x weight to manual annotations
        for _ in range(3):
            entries.append({
                "path": str(MANUAL_DIR / "images" / e["image"]),
                "cx": cx,
                "cy": cy,
                "source": "manual",
            })
    
    # PXL photos (for diversity)
    with open(PXL_DIR / "metadata.json") as f:
        pxl = json.load(f)
    for e in pxl:
        if e["center_x_norm"] < 0 or e["center_x_norm"] > 1 or e["center_y_norm"] < 0 or e["center_y_norm"] > 1:
            continue
        entries.append({
            "path": str(PXL_DIR / e["image_path"]),
            "cx": e["center_x_norm"],
            "cy": e["center_y_norm"],
            "source": "pxl",
        })
    
    random.shuffle(entries)
    return entries


def generator(entries, batch_size, augment=False):
    indices = list(range(len(entries)))
    while True:
        random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            images = []
            labels = []
            for idx in batch_idx:
                e = entries[idx]
                img = cv2.imread(e["path"])
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if augment:
                    if random.random() < 0.5:
                        dx = random.randint(-3, 3)
                        dy = random.randint(-3, 3)
                        M = np.float32([[1, 0, dx], [0, 1, dy]])
                        img = cv2.warpAffine(img, M, (INPUT_SIZE, INPUT_SIZE),
                                             borderValue=(128, 128, 128))
                        cx = e["cx"] + dx / INPUT_SIZE
                        cy = e["cy"] + dy / INPUT_SIZE
                    else:
                        cx, cy = e["cx"], e["cy"]
                    
                    if random.random() < 0.3:
                        alpha = random.uniform(0.9, 1.1)
                        beta = random.randint(-10, 10)
                        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                else:
                    cx, cy = e["cx"], e["cy"]
                
                images.append(img.astype(np.float32))
                labels.append([cx, cy])
            
            if len(images) == 0:
                continue
            yield np.array(images), np.array(labels)


def main():
    print("Loading data...")
    data = load_all_data()
    print(f"Total: {len(data)} (manual x3 + pxl)")
    
    split = int(0.8 * len(data))
    train = data[:split]
    val = data[split:]
    print(f"Train: {len(train)}, Val: {len(val)}")
    
    model = load_v4_as_keras()
    model.summary()
    
    train_gen = generator(train, BATCH_SIZE, augment=True)
    val_gen = generator(val, BATCH_SIZE, augment=False)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(ARTIFACTS / "center_model_v7_ft_manual.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
    ]
    
    print("\nTraining...")
    history = model.fit(
        train_gen,
        steps_per_epoch=max(1, len(train) // BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, len(val) // BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    
    # Evaluate
    val_gen_eval = generator(val, BATCH_SIZE, augment=False)
    val_images, val_labels = [], []
    for _ in range(len(val) // BATCH_SIZE + 1):
        try:
            x, y = next(val_gen_eval)
            val_images.extend(x)
            val_labels.extend(y)
        except StopIteration:
            break
    
    val_images = np.array(val_images[:len(val)])
    val_labels = np.array(val_labels[:len(val)])
    preds = model.predict(val_images, verbose=0)
    errors_px = np.sqrt((preds[:, 0] - val_labels[:, 0])**2 + (preds[:, 1] - val_labels[:, 1])**2) * INPUT_SIZE
    val_mae_px = np.mean(errors_px)
    print(f"\nVal MAE: {val_mae_px:.4f} px")
    
    # Export
    print("\nExporting int8...")
    model = keras.models.load_model(str(ARTIFACTS / "center_model_v7_ft_manual.keras"))
    
    def rep_dataset():
        for i in range(min(100, len(train))):
            e = train[i]
            img = cv2.imread(e["path"])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            yield [img[None, ...]]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    out_dir = ARTIFACTS / "center_model_v7_ft_manual_int8"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_int8.tflite", "wb") as f:
        f.write(tflite_model)
    
    print(f"Exported to {out_dir / 'model_int8.tflite'}")
    print(f"Final val MAE: {val_mae_px:.4f} px")


if __name__ == "__main__":
    main()
