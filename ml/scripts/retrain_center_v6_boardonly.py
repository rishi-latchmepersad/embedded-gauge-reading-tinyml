"""Retrain center detector with ONLY manual board captures (no PXL photos)."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


PROJECT_ROOT = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")
MANUAL_DIR = PROJECT_ROOT / "ml" / "data" / "center_training_manual"
ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts" / "deployment"

INPUT_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 150
PATIENCE = 20
LEARNING_RATE = 5e-5  # Lower LR for smaller dataset


def build_model(input_shape=(224, 224, 3)):
    base = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        alpha=0.5,  # Smaller model for less data
    )
    base.trainable = True
    for layer in base.layers[:-15]:
        layer.trainable = False

    x = base.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(2, activation="sigmoid", name="center_xy")(x)

    model = keras.Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def load_manual_data():
    with open(MANUAL_DIR / "metadata.json") as f:
        entries = json.load(f)
    
    data = []
    for e in entries:
        cx = np.clip(e["center_x"], 0.0, 1.0)
        cy = np.clip(e["center_y"], 0.0, 1.0)
        data.append({
            "path": str(MANUAL_DIR / "images" / e["image"]),
            "cx": cx,
            "cy": cy,
        })
    
    random.shuffle(data)
    return data


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
                    # Small translation (simulates OBB jitter)
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
                    
                    # Slight brightness/contrast
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
    data = load_manual_data()
    print(f"Total manual entries: {len(data)}")
    
    # 80/20 split
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    model = build_model()
    model.summary()
    
    train_gen = generator(train_data, BATCH_SIZE, augment=True)
    val_gen = generator(val_data, BATCH_SIZE, augment=False)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(ARTIFACTS / "center_model_v6_boardonly.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
        ),
    ]
    
    print("\nTraining...")
    history = model.fit(
        train_gen,
        steps_per_epoch=max(1, len(train_data) // BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, len(val_data) // BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    
    # Evaluate on val set
    val_gen_eval = generator(val_data, BATCH_SIZE, augment=False)
    val_images, val_labels = [], []
    for _ in range(len(val_data) // BATCH_SIZE + 1):
        try:
            x, y = next(val_gen_eval)
            val_images.extend(x)
            val_labels.extend(y)
        except StopIteration:
            break
    
    val_images = np.array(val_images[:len(val_data)])
    val_labels = np.array(val_labels[:len(val_data)])
    preds = model.predict(val_images, verbose=0)
    errors_px = np.sqrt((preds[:, 0] - val_labels[:, 0])**2 + (preds[:, 1] - val_labels[:, 1])**2) * INPUT_SIZE
    val_mae_px = np.mean(errors_px)
    print(f"\nVal MAE: {val_mae_px:.4f} px")
    
    # Export int8
    print("\nExporting int8...")
    model = keras.models.load_model(str(ARTIFACTS / "center_model_v6_boardonly.keras"))
    
    def rep_dataset():
        for i in range(min(100, len(train_data))):
            e = train_data[i]
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
    out_dir = ARTIFACTS / "center_model_v6_boardonly_int8"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_int8.tflite", "wb") as f:
        f.write(tflite_model)
    
    print(f"Exported to {out_dir / 'model_int8.tflite'}")
    print(f"Final val MAE: {val_mae_px:.4f} px")


if __name__ == "__main__":
    main()
