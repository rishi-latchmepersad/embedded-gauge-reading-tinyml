import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from pathlib import Path
from scripts.train_polar_needle import merge_all_manifests, create_polar_dataset
from embedded_gauge_reading_tinyml.polar_model import build_polar_tiny_model
from tensorflow import keras

df = merge_all_manifests(Path(".."))
df_small = df.head(20)

ds = create_polar_dataset(
    df_small,
    batch_size=4,
    shuffle=False,
    use_weights=False,
    augment=False,
    polar_size=224,
)

model = build_polar_tiny_model(polar_size=224)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={"gauge_value": keras.losses.MeanSquaredError()},
    metrics={"gauge_value": ["mae"]},
)

history = model.fit(ds, epochs=2, verbose=1)
print("Training loop OK!")
print("Epoch 1 mae:", history.history["gauge_value_mae"][0])
print("Epoch 2 mae:", history.history["gauge_value_mae"][1])
