import sys

sys.path.insert(0, "src")
from embedded_gauge_reading_tinyml.polar_model import (
    build_polar_needle_segmentation_model,
)
import numpy as np

model = build_polar_needle_segmentation_model(polar_size=224, base_filters=32, depth=4)
print("Model output_names:", model.output_names)
print("Model outputs:")
for i, output in enumerate(model.outputs):
    print(f"  Output {i}: name={output.name}, shape={output.shape}")

# Test with dict targets in the SAME order as model.output_names
x = np.random.rand(2, 224, 224, 3).astype(np.float32)

# Order targets to match model.output_names
if model.output_names[0] == "gauge_value":
    y = {
        "gauge_value": np.random.rand(2, 1).astype(np.float32),
        "needle_mask": np.random.rand(2, 224, 224, 1).astype(np.float32),
    }
else:
    y = {
        "needle_mask": np.random.rand(2, 224, 224, 1).astype(np.float32),
        "gauge_value": np.random.rand(2, 1).astype(np.float32),
    }

print("Target shapes:")
for k, v in y.items():
    print(f"  {k}: {v.shape}")

model.compile(
    optimizer="adam", loss={"gauge_value": "mse", "needle_mask": "binary_crossentropy"}
)
loss = model.train_on_batch(x, y)
print("Loss:", loss)
print("SUCCESS")
