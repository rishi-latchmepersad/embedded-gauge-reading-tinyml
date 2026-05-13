import sys
sys.path.insert(0, 'src')
from embedded_gauge_reading_tinyml.polar_model import build_polar_needle_segmentation_model
import numpy as np
import tensorflow as tf

model = build_polar_needle_segmentation_model(polar_size=224, base_filters=32, depth=4)
print('Model output_names:', model.output_names)
for i, output in enumerate(model.outputs):
    print(f'  Output {i}: name={output.name}, shape={output.shape}')

# Create synthetic data with EXACT same shapes as training script
N = 16
x = np.random.rand(N, 224, 224, 3).astype(np.float32)
y_val = np.random.rand(N).astype(np.float32)
m_val = np.random.rand(N, 224, 224, 1).astype(np.float32)

print('x shape:', x.shape)
print('y_val shape:', y_val.shape)
print('m_val shape:', m_val.shape)

# Create dataset exactly like training script
dataset = tf.data.Dataset.from_tensor_slices((x, y_val, m_val))
dataset = dataset.map(lambda a, b, c: (a, {"gauge_value": b, "needle_mask": c}))
dataset = dataset.batch(8)

# Check a batch
for batch_x, batch_y in dataset.take(1):
    print('Batch target keys:', list(batch_y.keys()))
    for k, v in batch_y.items():
        print(f'  {k}: shape={v.shape}')

# Compile exactly like training script
model.compile(
    optimizer='adam',
    loss={
        "gauge_value": 'mse',
        "needle_mask": 'binary_crossentropy',
    },
    loss_weights={
        "gauge_value": 1.0,
        "needle_mask": 1.0,
    },
)

# Try fit for 1 epoch
print('Trying model.fit() for 1 epoch...')
model.fit(dataset, epochs=1)
print('SUCCESS')
