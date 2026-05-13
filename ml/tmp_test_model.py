import sys
sys.path.insert(0, 'src')
from embedded_gauge_reading_tinyml.polar_model import build_polar_needle_segmentation_model
import numpy as np

model = build_polar_needle_segmentation_model(polar_size=224, base_filters=32, depth=4)

# Compile with loss dict matching output order
model.compile(optimizer='adam', loss={'gauge_value': 'mse', 'needle_mask': 'binary_crossentropy'})

x = np.random.rand(2, 224, 224, 3).astype(np.float32)

# Target dict matching output order
y = {'gauge_value': np.random.rand(2).astype(np.float32), 'needle_mask': np.random.rand(2, 224, 224, 1).astype(np.float32)}

print('Model outputs:', model.output_names)
print('Target keys:', list(y.keys()))

try:
    loss = model.train_on_batch(x, y)
    print('Loss:', loss)
    print('SUCCESS')
except Exception as e:
    print('ERROR:', e)
