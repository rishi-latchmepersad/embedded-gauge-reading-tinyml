import sys
sys.path.insert(0, 'src')
from embedded_gauge_reading_tinyml.polar_model import build_polar_needle_segmentation_model
import numpy as np

model = build_polar_needle_segmentation_model(polar_size=224, base_filters=32, depth=4)
print('Model outputs:')
for i, output in enumerate(model.outputs):
    print(f'  Output {i} ({model.output_names[i]}): shape={output.shape}')

x = np.random.rand(2, 224, 224, 3).astype(np.float32)
predictions = model.predict(x)
print('Predictions:')
for k, v in predictions.items():
    print(f'  {k}: shape={v.shape}')

# Test with dict targets
y = {'gauge_value': np.random.rand(2).astype(np.float32), 'needle_mask': np.random.rand(2, 224, 224, 1).astype(np.float32)}
model.compile(optimizer='adam', loss={'gauge_value': 'mse', 'needle_mask': 'binary_crossentropy'})
loss = model.train_on_batch(x, y)
print('Loss:', loss)
print('SUCCESS')
