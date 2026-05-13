import numpy as np
import tensorflow as tf

N = 8
polar_images = np.random.rand(N, 224, 224, 3).astype(np.float32)
values = np.random.rand(N).astype(np.float32)
masks = np.random.rand(N, 224, 224, 1).astype(np.float32)

print('masks shape before dataset:', masks.shape)

dataset = tf.data.Dataset.from_tensor_slices((polar_images, values, masks))
dataset = dataset.map(lambda x, y, m: (x, {"gauge_value": y, "needle_mask": m}))
dataset = dataset.batch(4)

for batch_x, batch_y in dataset.take(1):
    print('Type of batch_y:', type(batch_y))
    print('Keys in batch_y:', list(batch_y.keys()))
    for k, v in batch_y.items():
        print(f'  {k}: shape={v.shape}')
    
    # Check if order matches Python dict order
    print('Items in order:')
    for k, v in batch_y.items():
        print(f'  {k}: {v.shape}')
