import numpy as np
from embedded_gauge_reading_tinyml import geometry_cascade as gc
print("before-resize", flush=True)
image = np.zeros((224, 224, 3), dtype=np.uint8)
out = gc.resize_with_pad_rgb(image, (0.0, 0.0, 224.0, 224.0), image_size=224)
print(out.shape, flush=True)
print("after-resize", flush=True)
