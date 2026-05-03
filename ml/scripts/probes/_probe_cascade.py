import numpy as np
import tensorflow as tf
from embedded_gauge_reading_tinyml import geometry_cascade as gc

class M:
    def __init__(self, outs):
        self.outs = outs
        self.calls = 0
    def __call__(self, inputs, training=False):
        print(f"call {self.calls}", flush=True)
        out = self.outs[self.calls]
        self.calls += 1
        return out

localizer = M([
    {
        "gauge_value": tf.constant([[17.0]], dtype=tf.float32),
        "keypoint_heatmaps": tf.constant(np.full((1, 4, 4, 2), 0.2, dtype=np.float32), dtype=tf.float32),
        "keypoint_coords": tf.constant([[[110.0, 110.0], [130.0, 110.0]]], dtype=tf.float32),
    }
])
source_image = np.zeros((224, 224, 3), dtype=np.uint8)
print("before", flush=True)
result = gc.run_geometry_cascade(
    model=localizer,
    source_image=source_image,
    base_crop_box_xyxy=(0.0, 0.0, 224.0, 224.0),
    image_height=224,
    image_width=224,
    confidence_threshold=0.5,
    recrop_scale=0.75,
    min_recrop_size=64.0,
)
print("after", result.final_value, result.used_second_pass, flush=True)
