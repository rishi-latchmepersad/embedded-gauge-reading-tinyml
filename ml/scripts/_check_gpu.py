"""Check GPU availability and training capability."""
from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

print(f"TF version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU'))}")
