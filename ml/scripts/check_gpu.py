#!/usr/bin/env python3
"""Check TensorFlow GPU availability."""

import tensorflow as tf
import os

print("=" * 60)
print("TensorFlow GPU Check")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print()

# List GPU devices
gpus = tf.config.list_physical_devices("GPU")
print(f"GPU devices found: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")

if not gpus:
    print("\nWARNING: No GPU devices found!")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "Not set"))
else:
    print("\n✓ GPU is ready for training!")

print("=" * 60)
