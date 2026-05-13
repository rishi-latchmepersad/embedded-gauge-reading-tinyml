"""Debug script to check cv2.resize behavior on 3D arrays."""

import numpy as np
import cv2

# Create a 3D array with shape (100, 100, 1)
arr_3d = np.random.rand(100, 100, 1).astype(np.float32)
print(f"Original 3D shape: {arr_3d.shape}")

# Resize with cv2
resized = cv2.resize(arr_3d, (224, 224))
print(f"After cv2.resize: {resized.shape}")

# Create a 2D array with shape (100, 100)
arr_2d = np.random.rand(100, 100).astype(np.float32)
print(f"Original 2D shape: {arr_2d.shape}")

# Resize with cv2
resized_2d = cv2.resize(arr_2d, (224, 224))
print(f"After cv2.resize 2D: {resized_2d.shape}")

# Add channel to 2D
with_channel = resized_2d[..., np.newaxis]
print(f"After adding channel: {with_channel.shape}")
