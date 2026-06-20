"""Upscale 224x224 training PNGs to 320x320 with nearest-neighbor.

This ensures the model sees only 320x320 images during training,
matching the firmware's native 320x320 input.  The previous training
mix of 224→320 upscaled and native 320x320 images caused a ~26px
center-x bias at inference time.
"""
from pathlib import Path
from PIL import Image

CAPTURED_DIR = Path(__file__).resolve().parents[1] / "data" / "captured_images"
UPSCALED_DIR = CAPTURED_DIR.parent / "captured_images_320"
UPSCALED_DIR.mkdir(exist_ok=True)

TARGET_SIZE = (320, 320)

# Process all trainable image files (PNG + JPG), skipping diagnostic derivatives
IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg")
image_files: list[Path] = []
for g in IMAGE_GLOBS:
    image_files.extend(sorted(CAPTURED_DIR.glob(g)))

print(f"Found {len(image_files)} image files in {CAPTURED_DIR}")

upscaled = 0
skipped = 0
for src in image_files:
    # Skip diagnostic derivative images
    name = src.name
    if any(tag in name for tag in (".gray.", "_preview", "_yuy2", "_glare_")):
        continue
    img = Image.open(src).convert("RGB")
    if img.size == TARGET_SIZE:
        # Already 320x320, just copy as PNG
        out_name = src.stem + ".png"
        img.save(UPSCALED_DIR / out_name)
        skipped += 1
        continue
    # Resize with nearest-neighbor (same as training's tf.image.resize nearest)
    img_up = img.resize(TARGET_SIZE, resample=Image.NEAREST)
    out_name = src.stem + ".png"
    img_up.save(UPSCALED_DIR / out_name)
    upscaled += 1

# Also copy YUV files (already 320x320, just reference them)
yuv_files = sorted(CAPTURED_DIR.glob("*.yuv422"))
print(f"Found {len(yuv_files)} YUV files")

print(f"\nResized: {upscaled}, Already 320x320: {skipped}")
print(f"Output: {UPSCALED_DIR}")
print(f"\nNext: update CAPTURED_IMAGES_DIR in train_obb_center_mobilenetv2.py")
print(f"       to point to: {UPSCALED_DIR}")
