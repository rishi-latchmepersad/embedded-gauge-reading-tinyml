"""Save the fixed training crop from a recent capture as a PNG for visual inspection."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "src")
from embedded_gauge_reading_tinyml.board_crop_compare import load_yuv422_capture_as_rgb
from PIL import Image

CAPTURES_DIR = Path("../captured_images")
IMAGE_SIZE = 224
TX0, TY0, TX1, TY1 = int(0.103*224), int(0.254*224), int(0.794*224), int(0.803*224)

# a known training sample at ~34C for comparison
TRAINING_SAMPLE = Path("../captured_images/capture_2026-04-03_15-46-04.png")  # labelled 19C
TRAINING_SAMPLE2 = Path("../captured_images/capture_2026-04-03_08-20-49.png")  # labelled 45C

yuv = sorted(CAPTURES_DIR.glob("capture_2026-04-20_16-*.yuv422"))[0]
rgb = load_yuv422_capture_as_rgb(yuv, image_width=IMAGE_SIZE, image_height=IMAGE_SIZE)

# save full frame
Image.fromarray(rgb).save("/tmp/full_frame.png")
# save fixed crop
crop = rgb[TY0:TY1, TX0:TX1]
Image.fromarray(crop).save("/tmp/fixed_crop.png")

print(f"Full frame saved: /tmp/full_frame.png  shape={rgb.shape}")
print(f"Fixed crop saved: /tmp/fixed_crop.png  shape={crop.shape}")
print(f"Crop luma mean: {rgb[TY0:TY1, TX0:TX1, 0].mean():.1f}")
print(f"Full frame luma mean: {rgb[:,:,0].mean():.1f}")

# check a training PNG for comparison
if TRAINING_SAMPLE.exists():
    tr = np.array(Image.open(TRAINING_SAMPLE).convert('RGB'))
    print(f"Training sample {TRAINING_SAMPLE.name}: shape={tr.shape} luma_mean={tr[:,:,0].mean():.1f}")
