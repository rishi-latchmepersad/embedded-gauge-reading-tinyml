"""Convert YUV422 captures to PNG for visual inspection."""
import numpy as np
from PIL import Image
from pathlib import Path
import sys

caps_dir = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml/captured_images")
pattern = sys.argv[1] if len(sys.argv) > 1 else "capture_2026-04-18_17-*.yuv422"

for cap in sorted(caps_dir.glob(pattern)):
    if cap.stat().st_size == 0:
        print(f"SKIP {cap.name} (empty)")
        continue
    raw = cap.read_bytes()
    yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(224, 112, 4)
    luma = np.empty((224, 224), dtype=np.uint8)
    luma[:, 0::2] = yuyv[:, :, 0]
    luma[:, 1::2] = yuyv[:, :, 2]
    out = caps_dir / cap.with_suffix(".png").name
    Image.fromarray(luma, mode="L").save(out)
    print(f"saved {out.name}  mean_luma={luma.mean():.1f}")
