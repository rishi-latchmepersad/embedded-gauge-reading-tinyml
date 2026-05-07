"""Create metadata JSON for canonical splits."""

import json
from pathlib import Path

metadata = {
    "version": "v1",
    "created_from": "canonical_manifest_v1.csv",
    "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "random_state": 42,
    "bin_size": 5.0,
    "stratified": False,
    "reason": "All samples are hard_case or board_capture; some value bins had <2 samples",
    "statistics": {
        "total_samples": 141,
        "train_samples": 98,
        "val_samples": 21,
        "test_samples": 22,
        "value_range": {"min": -30.0, "max": 50.0},
    },
}

output_path = Path("data/splits/canonical_split_v1_metadata.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to {output_path}")
