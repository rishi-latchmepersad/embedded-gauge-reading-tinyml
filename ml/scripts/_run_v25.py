import subprocess, sys

VENV = "/home/rishi_latchmepersad/.cache/poetry/virtualenvs/embedded-gauge-reading-tinyml-fB-DUwEd-py3.12/bin/python"
PROJECT = "/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
OUTDIR = f"{PROJECT}/artifacts/training/polar_vote_hardcase_improved_v25"

cmd = [
    VENV,
    f"{PROJECT}/scripts/train_polar_angle_classifier_manifest.py",
    "--manifest-path", "data/weighted_full_range_v1.csv",
    "--crop-boxes", "data/rectified_crop_boxes_v5_all.csv",
    "--output-dir", OUTDIR,
    "--gauge-id", "littlegood_home_temp_gauge_c",
    "--representation", "vote",
    "--input-mode", "rgb_edge6_vote7",
    "--structure-mode", "vote",
    "--target-mode", "sweep",
    "--sweep-kernel", "reflect",
    "--bins", "180",
    "--sigma-bins", "1.0",
    "--epochs", "120",
    "--batch-size", "16",
    "--learning-rate", "5e-4",
    "--base-filters", "24",
    "--head-units", "96",
    "--dropout", "0.15",
    "--center-search-px", "3",
    "--center-mode", "image_center",
    "--vote-decode-mode", "topk_expectation",
    "--vote-decode-temperature", "0.5",
    "--vote-decode-topk", "10",
    "--loss-mode", "balanced_softmax",
    "--fraction-loss-weight", "0.15",
    "--fraction-loss-delta", "0.04",
    "--label-smoothing", "0.0",
    "--max-shift-bins", "3",
    "--seed", "21",
    "--extra-eval-manifest", "data/hard_cases_plus_board30_valid_with_new6.csv",
]

print("[V25] Starting training")
sys.stdout.flush()
subprocess.run(cmd, cwd=PROJECT)
