"""QARepVGG-Pro α=1.75 — uses full 2.5 MB budget.
Runs the main training script directly (defaults are now α=1.75, weight_decay=1e-5, etc.)

Usage: cd ml && poetry run python scripts/train_qat_qarepvgg_pro.py
"""
# The main script at train_qat_qarepvgg_pro.py has been updated to default to
# α=1.75, batch=6, QAT LR=1e-5, colour jitter, random erasing, L2 weight decay,
# full 30-epoch QAT schedule, and TFLite parabolic sub-pixel eval.
#
# Run it directly:
#   cd ml && poetry run python scripts/train_qat_qarepvgg_pro.py
