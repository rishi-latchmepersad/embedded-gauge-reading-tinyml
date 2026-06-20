"""Embedded gauge reading with TinyML — gauge reading on STM32N6 NPU."""

# NOTE: GPU memory limit must be set in the training script BEFORE any TF
# imports.  This __init__.py cannot control the limit because TF initialises
# the GPU device during its own import, which happens before this module runs.
