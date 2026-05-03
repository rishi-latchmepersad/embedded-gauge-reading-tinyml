# Justfile for embedded-gauge-reading-tinyml
# Install `just` from https://github.com/casey/just
# Run any recipe with: just <recipe-name>

# Default recipe — show available commands
default:
    @just --list

# ---------------------------------------------------------------------------
# Firmware (Windows / PowerShell side)
# ---------------------------------------------------------------------------

# Flash FSBL + App + optional model to STM32N657 via ST-Link
flash:
    cd firmware/stm32/n657 && flash_boot.bat

# Flash with model blob included
flash-model:
    cd firmware/stm32/n657 && set FLASH_MODEL=1 && flash_boot.bat

# ---------------------------------------------------------------------------
# ML (WSL / Ubuntu side — run from inside WSL)
# ---------------------------------------------------------------------------

# Run the classical CV baseline from WSL
baseline:
    cd ml && bash scripts/wsl_ml.sh baseline

# Run CNN training from WSL
train:
    cd ml && bash scripts/wsl_ml.sh train

# Run pytest suite from WSL
test:
    cd ml && bash scripts/wsl_ml.sh pytest

# Export deployable board artifacts from WSL
export:
    cd ml && bash scripts/wsl_ml.sh export

# Search for best MobileNetV2 width that fits N6 memory from WSL
fit-search:
    cd ml && bash scripts/wsl_ml.sh fit-search

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

# Clean transient outputs from tmp/ (keeps build_artifacts and single_image_baseline)
clean-tmp:
    rm -rf tmp/tmp_*/
    rm -f tmp/*.log
    rm -f tmp/*.txt
    rm -f tmp/*.png

# Full tmp wipe — use with caution
wipe-tmp:
    rm -rf tmp/*

# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

# Check that Poetry and TensorFlow GPU are visible inside WSL
gpu-check:
    cd ml && bash scripts/wsl_ml.sh gpu-check

# Install Poetry dependencies inside WSL
setup:
    cd ml && ~/.local/bin/poetry install --with dev
