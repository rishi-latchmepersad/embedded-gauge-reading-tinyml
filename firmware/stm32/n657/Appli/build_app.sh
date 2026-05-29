#!/bin/bash
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/firmware/stm32/n657/Appli
if [ -f "Debug/n657_Appli.elf" ]; then
    rm Debug/n657_Appli.elf
fi
# Use mingw32-make as the build system
mingw32-make -j4 all 2>&1
