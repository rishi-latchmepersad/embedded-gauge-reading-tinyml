$ErrorActionPreference = 'Stop'

# Launch the WSL tiny MobileNetV2 training wrapper from PowerShell.
$wslCommand = 'cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && bash scripts/run_mobilenetv2_tiny_full.sh'
wsl.exe -d Ubuntu-24.04 --exec /bin/bash -lc $wslCommand
