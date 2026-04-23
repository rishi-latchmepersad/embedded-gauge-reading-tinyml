param()

$ErrorActionPreference = 'Stop'

$repoRoot = 'D:\Projects\embedded-gauge-reading-tinyml\ml'
$script = 'scripts/run_mobilenetv2_geometry_cascade_localizer_longterm.sh'

wsl.exe -d Ubuntu-24.04 --exec /bin/bash -lc "cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && bash $script"
