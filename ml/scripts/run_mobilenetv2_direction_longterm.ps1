param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = 'Stop'

# Launch the long-term MobileNetV2 direction wrapper in Ubuntu WSL.
$wslCommand = 'cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && bash scripts/run_mobilenetv2_direction_longterm.sh'
if ($Args.Count -gt 0) {
  $wslCommand += ' ' + ($Args -join ' ')
}

wsl.exe -d Ubuntu-24.04 --exec /bin/bash -lc $wslCommand
