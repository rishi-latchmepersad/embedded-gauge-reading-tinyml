param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = 'Stop'

# Run the OBB + scalar firmware calibration fit in Ubuntu WSL and shut WSL down after.
wsl.exe --shutdown | Out-Null
try {
  $wslCommand = 'cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && bash scripts/run_calibrate_obb_scalar_firmware.sh'
  if ($Args.Count -gt 0) {
    $wslCommand += ' ' + ($Args -join ' ')
  }

  wsl.exe -d Ubuntu-24.04 --exec /bin/bash -lc $wslCommand
}
finally {
  wsl.exe --shutdown | Out-Null
}
