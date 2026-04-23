param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = 'Stop'

# Launch the prod-v0.3 OBB packaging wrapper in Ubuntu WSL.
$wslCommand = 'cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && bash scripts/run_board_package_prod_model_v0_3_obb_raw_int8.sh'
if ($Args.Count -gt 0) {
  $wslCommand += ' ' + ($Args -join ' ')
}

wsl.exe -d Ubuntu-24.04 --exec /bin/bash -lc $wslCommand
