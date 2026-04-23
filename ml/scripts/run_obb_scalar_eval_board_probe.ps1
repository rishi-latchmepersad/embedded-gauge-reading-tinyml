param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = 'Stop'

# Launch the OBB + scalar board-probe evaluation wrapper in Ubuntu WSL.
$wslCommand = 'cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && bash scripts/run_obb_scalar_eval_board_probe.sh'
if ($Args.Count -gt 0) {
  $wslCommand += ' ' + ($Args -join ' ')
}

wsl.exe -d Ubuntu-24.04 --exec /bin/bash -lc $wslCommand
