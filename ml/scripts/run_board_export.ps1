param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$wslCommand = 'cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && bash scripts/run_board_export.sh'
if ($Args.Count -gt 0) {
  $wslCommand += ' ' + ($Args -join ' ')
}

wsl.exe -d Ubuntu-24.04 --exec /bin/bash -lc $wslCommand
