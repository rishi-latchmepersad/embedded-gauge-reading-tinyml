param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = 'Stop'

# Keep the fine-tune on Windows CPU so we avoid WSL GPU hangs.
$env:CUDA_VISIBLE_DEVICES = ''
$env:TF_CPP_MIN_LOG_LEVEL = '2'

$repoRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repoRoot 'artifacts\training_logs'
$logFile = Join-Path $logDir 'scalar_head_finetune_from_best_board_weak_focus_cpu.log'
$baseModel = Join-Path $repoRoot 'artifacts\training\scalar_full_finetune_from_best_board30_clean_plus_new6\model.keras'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Push-Location $repoRoot
try {
  Write-Host "[WRAPPER] Starting board weak-focus head-only fine-tune on Windows CPU."
  Write-Host "[WRAPPER] Base model: $baseModel"
  Write-Host "[WRAPPER] Log file: $logFile"

  & poetry run python -u scripts/finetune_scalar_from_best.py `
    --base-model $baseModel `
    --device cpu `
    --hard-case-manifest data/board_weak_focus.csv `
    --hard-case-repeat 8 `
    --edge-focus-strength 1.5 `
    --epochs 6 `
    --learning-rate 5e-6 `
    --run-name scalar_head_finetune_from_best_board_weak_focus_cpu `
    @Args 2>&1 | Tee-Object -FilePath $logFile
}
finally {
  Pop-Location
}
