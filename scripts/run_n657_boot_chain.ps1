param()

# Step 1 programs the external-flash application image.
$appFlashScript = Join-Path $PSScriptRoot "flash_n657.ps1"

# Step 2 downloads the RAM-resident FSBL blink demo and starts it.
$fsblBlinkScript = Join-Path $PSScriptRoot "run_fsbl_blink.ps1"

Write-Host "Programming the N657 application image..."
& $appFlashScript
if ($LASTEXITCODE -ne 0) {
    throw "Application flash failed with exit code $LASTEXITCODE"
}

Write-Host "Launching the FSBL blink demo..."
& $fsblBlinkScript
if ($LASTEXITCODE -ne 0) {
    throw "FSBL blink demo failed with exit code $LASTEXITCODE"
}

Write-Host "Boot-chain smoke test complete."
