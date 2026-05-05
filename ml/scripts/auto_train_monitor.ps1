# Auto-Train Monitor for WSL
# Monitors training progress and restarts with new hyperparameters if target not met
# Usage: .\scripts\auto_train_monitor.ps1

param(
    [float]$TargetMAE = 5.0,
    [int]$MaxRuns = 10,
    [int]$CheckIntervalSeconds = 300,
    [int]$StallTimeoutMinutes = 30
)

$ErrorActionPreference = "Continue"

$RunDirBase = "/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/training"
$WslDistro = "Ubuntu-24.04"

# Hyperparameter configurations to try
$Configs = @(
    @{ LR = "5e-5";  Repeat = 4;  Edge = 1.5; Epochs = 80; Batch = 8 },
    @{ LR = "3e-5";  Repeat = 6;  Edge = 2.0; Epochs = 80; Batch = 8 },
    @{ LR = "1e-4";  Repeat = 3;  Edge = 1.5; Epochs = 80; Batch = 8 },
    @{ LR = "5e-5";  Repeat = 8;  Edge = 2.5; Epochs = 80; Batch = 8 },
    @{ LR = "3e-5";  Repeat = 2;  Edge = 1.0; Epochs = 80; Batch = 8 },
    @{ LR = "1e-5";  Repeat = 6;  Edge = 2.0; Epochs = 80; Batch = 8 },
    @{ LR = "5e-5";  Repeat = 4;  Edge = 1.5; Epochs = 100; Batch = 8 },
    @{ LR = "3e-5";  Repeat = 4;  Edge = 1.5; Epochs = 80; Batch = 16 },
    @{ LR = "5e-5";  Repeat = 10; Edge = 2.0; Epochs = 80; Batch = 8 },
    @{ LR = "1e-5";  Repeat = 4;  Edge = 1.5; Epochs = 120; Batch = 8 }
)

function Write-Header($text) {
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
}

function Get-LatestRunDir {
    $cmd = "ls -td ${RunDirBase}/hardcase_interval_* 2>/dev/null | head -1"
    $result = wsl -d $WslDistro -e bash -c $cmd 2>$null
    return $result.Trim()
}

function Get-BestValMAE($runDir) {
    $historyFile = "$runDir/history.json"
    $cmd = @"
import json, sys
try:
    with open('$historyFile') as f:
        h = json.load(f)
    val_mae = h.get('val_gauge_value_mae', [])
    if val_mae:
        print(f'{min(val_mae):.4f}')
    else:
        print('999')
except:
    print('999')
"@
    $result = wsl -d $WslDistro -e bash -c "python3 -c `"$cmd`"" 2>$null
    return [float]($result.Trim())
}

function Test-TrainingComplete($runDir) {
    $cmd = "test -f '$runDir/best_model.keras' && echo 'YES' || echo 'NO'"
    $result = wsl -d $WslDistro -e bash -c $cmd 2>$null
    return ($result.Trim() -eq "YES")
}

function Test-TrainingStalled($runDir, $lastEpoch, $lastTime) {
    $historyFile = "$runDir/history.json"
    $cmd = @"
import json, sys
try:
    with open('$historyFile') as f:
        h = json.load(f)
    epochs = h.get('epoch', [])
    if epochs:
        print(epochs[-1])
    else:
        print('0')
except:
    print('0')
"@
    $result = wsl -d $WslDistro -e bash -c "python3 -c `"$cmd`"" 2>$null
    $currentEpoch = [int]($result.Trim())
    $currentTime = Get-Date
    
    if ($currentEpoch -eq $lastEpoch) {
        $elapsed = ($currentTime - $lastTime).TotalMinutes
        if ($elapsed -gt $StallTimeoutMinutes) {
            return $true, $currentEpoch, $currentTime
        }
    }
    
    return $false, $currentEpoch, $currentTime
}

function Start-TrainingRun($config, $runNum) {
    Write-Header "STARTING RUN ${runNum}/${MaxRuns}"
    Write-Host "Learning Rate: $($config.LR)"
    Write-Host "Hard-case repeat: $($config.Repeat)"
    Write-Host "Edge focus: $($config.Edge)"
    Write-Host "Epochs: $($config.Epochs)"
    Write-Host "Batch size: $($config.Batch)"
    Write-Host ""
    
    $wslCmd = @"
export PATH=/home/rishi_latchmepersad/.local/bin:\$PATH && 
>& cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml && 
poetry run python scripts/train_hardcase_interval.py 
--device gpu 
--epochs $($config.Epochs) 
--batch-size $($config.Batch) 
--learning-rate $($config.LR) 
--hard-case-repeat $($config.Repeat) 
--edge-focus-strength $($config.Edge) 
--no-gpu-memory-growth
"@
    
    # Start training in background via WSL
    $logFile = "${env:TEMP}\train_run_${runNum}.log"
    Write-Host "Logging to: $logFile"
    Write-Host "Training started at: $(Get-Date)"
    Write-Host ""
    
    # Use Start-Process to run in background
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "wsl"
    $psi.Arguments = "-d $WslDistro -e bash -c `"$wslCmd`""
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    
    $process = [System.Diagnostics.Process]::Start($psi)
    
    # Stream output
    $stdout = $process.StandardOutput
    $stderr = $process.StandardError
    
    # Start output readers
    $stdoutJob = Start-Job -ScriptBlock {
        param($reader, $log)
        while ($line = $reader.ReadLine()) {
            Write-Host $line
            Add-Content -Path $log -Value $line
        }
    } -ArgumentList $stdout, $logFile
    
    return $process
}

function Restart-WSL {
    Write-Host "Restarting WSL..." -ForegroundColor Yellow
    wsl --shutdown
    Start-Sleep -Seconds 10
    Write-Host "WSL restarted." -ForegroundColor Green
}

# Main loop
Write-Header "AUTO-TRAIN MONITOR"
Write-Host "Target MAE: ${TargetMAE}C"
Write-Host "Max runs: $MaxRuns"
Write-Host "Check interval: ${CheckIntervalSeconds}s"
Write-Host "Stall timeout: ${StallTimeoutMinutes} min"
Write-Host ""

$bestMAE = 999.0
$bestRun = $null

for ($runNum = 1; $runNum -le $MaxRuns; $runNum++) {
    $config = $Configs[($runNum - 1) % $Configs.Count]
    
    # Restart WSL before each run (per AGENTS.md instructions)
    if ($runNum -gt 1) {
        Restart-WSL
    }
    
    # Start training
    $process = Start-TrainingRun $config $runNum
    
    # Monitor
    $lastEpoch = 0
    $lastTime = Get-Date
    $monitoring = $true
    
    while ($monitoring -and -not $process.HasExited) {
        Start-Sleep -Seconds $CheckIntervalSeconds
        
        $runDir = Get-LatestRunDir
        if (-not $runDir) {
            Write-Host "$(Get-Date -Format 'HH:mm:ss'): Waiting for run directory..."
            continue
        }
        
        # Check for stall
        $stalled, $currentEpoch, $lastTime = Test-TrainingStalled $runDir $lastEpoch $lastTime
        if ($stalled) {
            Write-Host "$(Get-Date -Format 'HH:mm:ss'): TRAINING STALLED at epoch $currentEpoch" -ForegroundColor Red
            Write-Host "Killing process and restarting..."
            $process.Kill()
            break
        }
        
        $lastEpoch = $currentEpoch
        
        # Check if complete
        if (Test-TrainingComplete $runDir) {
            $mae = Get-BestValMAE $runDir
            Write-Host ""
            Write-Host "$(Get-Date -Format 'HH:mm:ss'): Training complete!" -ForegroundColor Green
            Write-Host "Run: $runDir"
            Write-Host "Best val MAE: ${mae}C"
            
            if ($mae -lt $bestMAE) {
                $bestMAE = $mae
                $bestRun = $runDir
            }
            
            if ($mae -lt $TargetMAE) {
                Write-Host ""
                Write-Header "TARGET ACHIEVED!"
                Write-Host "Run: $runNum" -ForegroundColor Green
                Write-Host "MAE: ${mae}C (target: ${TargetMAE}C)" -ForegroundColor Green
                Write-Host "Model: $runDir" -ForegroundColor Green
                Write-Host ""
                Write-Host "Best overall: Run with MAE ${bestMAE}C" -ForegroundColor Cyan
                exit 0
            } else {
                Write-Host "Target not achieved. Best MAE: ${mae}C (target: ${TargetMAE}C)" -ForegroundColor Yellow
                $monitoring = $false
            }
        } else {
            Write-Host "$(Get-Date -Format 'HH:mm:ss'): Monitoring... Epoch $lastEpoch"
        }
    }
    
    # Clean up
    if (-not $process.HasExited) {
        $process.Kill()
    }
    
    Write-Host ""
    Write-Host "Run $runNum complete. Best so far: ${bestMAE}C" -ForegroundColor Cyan
    Write-Host ""
}

Write-Header "MAX RUNS REACHED"
Write-Host "Best MAE achieved: ${bestMAE}C" -ForegroundColor Yellow
Write-Host "Best run: $bestRun" -ForegroundColor Yellow
Write-Host "Target was: ${TargetMAE}C" -ForegroundColor Yellow
Write-Host ""
Write-Host "Consider:"
Write-Host "- Increasing epochs"
Write-Host "- Trying different architecture"
Write-Host "- Adding more data augmentation"
Write-Host "- Using a larger model"
