# Auto-Train Loop for Windows CPU
# Runs training, evaluates, and restarts with new hyperparameters until target is met
# Usage: .\scripts\auto_train_loop_windows.ps1

param(
    [float]$TargetMAE = 5.0,
    [int]$MaxRuns = 10,
    [int]$Epochs = 80,
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Continue"
$RunDirBase = "D:\Projects\embedded-gauge-reading-tinyml\ml\artifacts\training"
$ScriptDir = "D:\Projects\embedded-gauge-reading-tinyml\ml\scripts"
$SrcDir = "D:\Projects\embedded-gauge-reading-tinyml\ml\src"

# Hyperparameter configurations to try
$Configs = @(
    @{ LR = "5e-5";  Repeat = 4;  Edge = 1.5;  Note = "Conservative" },
    @{ LR = "3e-5";  Repeat = 6;  Edge = 2.0;  Note = "More hard cases" },
    @{ LR = "1e-4";  Repeat = 3;  Edge = 1.5;  Note = "Higher LR" },
    @{ LR = "5e-5";  Repeat = 8;  Edge = 2.5;  Note = "Aggressive hard cases" },
    @{ LR = "3e-5";  Repeat = 2;  Edge = 1.0;  Note = "Less hard cases" },
    @{ LR = "1e-5";  Repeat = 6;  Edge = 2.0;  Note = "Very low LR" },
    @{ LR = "5e-5";  Repeat = 4;  Edge = 1.5;  Note = "More epochs"; Epochs = 120 },
    @{ LR = "3e-5";  Repeat = 4;  Edge = 1.5;  Note = "Larger batch"; Batch = 16 },
    @{ LR = "5e-5";  Repeat = 10; Edge = 2.0;  Note = "Max hard cases" },
    @{ LR = "1e-5";  Repeat = 4;  Edge = 1.5;  Note = "Very conservative"; Epochs = 120 }
)

function Write-Header($text) {
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
}

function Get-LatestRunDir {
    $dirs = Get-ChildItem -Path $RunDirBase -Directory -Filter "hardcase_interval_*" | 
            Sort-Object LastWriteTime -Descending
    if ($dirs) {
        return $dirs[0].FullName
    }
    return $null
}

function Get-BestValMAE($runDir) {
    $historyFile = Join-Path $runDir "history.json"
    if (Test-Path $historyFile) {
        try {
            $h = Get-Content $historyFile | ConvertFrom-Json
            $valMAE = $h.val_gauge_value_mae
            if ($valMAE -and $valMAE.Count -gt 0) {
                return ($valMAE | Measure-Object -Minimum).Minimum
            }
        } catch {
            return 999
        }
    }
    return 999
}

function Test-TrainingComplete($runDir) {
    return (Test-Path (Join-Path $runDir "best_model.keras"))
}

function Invoke-Training($config, $runNum) {
    $epochs = if ($config.Epochs) { $config.Epochs } else { $Epochs }
    $batch = if ($config.Batch) { $config.Batch } else { $BatchSize }
    
    Write-Header "STARTING RUN ${runNum}/${MaxRuns}: $($config.Note)"
    Write-Host "LR: $($config.LR), Hard-case repeat: $($config.Repeat), Edge: $($config.Edge)"
    Write-Host "Epochs: $epochs, Batch: $batch"
    Write-Host ""
    
    $cmd = "python `"$ScriptDir\train_hardcase_interval.py`" --device cpu --epochs $epochs --batch-size $batch --learning-rate $($config.LR) --hard-case-repeat $($config.Repeat) --edge-focus-strength $($config.Edge)"
    
    Write-Host "Command: $cmd"
    Write-Host ""
    
    # Run training
    $logFile = "${env:TEMP}\train_run_${runNum}_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    Write-Host "Log file: $logFile"
    
    Invoke-Expression $cmd | Tee-Object -FilePath $logFile
    
    return $LASTEXITCODE
}

function Invoke-HardCaseEval($runDir) {
    Write-Host ""
    Write-Host "Evaluating on hard cases..."
    
    # Create eval script
    $evalScript = @"
import sys
sys.path.insert(0, r"$SrcDir")

import json
import numpy as np
from pathlib import Path
import tensorflow as tf

from embedded_gauge_reading_tinyml.training import load_gauge_specs, load_dataset, _build_training_examples
from embedded_gauge_reading_tinyml.presets import LABELLED_DIR, RAW_DIR

run_dir = r"$runDir"
manifest_path = r"D:\Projects\embedded-gauge-reading-tinyml\ml\data\hard_cases_plus_board30_valid_with_new6.csv"

# Load model
model_path = Path(run_dir) / "best_model.keras"
if not model_path.exists():
    model_path = Path(run_dir) / "final_model.keras"

print(f"Loading model from {model_path}")
model = tf.keras.models.load_model(str(model_path))

# Load hard case paths
hard_paths = set()
with open(manifest_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("image_path"):
            hard_paths.add(line.split(",")[0])

print(f"Hard cases to evaluate: {len(hard_paths)}")

# Load specs and dataset
specs = load_gauge_specs()
spec = specs["littlegood_home_temp_gauge_c"]
samples = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
examples, _ = _build_training_examples(samples, spec, image_height=224, image_width=224)

# Filter to hard cases
hard_examples = [e for e in examples if e.image_path in hard_paths]
print(f"Found {len(hard_examples)} hard examples in dataset")

# Evaluate
errors = []
for ex in hard_examples:
    img = tf.io.read_file(ex.image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, 0)
    
    pred = model(img, training=False)
    if isinstance(pred, dict):
        pred_val = float(pred["gauge_value"][0])
    else:
        pred_val = float(pred[0])
    
    error = abs(pred_val - ex.gauge_value)
    errors.append(error)

errors = np.array(errors)
mae = np.mean(errors)
max_err = np.max(errors)
pct_under_5 = np.mean(errors < 5.0) * 100

print(f"\n{'='*50}")
print(f"HARD CASE EVALUATION RESULTS")
print(f"{'='*50}")
print(f"MAE: {mae:.2f}C")
print(f"Max error: {max_err:.2f}C")
print(f"% under 5C: {pct_under_5:.1f}%")
print(f"{'='*50}")

# Save results
results = {
    "run_dir": str(run_dir),
    "mae": float(mae),
    "max_error": float(max_err),
    "pct_under_5c": float(pct_under_5),
    "num_evaluated": len(errors),
    "errors": [float(e) for e in errors]
}

with open(Path(run_dir) / "hard_case_eval.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {Path(run_dir) / 'hard_case_eval.json'}")

# Return exit code based on target
if mae < 5.0 and pct_under_5 > 80:
    print("\nSUCCESS: Target achieved!")
    sys.exit(0)
else:
    print("\nTarget not achieved.")
    sys.exit(1)
"@
    
    $evalFile = "${env:TEMP}\eval_hard_cases_$(Get-Date -Format 'yyyyMMdd_HHmmss').py"
    $evalScript | Out-File -FilePath $evalFile -Encoding UTF8
    
    try {
        python $evalFile
        return $LASTEXITCODE
    } finally {
        Remove-Item $evalFile -ErrorAction SilentlyContinue
    }
}

# Main loop
Write-Header "AUTO-TRAIN LOOP (Windows CPU)"
Write-Host "Target MAE: ${TargetMAE}C"
Write-Host "Max runs: $MaxRuns"
Write-Host ""

$bestMAE = 999.0
$bestRun = $null

for ($runNum = 1; $runNum -le $MaxRuns; $runNum++) {
    $config = $Configs[($runNum - 1) % $Configs.Count]
    
    # Run training
    Invoke-Training $config $runNum
    
    # Find the run directory
    $runDir = Get-LatestRunDir
    if (-not $runDir) {
        Write-Host "ERROR: No run directory found!" -ForegroundColor Red
        continue
    }
    
    Write-Host ""
    Write-Host "Run directory: $runDir"
    
    # Wait a bit for files to be written
    Start-Sleep -Seconds 5
    
    # Check if training completed
    if (-not (Test-TrainingComplete $runDir)) {
        Write-Host "WARNING: Training may not have completed successfully" -ForegroundColor Yellow
    }
    
    # Get best validation MAE
    $valMAE = Get-BestValMAE $runDir
    Write-Host "Best validation MAE: ${valMAE}C"
    
    if ($valMAE -lt $bestMAE) {
        $bestMAE = $valMAE
        $bestRun = $runDir
    }
    
    # Evaluate on hard cases
    $evalResult = Invoke-HardCaseEval $runDir
    
    if ($evalResult -eq 0) {
        Write-Header "TARGET ACHIEVED ON RUN ${runNum}!"
        Write-Host "Model: $runDir" -ForegroundColor Green
        Write-Host "MAE: ${valMAE}C (target: ${TargetMAE}C)" -ForegroundColor Green
        Write-Host ""
        Write-Host "Best overall: Run with MAE ${bestMAE}C" -ForegroundColor Cyan
        exit 0
    }
    
    Write-Host ""
    Write-Host "Run $runNum did not meet target. Best so far: ${bestMAE}C" -ForegroundColor Yellow
    Write-Host "Trying next configuration..."
    Write-Host ""
}

Write-Header "MAX RUNS REACHED"
Write-Host "Best MAE achieved: ${bestMAE}C" -ForegroundColor Yellow
Write-Host "Best run: $bestRun" -ForegroundColor Yellow
Write-Host "Target was: ${TargetMAE}C" -ForegroundColor Yellow
Write-Host ""
Write-Host "Consider:"
Write-Host "- More data augmentation"
Write-Host "- Different model architecture"
Write-Host "- Longer training (more epochs)"
Write-Host "- Larger model (alpha > 1.0)"
