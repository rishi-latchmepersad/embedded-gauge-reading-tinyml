# Build Canonical Manifest and Create Splits
# This script builds the canonical training manifest and creates deterministic splits

$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\embedded-gauge-reading-tinyml"
$MLRoot = Join-Path $ProjectRoot "ml"
$ScriptsDir = Join-Path $MLRoot "scripts"
$DataDir = Join-Path $MLRoot "data"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Canonical Manifest" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Repository root: $ProjectRoot"
Write-Host "ML root: $MLRoot"
Write-Host "Data dir: $DataDir"
Write-Host ""

# Check if poetry is available
Write-Host "Checking Python environment..." -ForegroundColor Yellow
try {
    # Try to find poetry in common locations
    $poetryPaths = @(
        "$env:USERPROFILE\.local\bin\poetry",
        "$env:USERPROFILE\AppData\Roaming\Python\Python311\Scripts\poetry.exe",
        "$env:APPDATA\Python\Python311\Scripts\poetry.exe"
    )
    
    $poetryPath = $null
    foreach ($path in $poetryPaths) {
        if (Test-Path $path) {
            $poetryPath = $path
            break
        }
    }
    
    if ($poetryPath) {
        Write-Host "Found poetry at: $poetryPath" -ForegroundColor Green
        $pythonCmd = "& `"$poetryPath`" run python"
    } else {
        # Fall back to system python
        Write-Host "Poetry not found, using system Python..." -ForegroundColor Yellow
        $pythonCmd = "python"
    }
    
    # Test python
    & $pythonCmd --version
} catch {
    Write-Host "Error: Python environment not found. Please install dependencies first." -ForegroundColor Red
    Write-Host "Run: poetry install" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Step 1: Building canonical manifest..." -ForegroundColor Cyan

# Build canonical manifest
$manifestArgs = @(
    "--data-dir", $DataDir,
    "--repo-root", $ProjectRoot,
    "--output", (Join-Path $DataDir "canonical_manifest_v1.csv"),
    "--conflicts-output", (Join-Path $DataDir "canonical_manifest_conflicts_v1.csv"),
    "--conflict-threshold", "1.0",
    "--verbose"
)

try {
    & $pythonCmd (Join-Path $ScriptsDir "build_canonical_manifest.py") @manifestArgs
    Write-Host ""
    Write-Host "✓ Canonical manifest built successfully!" -ForegroundColor Green
} catch {
    Write-Host "Error building canonical manifest: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Creating deterministic splits..." -ForegroundColor Cyan

# Create splits
$splitArgs = @(
    "--manifest", (Join-Path $DataDir "canonical_manifest_v1.csv"),
    "--output-dir", (Join-Path $DataDir "splits"),
    "--train-ratio", "0.70",
    "--val-ratio", "0.15",
    "--test-ratio", "0.15",
    "--random-state", "42",
    "--bin-size", "5.0",
    "--verbose"
)

try {
    & $pythonCmd (Join-Path $ScriptsDir "create_splits.py") @splitArgs
    Write-Host ""
    Write-Host "✓ Splits created successfully!" -ForegroundColor Green
} catch {
    Write-Host "Error creating splits: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Show file sizes
$canonicalPath = Join-Path $DataDir "canonical_manifest_v1.csv"
$conflictsPath = Join-Path $DataDir "canonical_manifest_conflicts_v1.csv"
$splitDir = Join-Path $DataDir "splits"

if (Test-Path $canonicalPath) {
    $lines = (Get-Content $canonicalPath).Count
    Write-Host "Canonical manifest: $lines rows" -ForegroundColor Green
}

if (Test-Path $conflictsPath) {
    $lines = (Get-Content $conflictsPath).Count
    Write-Host "Conflicts: $lines rows" -ForegroundColor Yellow
}

if (Test-Path $splitDir) {
    $trainLines = (Get-Content (Join-Path $splitDir "canonical_split_v1_train.csv")).Count
    $valLines = (Get-Content (Join-Path $splitDir "canonical_split_v1_val.csv")).Count
    $testLines = (Get-Content (Join-Path $splitDir "canonical_split_v1_test.csv")).Count
    Write-Host "Train split: $trainLines rows" -ForegroundColor Green
    Write-Host "Val split: $valLines rows" -ForegroundColor Green
    Write-Host "Test split: $testLines rows" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done! Ready for training." -ForegroundColor Green
