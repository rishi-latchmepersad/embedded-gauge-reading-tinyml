$ErrorActionPreference = "Stop"

$REPO_ROOT = "D:\Projects\embedded-gauge-reading-tinyml\ml"
$LOG_DIR = Join-Path $REPO_ROOT "artifacts\training_logs"
$LOG_FILE = Join-Path $LOG_DIR "mobilenetv2_source_crop_corner_v1.log"
$BASE_MODEL_SRC = Join-Path $REPO_ROOT "artifacts\training\mobilenetv2_rectifier_hardcase_finetune_v3\model.keras"
$BASE_MODEL_LOCAL = Join-Path $REPO_ROOT "..\tmp\source_crop_corner_v1.model.keras"
$BOXES_CSV = Join-Path $REPO_ROOT "data\rectified_crop_boxes_v5_all.csv"
$HARD_CASE_MANIFEST = Join-Path $REPO_ROOT "data\hard_cases_plus_board30_valid_with_new6.csv"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $BASE_MODEL_LOCAL) | Out-Null

Copy-Item -Path $BASE_MODEL_SRC -Destination $BASE_MODEL_LOCAL -Force

Set-Location $REPO_ROOT

Write-Host "[WRAPPER] Starting source crop-corner heatmap fine-tune v1."
Write-Host "[WRAPPER] Base model: $BASE_MODEL_LOCAL"
Write-Host "[WRAPPER] Boxes CSV:  $BOXES_CSV"
Write-Host "[WRAPPER] Hard set:   $HARD_CASE_MANIFEST"
Write-Host "[WRAPPER] Log file:   $LOG_FILE"

$pythonArgs = @(
    "scripts\run_training.py",
    "--model-family", "mobilenet_v2_source_crop_corner",
    "--device", "gpu",
    "--no-gpu-memory-growth",
    "--batch-size", "4",
    "--epochs", "50",
    "--learning-rate", "1e-5",
    "--init-model", "$BASE_MODEL_LOCAL",
    "--val-manifest", "$HARD_CASE_MANIFEST",
    "--hard-case-manifest", "$HARD_CASE_MANIFEST",
    "--hard-case-repeat", "4",
    "--edge-focus-strength", "1.0",
    "--precomputed-crop-boxes", "$BOXES_CSV",
    "--run-name", "mobilenetv2_source_crop_corner_v1"
)

& cmd /c "poetry run python -u $pythonArgs 2>&1" | Tee-Object -FilePath $LOG_FILE -Append
