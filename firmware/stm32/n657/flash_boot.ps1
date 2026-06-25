<#
.SYNOPSIS
    Sign and flash STM32N657 for boot-from-flash.

.DESCRIPTION
    Flashes the OBB localizer and tip-focus SimCC model needed by the live
    board firmware. The script signs the application, programs the FSBL, and
    flashes both xSPI2 weight blobs.

    Prerequisites:
      - Board in NUCLEO dev/programming mode (see board manual)
      - ST-Link connected via USB
      - STM32CubeProgrammer N6 installed

    After flashing: set BOOT0=0, BOOT1=0 and power-cycle the board.
#>

$ErrorActionPreference = "Stop"

$CubeDir    = "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin"
$SignTool   = "$CubeDir\STM32_SigningTool_CLI.exe"
$ProgCli    = "$CubeDir\STM32_Programmer_CLI.exe"
$ExtLoader  = "$CubeDir\ExternalLoader\MX25UM51245G_STM32N6570-NUCLEO.stldr"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Resolve-Path "$ScriptDir\..\..\.."

# ---------- paths ----------
$FsblBin     = "$ScriptDir\FSBL\Debug\n657_FSBL.bin"
$FsblTrusted = "$ScriptDir\FSBL\Debug\FSBL_trusted.bin"
$ObbRaw      = "$ScriptDir\st_ai_output\packages\obb_box_board_bbox_deploy_candidate\st_ai_output\obb_box_board_bbox_deploy_candidate_atonbuf.xSPI2.raw"
$TipFocusRaw = "$ScriptDir\st_ai_output\packages\tip_focus_v4_112_int8_n6_npu\st_ai_output\network_atonbuf.xSPI2.raw"

if (-not (Test-Path $ObbRaw -PathType Leaf)) {
    $ObbRaw = "$RepoRoot\firmware\stm32\n657\st_ai_output\packages\obb_box_board_bbox_deploy_candidate\st_ai_output\obb_box_board_bbox_deploy_candidate_atonbuf.xSPI2.raw"
}
if (-not (Test-Path $TipFocusRaw -PathType Leaf)) {
    $TipFocusRaw = "$RepoRoot\firmware\stm32\n657\st_ai_output\packages\tip_focus_v4_112_int8_n6_npu\st_ai_output\network_atonbuf.xSPI2.raw"
}

$ObbBin          = "$ScriptDir\Appli\Debug\obb_box_board_bbox_deploy_candidate.bin"
$TipFocusBin     = "$ScriptDir\Appli\Debug\tip_focus_v4_112_int8_n6_npu.bin"
$AppBin          = "$ScriptDir\Appli\Debug\n657_Appli.bin"
$AppSign         = "$ScriptDir\Appli\Debug\n657_Appli_sign_new.bin"
$AppSignTmp      = "$ScriptDir\Appli\Debug\n657_Appli_sign_tmp.bin"
$AppSignFallback = "$ScriptDir\Appli\Debug\n657_Appli_Signed.bin"
$SigReportDir    = "$RepoRoot\tmp\flash_signatures"

# ---------- helpers ----------
function Die ($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}
function Do-Sign ($bin, $type, $out) {
    if (Test-Path $out -PathType Leaf) {
        Remove-Item -LiteralPath $out -Force -ErrorAction SilentlyContinue
    }
    $args = @(
        "-bin", $bin,
        "-nk",
        "-of", "0x80000000",
        "-t", $type,
        "-hv", "2.3",
        "-o", $out,
        "-dump", $out,
        "-align"
    )
    & $SignTool @args
    if ($LASTEXITCODE -ne 0) { Die "Signing failed: $type ($bin)" }
}
function Do-Flash ($bin, $addr, [string]$label) {
    $labelInfo = if ($label) { " ($label)" } else { "" }
    Write-Host "Flashing${labelInfo}: $bin -> 0x$($addr.ToString('X8'))"
    $fileSize = (Get-Item -LiteralPath $bin).Length
    Write-Host "  size = $fileSize bytes"
    & $ProgCli -c port=SWD mode=HOTPLUG -el $ExtLoader -hardRst -w $bin $addr
    if ($LASTEXITCODE -ne 0) { Die "Flash failed${labelInfo}: $bin" }
    Write-Host "  done."
}

# ---------- prerequisites ----------
if (-not (Test-Path $SignTool  -PathType Leaf)) { Die "Signing tool not found: $SignTool" }
if (-not (Test-Path $ProgCli   -PathType Leaf)) { Die "Programmer CLI not found: $ProgCli" }
if (-not (Test-Path $ExtLoader -PathType Leaf)) { Die "External loader not found: $ExtLoader" }
if (-not (Test-Path $FsblBin   -PathType Leaf)) { Die "FSBL binary not found: $FsblBin" }
if (-not (Test-Path $AppBin    -PathType Leaf)) { Die "Application binary not found: $AppBin" }
if (-not (Test-Path $ObbRaw -PathType Leaf)) { Die "OBB model not found: $ObbRaw" }
if (-not (Test-Path $TipFocusRaw -PathType Leaf)) { Die "Tip-focus model not found: $TipFocusRaw" }

if (-not (Test-Path $SigReportDir -PathType Container)) {
    New-Item -ItemType Directory -Path $SigReportDir -Force | Out-Null
}

# ================== Step 1: Sign FSBL ==================
Write-Host "`n=== Step 1: Sign FSBL binary ==="
$FsblTrustedTmp = "$ScriptDir\FSBL\Debug\FSBL_trusted_$(Get-Random).bin"
Do-Sign -bin $FsblBin -type fsbl -out $FsblTrustedTmp
if (Test-Path $FsblTrusted -PathType Leaf) { Remove-Item -LiteralPath $FsblTrusted -Force }
Move-Item -LiteralPath $FsblTrustedTmp -Destination $FsblTrusted -Force
Write-Host "Trusted FSBL: $FsblTrusted"

# ================== Step 2: Flash FSBL ==================
Write-Host "`n=== Step 2: Flash FSBL at 0x70000000 ==="
Do-Flash -bin $FsblTrusted -addr 0x70000000 -label "FSBL"

# ================== Step 3: Flash OBB model ==================
Write-Host "`n=== Step 3: Flash OBB model at 0x70700000 ==="
Copy-Item -LiteralPath $ObbRaw -Destination $ObbBin -Force
Do-Flash -bin $ObbBin -addr 0x70700000 -label "OBB-localizer"

Write-Host "`n=== Step 4: Flash tip-focus SimCC model at 0x70400000 ==="
Copy-Item -LiteralPath $TipFocusRaw -Destination $TipFocusBin -Force
Do-Flash -bin $TipFocusBin -addr 0x70400000 -label "TipFocus-SimCC"

Write-Host "`n=== Step 5: Extract model signatures ==="
python "$RepoRoot\ml\scripts\extract_model_signature.py" "$ObbRaw" > "$SigReportDir\obb_signature.txt"
if ($LASTEXITCODE -ne 0) { Die "OBB signature extraction failed" }
python "$RepoRoot\ml\scripts\extract_model_signature.py" "$TipFocusRaw" > "$SigReportDir\tip_focus_signature.txt"
if ($LASTEXITCODE -ne 0) { Die "Tip-focus signature extraction failed" }
Write-Host "OBB signature: $SigReportDir\obb_signature.txt"
Write-Host "Tip-focus signature: $SigReportDir\tip_focus_signature.txt"

# ================== Step 6: Sign app ==================
Write-Host "`n=== Step 6: Sign application binary ==="
if (Test-Path $AppSignTmp -PathType Leaf) { Remove-Item -LiteralPath $AppSignTmp -Force }
Do-Sign -bin $AppBin -type ssbl -out $AppSignTmp
if (Test-Path $AppSign -PathType Leaf) { Remove-Item -LiteralPath $AppSign -Force }
if (Test-Path $AppSignTmp -PathType Leaf) {
    Move-Item -LiteralPath $AppSignTmp -Destination $AppSign -Force
} elseif (Test-Path $AppSignFallback -PathType Leaf) {
    Copy-Item -LiteralPath $AppSignFallback -Destination $AppSign -Force
}
if (-not (Test-Path $AppSign -PathType Leaf)) {
    Die "Signed application artifact not found. Tried: $AppSignTmp, $AppSignFallback"
}
Write-Host "Signed binary: $AppSign"

# ================== Step 7: Flash app ==================
Write-Host "`n=== Step 7: Flash signed application at 0x70100000 ==="
Do-Flash -bin $AppSign -addr 0x70100000 -label "App"

Write-Host "`n=== Done! ==="
Write-Host "Now set flash-boot mode (BOOT0=0, BOOT1=0) and power-cycle the board."
