<#
.SYNOPSIS
    Sign and flash STM32N657 for boot-from-flash.

.DESCRIPTION
    Flashes the 384x384 grayscale ellipse model followed by the 224x224
    grayscale keypoint U-Net. The script signs the application, programs the
    FSBL, and validates the xSPI2 model slots before flashing both blobs.

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
$EllipseRaw  = "$ScriptDir\st_ai_output\packages\ellipse_iter8_universal_wide_deep_int8_n6_npu\st_ai_output\ellipse_iter8_universal_wide_deep_int8_atonbuf.xSPI2.raw"
$CenterTipRaw = "$ScriptDir\st_ai_output\packages\keypoint_unet_224g_wide_aug_int8_n6_npu\st_ai_output\keypoint_unet_224g_wide_aug_int8_atonbuf.xSPI2.raw"
$SignatureTool = "$ScriptDir\tools\extract_model_signature.py"

if (-not (Test-Path $EllipseRaw -PathType Leaf)) {
    $EllipseRaw = "$RepoRoot\firmware\stm32\n657\st_ai_output\packages\ellipse_iter8_universal_wide_deep_int8_n6_npu\st_ai_output\ellipse_iter8_universal_wide_deep_int8_atonbuf.xSPI2.raw"
}
if (-not (Test-Path $CenterTipRaw -PathType Leaf)) {
    $CenterTipRaw = "$RepoRoot\firmware\stm32\n657\st_ai_output\packages\keypoint_unet_224g_wide_aug_int8_n6_npu\st_ai_output\keypoint_unet_224g_wide_aug_int8_atonbuf.xSPI2.raw"
}

$EllipseBin      = "$ScriptDir\Appli\Debug\ellipse_iter8_universal_wide_deep_int8_n6_npu.bin"
$CenterTipBin    = "$ScriptDir\Appli\Debug\keypoint_unet_224g_wide_aug_int8_n6_npu.bin"
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
    # HWRSTPULSE lets the N657 external loader take control of xSPI2 cleanly;
    # HOTPLUG/-hardRst can leave the running application holding the flash in
    # a state where sector erase fails before any model update starts.
    # CubeProgrammer requires verification to follow the write in the same
    # invocation; a second CLI process rejects -v as an out-of-sequence command.
    & $ProgCli -c port=SWD mode=HWRSTPULSE -el $ExtLoader -w $bin $addr -v
    if ($LASTEXITCODE -ne 0) { Die "Flash failed${labelInfo}: $bin" }
    Write-Host "  write and verification complete."
}
function Check-Range ($name, [uint64]$start, [uint64]$length, [uint64]$slotStart, [uint64]$slotLength) {
    $end = $start + $length
    $slotEnd = $slotStart + $slotLength
    if ($start -lt $slotStart -or $end -gt $slotEnd) {
        Die "$name ($length bytes) exceeds its assigned flash window 0x$($slotStart.ToString('X8'))..0x$(($slotEnd - 1).ToString('X8'))"
    }
}
function Check-No-Overlap ($aName, [uint64]$aStart, [uint64]$aLength, $bName, [uint64]$bStart, [uint64]$bLength) {
    if (($aStart -lt ($bStart + $bLength)) -and ($bStart -lt ($aStart + $aLength))) {
        Die "Flash overlap: $aName and $bName"
    }
}
function Assert-ConsoleSafeApplication ($bin) {
    # Refuse to flash an image that predates the UART/raw-frame fix.  This is
    # deliberately checked before signing so a stale Debug artifact cannot be
    # wrapped in a valid SSBL signature and look like a successful deployment.
    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $bin))
    $ascii = [System.Text.Encoding]::ASCII.GetString($bytes)
    $requiredMarker = "[BOOT] firmware=2026-08-02-baseline-redesign-console-safe"
    $removedMarker = "snapshot-copy progress +64KiB"
    if (-not $ascii.Contains($requiredMarker)) {
        Die "Application is not the console-safe build: missing boot marker '$requiredMarker'"
    }
    if ($ascii.Contains($removedMarker)) {
        Die "Application is stale: found removed raw-copy progress marker '$removedMarker'"
    }
    $hash = (Get-FileHash -LiteralPath $bin -Algorithm SHA256).Hash
    Write-Host "Console-safe application verified: marker present, stale raw-copy marker absent"
    Write-Host "Application SHA256: $hash"
}

# ---------- prerequisites ----------
if (-not (Test-Path $SignTool  -PathType Leaf)) { Die "Signing tool not found: $SignTool" }
if (-not (Test-Path $ProgCli   -PathType Leaf)) { Die "Programmer CLI not found: $ProgCli" }
if (-not (Test-Path $ExtLoader -PathType Leaf)) { Die "External loader not found: $ExtLoader" }
if (-not (Test-Path $FsblBin   -PathType Leaf)) { Die "FSBL binary not found: $FsblBin" }
if (-not (Test-Path $AppBin    -PathType Leaf)) { Die "Application binary not found: $AppBin" }
if (-not (Test-Path $EllipseRaw -PathType Leaf)) { Die "Ellipse model not found: $EllipseRaw" }
if (-not (Test-Path $CenterTipRaw -PathType Leaf)) { Die "Center/tip model not found: $CenterTipRaw" }
Assert-ConsoleSafeApplication $AppBin
if (-not (Test-Path $SignatureTool -PathType Leaf)) {
    $SignatureTool = "$RepoRoot\ml\scripts\extract_model_signature.py"
}
if (-not (Test-Path $SignatureTool -PathType Leaf)) { Die "Signature tool not found: $SignatureTool" }

if (-not (Test-Path $SigReportDir -PathType Container)) {
    New-Item -ItemType Directory -Path $SigReportDir -Force | Out-Null
}

# Reserve non-overlapping 4 MiB xSPI2 slots. The generated blobs are much
# smaller, but checking the complete slot keeps future model replacements
# from colliding with the signed app or with each other.
$FsblStart = [uint64]0x70000000; $FsblWindow = [uint64]0x00100000
$AppStart = [uint64]0x70100000; $AppWindow = [uint64]0x00300000
$EllipseStart = [uint64]0x70400000; $ModelWindow = [uint64]0x00400000
$CenterTipStart = [uint64]0x70800000
Check-Range "FSBL" $FsblStart ([uint64](Get-Item $FsblBin).Length) $FsblStart $FsblWindow
Check-Range "application" $AppStart ([uint64](Get-Item $AppBin).Length) $AppStart $AppWindow
Check-Range "gauge ellipse model" $EllipseStart ([uint64](Get-Item $EllipseRaw).Length) $EllipseStart $ModelWindow
Check-Range "gauge center/tip model" $CenterTipStart ([uint64](Get-Item $CenterTipRaw).Length) $CenterTipStart $ModelWindow
Check-No-Overlap "FSBL" $FsblStart ([uint64](Get-Item $FsblBin).Length) "application" $AppStart ([uint64](Get-Item $AppBin).Length)
Check-No-Overlap "FSBL" $FsblStart ([uint64](Get-Item $FsblBin).Length) "ellipse model" $EllipseStart ([uint64](Get-Item $EllipseRaw).Length)
Check-No-Overlap "FSBL" $FsblStart ([uint64](Get-Item $FsblBin).Length) "center/tip model" $CenterTipStart ([uint64](Get-Item $CenterTipRaw).Length)
Check-No-Overlap "application" $AppStart ([uint64](Get-Item $AppBin).Length) "ellipse model" $EllipseStart ([uint64](Get-Item $EllipseRaw).Length)
Check-No-Overlap "ellipse model" $EllipseStart ([uint64](Get-Item $EllipseRaw).Length) "center/tip model" $CenterTipStart ([uint64](Get-Item $CenterTipRaw).Length)
Write-Host "Flash layout check passed: app 0x70100000, ellipse 0x70400000, center/tip 0x70800000"

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

# ================== Step 3: Flash ellipse model ==================
Write-Host "`n=== Step 3: Flash 384x384 grayscale multiscale ellipse model at 0x70400000 ==="
Copy-Item -LiteralPath $EllipseRaw -Destination $EllipseBin -Force
Do-Flash -bin $EllipseBin -addr 0x70400000 -label "Ellipse-iter8-384-gray"

Write-Host "`n=== Step 4: Flash 224x224 grayscale compact keypoint U-Net at 0x70800000 (no HyperRAM) ==="
Copy-Item -LiteralPath $CenterTipRaw -Destination $CenterTipBin -Force
Do-Flash -bin $CenterTipBin -addr 0x70800000 -label "Keypoint-wide-aug-224-gray-no-hyperram"

Write-Host "`n=== Step 5: Extract model signatures ==="
python "$SignatureTool" "$EllipseRaw" > "$SigReportDir\ellipse_iter8_signature.txt"
if ($LASTEXITCODE -ne 0) { Die "Ellipse signature extraction failed" }
python "$SignatureTool" "$CenterTipRaw" > "$SigReportDir\keypoint_wide_aug_signature.txt"
if ($LASTEXITCODE -ne 0) { Die "Center/tip signature extraction failed" }
Write-Host "Ellipse signature: $SigReportDir\ellipse_iter8_signature.txt"
Write-Host "Center/tip signature: $SigReportDir\keypoint_wide_aug_signature.txt"

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
Check-Range "signed application" $AppStart ([uint64](Get-Item $AppSign).Length) $AppStart $AppWindow
Check-No-Overlap "signed application" $AppStart ([uint64](Get-Item $AppSign).Length) "ellipse model" $EllipseStart ([uint64](Get-Item $EllipseRaw).Length)
Check-No-Overlap "signed application" $AppStart ([uint64](Get-Item $AppSign).Length) "center/tip model" $CenterTipStart ([uint64](Get-Item $CenterTipRaw).Length)
Write-Host "Signed binary: $AppSign"

# ================== Step 7: Flash app ==================
Write-Host "`n=== Step 7: Flash signed application at 0x70100000 ==="
Do-Flash -bin $AppSign -addr 0x70100000 -label "App"

Write-Host "`n=== Done! ==="
Write-Host "Now set flash-boot mode (BOOT0=0, BOOT1=0) and power-cycle the board."
