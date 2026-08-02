@echo off
REM ============================================================
REM  flash_boot.bat  --  Sign and flash STM32N657 for boot-from-flash
REM
REM  Usage: flash_boot.bat
REM  Prerequisites:
REM    - Board in the NUCLEO dev/programming mode described by the board
REM      manual. Do not rely on the older JP3 wording in this file.
REM    - ST-Link connected via USB
REM    - STM32CubeProgrammer N6 installed
REM
REM  After flashing: set flash-boot mode (BOOT0=0, BOOT1=0) and power-cycle
REM  the board.
REM ============================================================

set "CUBE=C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin"
set "SIGN=%CUBE%\STM32_SigningTool_CLI.exe"
set "PROG=%CUBE%\STM32_Programmer_CLI.exe"
set "ELDR=%CUBE%\ExternalLoader\MX25UM51245G_STM32N6570-NUCLEO.stldr"

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\..\"
set "FSBL_BIN=%SCRIPT_DIR%FSBL\Debug\n657_FSBL.bin"
set "FSBL_TRUSTED=%SCRIPT_DIR%FSBL\Debug\FSBL_trusted.bin"
set "ELLIPSE_RAW=%SCRIPT_DIR%st_ai_output\packages\ellipse_iter8_universal_wide_deep_int8_n6_npu\st_ai_output\ellipse_iter8_universal_wide_deep_int8_atonbuf.xSPI2.raw"
set "CENTER_TIP_RAW=%SCRIPT_DIR%st_ai_output\packages\keypoint_unet_224g_wide_aug_int8_n6_npu\st_ai_output\keypoint_unet_224g_wide_aug_int8_atonbuf.xSPI2.raw"

if not exist "%ELLIPSE_RAW%" set "ELLIPSE_RAW=%REPO_ROOT%firmware\stm32\n657\st_ai_output\packages\ellipse_iter8_universal_wide_deep_int8_n6_npu\st_ai_output\ellipse_iter8_universal_wide_deep_int8_atonbuf.xSPI2.raw"
if not exist "%CENTER_TIP_RAW%" set "CENTER_TIP_RAW=%REPO_ROOT%firmware\stm32\n657\st_ai_output\packages\keypoint_unet_224g_wide_aug_int8_n6_npu\st_ai_output\keypoint_unet_224g_wide_aug_int8_atonbuf.xSPI2.raw"

REM CubeProgrammer v2.21 does not accept .raw extension with -w; stage as .bin
set "ELLIPSE_BIN=%SCRIPT_DIR%Appli\Debug\ellipse_iter8_universal_wide_deep_int8_n6_npu.bin"
set "CENTER_TIP_BIN=%SCRIPT_DIR%Appli\Debug\keypoint_unet_224g_wide_aug_int8_n6_npu.bin"

set "APP_BIN=%SCRIPT_DIR%Appli\Debug\n657_Appli.bin"
set "APP_SIGN=%SCRIPT_DIR%Appli\Debug\n657_Appli_sign_new.bin"
set "APP_SIGN_TMP=%SCRIPT_DIR%Appli\Debug\n657_Appli_sign_tmp.bin"
set "APP_SIGN_FALLBACK=%SCRIPT_DIR%Appli\Debug\n657_Appli_Signed.bin"
set "SIG_REPORT_DIR=%REPO_ROOT%tmp\flash_signatures"
set "FLASH_MODEL=1"
set "FLASH_APP=1"

if not exist "%SIG_REPORT_DIR%" (
    mkdir "%SIG_REPORT_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create signature report directory: "%SIG_REPORT_DIR%"
        exit /b 1
    )
)

if not exist "%SIGN%" (
    echo ERROR: Signing tool not found: "%SIGN%"
    exit /b 1
)
if not exist "%PROG%" (
    echo ERROR: Programmer CLI not found: "%PROG%"
    exit /b 1
)
if not exist "%ELDR%" (
    echo ERROR: External loader not found: "%ELDR%"
    exit /b 1
)
if not exist "%FSBL_BIN%" (
    echo ERROR: FSBL binary not found: "%FSBL_BIN%"
    exit /b 1
)
if "%FLASH_APP%"=="1" if not exist "%APP_BIN%" (
    echo ERROR: Application binary not found: "%APP_BIN%"
    exit /b 1
)
if "%FLASH_APP%"=="1" (
    REM Refuse to sign/flash an older image that can still emit raw frame bytes.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$b=[IO.File]::ReadAllBytes('%APP_BIN%');$s=[Text.Encoding]::ASCII.GetString($b);if(-not $s.Contains('[BOOT] firmware=2026-08-02-baseline-redesign-console-safe')){Write-Error 'Application is not the console-safe build';exit 1};if($s.Contains('snapshot-copy progress +64KiB')){Write-Error 'Application contains the removed raw-copy progress marker';exit 1};Write-Host ('Console-safe application verified. SHA256=' + (Get-FileHash -LiteralPath '%APP_BIN%' -Algorithm SHA256).Hash)"
    if errorlevel 1 (
        echo ERROR: Refusing to flash stale or non-console-safe application.
        exit /b 1
    )
)
if "%FLASH_MODEL%"=="1" if not exist "%ELLIPSE_RAW%" (
    echo ERROR: Ellipse model not found: "%ELLIPSE_RAW%"
    exit /b 1
)
if "%FLASH_MODEL%"=="1" if not exist "%CENTER_TIP_RAW%" (
    echo ERROR: Center/tip model not found: "%CENTER_TIP_RAW%"
    exit /b 1
)

echo.
echo === Step 2: Sign FSBL binary ===
set "FSBL_TRUSTED_TMP=%SCRIPT_DIR%FSBL\Debug\FSBL_trusted_%RANDOM%.bin"
"%SIGN%" -bin "%FSBL_BIN%" -nk -of 0x80000000 -t fsbl -hv 2.3 -o "%FSBL_TRUSTED_TMP%" -dump "%FSBL_TRUSTED_TMP%" -align
if errorlevel 1 (
    echo ERROR: FSBL signing failed.
    exit /b 1
)
if exist "%FSBL_TRUSTED%" (
    del /f /q "%FSBL_TRUSTED%"
)
move /y "%FSBL_TRUSTED_TMP%" "%FSBL_TRUSTED%" >nul
echo Trusted FSBL: %FSBL_TRUSTED%

echo.
echo === Step 3: Flash FSBL at 0x70000000 ===
REM HWRSTPULSE gives the external loader control of xSPI2 before erase/write;
REM verify in the same invocation so CubeProgrammer does not reject a second
REM standalone verification command.
"%PROG%" -c port=SWD mode=HWRSTPULSE -el "%ELDR%" -w "%FSBL_TRUSTED%" 0x70000000 -v
if errorlevel 1 (
    echo ERROR: FSBL flash failed.
    exit /b 1
)

echo.
if "%FLASH_MODEL%"=="1" (
    echo === Step 4a: Flash 384x384 grayscale ellipse v2 model at 0x70400000 ===
    echo Ellipse source: "%ELLIPSE_RAW%"
    for %%I in ("%ELLIPSE_RAW%") do echo Ellipse source size: %%~zI bytes
    copy /y "%ELLIPSE_RAW%" "%ELLIPSE_BIN%" >nul
    if errorlevel 1 (
        echo ERROR: Could not stage ellipse model as .bin.
        exit /b 1
    )
    "%PROG%" -c port=SWD mode=HWRSTPULSE -el "%ELDR%" -w "%ELLIPSE_BIN%" 0x70400000 -v
    if errorlevel 1 (
        echo ERROR: Ellipse model flash failed.
        exit /b 1
    )
    echo Ellipse model flashed at 0x70400000.

    echo === Step 4b: Flash 224x224 grayscale wide-aug keypoint U-Net at 0x70800000 ===
    echo Center/tip source: "%CENTER_TIP_RAW%"
    for %%I in ("%CENTER_TIP_RAW%") do echo Center/tip source size: %%~zI bytes
    copy /y "%CENTER_TIP_RAW%" "%CENTER_TIP_BIN%" >nul
    if errorlevel 1 (
        echo ERROR: Could not stage center/tip model as .bin.
        exit /b 1
    )
    "%PROG%" -c port=SWD mode=HWRSTPULSE -el "%ELDR%" -w "%CENTER_TIP_BIN%" 0x70800000 -v
    if errorlevel 1 (
        echo ERROR: Center/tip model flash failed.
        exit /b 1
    )
    echo Center/tip model flashed at 0x70800000.
)

if "%FLASH_MODEL%"=="1" (
    echo.
    echo === Step 4d: Extract model signatures for firmware update ===
    python "%SCRIPT_DIR%tools\extract_model_signature.py" "%ELLIPSE_RAW%" > "%SIG_REPORT_DIR%\gauge_ellipse_v2_signature.txt"
    if errorlevel 1 (
        echo ERROR: Ellipse signature extraction failed.
        exit /b 1
    )
    echo Ellipse signature report: "%SIG_REPORT_DIR%\gauge_ellipse_v2_signature.txt"
    python "%SCRIPT_DIR%tools\extract_model_signature.py" "%CENTER_TIP_RAW%" > "%SIG_REPORT_DIR%\gauge_keypoint_unet_wide_aug_signature.txt"
    if errorlevel 1 (
        echo ERROR: Center/tip signature extraction failed.
        exit /b 1
    )
    echo Center/tip signature report: "%SIG_REPORT_DIR%\gauge_keypoint_unet_wide_aug_signature.txt"
)

if "%FLASH_APP%"=="1" (
    echo.
    echo === Step 5: Sign application binary ===
    if exist "%APP_SIGN_TMP%" del /f /q "%APP_SIGN_TMP%"
    "%SIGN%" -bin "%APP_BIN%" -nk -of 0x80000000 -t ssbl -hv 2.3 -o "%APP_SIGN_TMP%" -align
    if errorlevel 1 (
        echo ERROR: Signing failed.
        exit /b 1
    )
    if exist "%APP_SIGN%" (
        del /f /q "%APP_SIGN%"
    )
    if exist "%APP_SIGN_TMP%" (
        move /y "%APP_SIGN_TMP%" "%APP_SIGN%" >nul
    ) else if exist "%APP_SIGN_FALLBACK%" (
        copy /y "%APP_SIGN_FALLBACK%" "%APP_SIGN%" >nul
    )
    if not exist "%APP_SIGN%" (
        echo ERROR: Signed application artifact not found.
        echo Tried:
        echo   - %APP_SIGN_TMP%
        echo   - %APP_SIGN_FALLBACK%
        exit /b 1
    )
    echo Signed binary: %APP_SIGN%

    echo.
    echo === Step 6: Flash signed application at 0x70100000 ===
    "%PROG%" -c port=SWD mode=HWRSTPULSE -el "%ELDR%" -w "%APP_SIGN%" 0x70100000 -v
    if errorlevel 1 (
        echo ERROR: Application flash failed.
        exit /b 1
    )
) else (
    echo.
    echo === Step 5: Skipping signed application flash for smoke test ===
    echo Set FLASH_APP=1 if you want to add the app back after the LED boot test passes.
)

echo.
echo === Done! ===
echo Now set flash-boot mode (BOOT0=0, BOOT1=0) and power-cycle the board.
echo.
