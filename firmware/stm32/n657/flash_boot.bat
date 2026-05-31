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

set SCRIPT_DIR=%~dp0
set "REPO_ROOT=%SCRIPT_DIR%..\..\..\"
set "FSBL_BIN=%SCRIPT_DIR%FSBL\Debug\n657_FSBL.bin"
set "FSBL_TRUSTED=%SCRIPT_DIR%FSBL\Debug\FSBL_trusted.bin"
REM Prod v0.8: the production pipeline needs only scalar + OBB blobs.
REM Legacy blobs (rectifier, source-crop-box, tip-focus) are optional and
REM only flashed when present; missing ones produce a warning, not a hard error.
REM Prefer firmware-local model artifacts first; repo-root artifacts may be stale.
set "SCALAR_RAW=%SCRIPT_DIR%st_ai_output\packages\mobilenetv2_rectified_scalar_finetune_v2\st_ai_output\scalar_full_finetune_from_best_piecewise_calibrated_int8_atonbuf.xSPI2.raw"
set "OBB_RAW=%SCRIPT_DIR%st_ai_output\packages\prod_model_v0.3_obb_int8\st_ai_output\mobilenetv2_obb_longterm_atonbuf.xSPI2.raw"
if not exist "%SCALAR_RAW%" set "SCALAR_RAW=%REPO_ROOT%firmware\stm32\n657\st_ai_output\packages\mobilenetv2_rectified_scalar_finetune_v2\st_ai_output\scalar_full_finetune_from_best_piecewise_calibrated_int8_atonbuf.xSPI2.raw"
if not exist "%OBB_RAW%" set "OBB_RAW=%REPO_ROOT%firmware\stm32\n657\st_ai_output\packages\prod_model_v0.3_obb_int8\st_ai_output\mobilenetv2_obb_longterm_atonbuf.xSPI2.raw"
REM CubeProgrammer v2.21 does not accept .raw extension with -w; stage as .bin
set "SCALAR_BIN=%SCRIPT_DIR%Appli\Debug\scalar_model_stage.bin"
set "OBB_BIN=%SCRIPT_DIR%Appli\Debug\obb_model_stage.bin"
REM Optional legacy blobs (not required for Prod v0.8)
set "RECTIFIER_RAW=%SCRIPT_DIR%st_ai_output\atonbuf.rectifier.xSPI2.raw"
if not exist "%RECTIFIER_RAW%" set "RECTIFIER_RAW=%REPO_ROOT%st_ai_output\atonbuf.rectifier.xSPI2.raw"
set "RECTIFIER_BIN=%SCRIPT_DIR%Appli\Debug\rectifier_model_stage.bin"
set "SOURCE_CROP_BOX_RAW=%SCRIPT_DIR%st_ai_output\atonbuf.source_crop_box.xSPI2.raw"
if not exist "%SOURCE_CROP_BOX_RAW%" set "SOURCE_CROP_BOX_RAW=%REPO_ROOT%st_ai_output\atonbuf.source_crop_box.xSPI2.raw"
set "SOURCE_CROP_BOX_BIN=%SCRIPT_DIR%Appli\Debug\source_crop_box_model_stage.bin"
set "TIP_FOCUS_WEIGHTS_RAW=%SCRIPT_DIR%st_ai_output\packages\tip_focus_v4_112_int8_n6_npu\st_ai_output\network_atonbuf.xSPI2.raw"
if not exist "%TIP_FOCUS_WEIGHTS_RAW%" set "TIP_FOCUS_WEIGHTS_RAW=%REPO_ROOT%st_ai_output\packages\tip_focus_v4_112_int8_n6_npu\st_ai_output\network_atonbuf.xSPI2.raw"
set "TIP_FOCUS_WEIGHTS_BIN=%SCRIPT_DIR%Appli\Debug\tip_focus_weights_stage.bin"
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
if "%FLASH_MODEL%"=="1" if not exist "%SCALAR_RAW%" (
    echo ERROR: Scalar model not found: "%SCALAR_RAW%"
    exit /b 1
)
if "%FLASH_MODEL%"=="1" if not exist "%OBB_RAW%" (
    echo ERROR: OBB model not found: "%OBB_RAW%"
    exit /b 1
)
REM Optional legacy blobs — warn but do not hard-fail
if "%FLASH_MODEL%"=="1" if not exist "%RECTIFIER_RAW%" (
    echo WARNING: Rectifier model not found: "%RECTIFIER_RAW%"
    echo Prod v0.8 does not require the rectifier; skipping.
)
if "%FLASH_MODEL%"=="1" if not exist "%SOURCE_CROP_BOX_RAW%" (
    echo WARNING: Source-crop-box model not found: "%SOURCE_CROP_BOX_RAW%"
    echo Prod v0.8 does not require source-crop-box; skipping.
)
if "%FLASH_MODEL%"=="1" if not exist "%TIP_FOCUS_WEIGHTS_RAW%" (
    echo WARNING: Tip-focus weights not found: "%TIP_FOCUS_WEIGHTS_RAW%"
    echo Prod v0.8 disables the tip-focus geometry stage; skipping is safe.
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
"%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%FSBL_TRUSTED%" 0x70000000
if errorlevel 1 (
    echo ERROR: FSBL flash failed.
    exit /b 1
)

echo.
if "%FLASH_MODEL%"=="1" (
    echo === Step 4a: Flash scalar model at 0x70200000 (required) ===
    echo Scalar source: "%SCALAR_RAW%"
    for %%I in ("%SCALAR_RAW%") do echo Scalar source size: %%~zI bytes
    copy /y "%SCALAR_RAW%" "%SCALAR_BIN%" >nul
    if errorlevel 1 (
        echo ERROR: Could not stage scalar model as .bin.
        exit /b 1
    )
    "%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%SCALAR_BIN%" 0x70200000
    if errorlevel 1 (
        echo ERROR: Scalar model flash failed.
        exit /b 1
    )
    echo Scalar model flashed at 0x70200000.

    echo === Step 4b: Flash OBB model at 0x70700000 (required) ===
    echo OBB source: "%OBB_RAW%"
    for %%I in ("%OBB_RAW%") do echo OBB source size: %%~zI bytes
    copy /y "%OBB_RAW%" "%OBB_BIN%" >nul
    if errorlevel 1 (
        echo ERROR: Could not stage OBB model as .bin.
        exit /b 1
    )
    "%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%OBB_BIN%" 0x70700000
    if errorlevel 1 (
        echo ERROR: OBB model flash failed.
        exit /b 1
    )
    echo OBB model flashed at 0x70700000.

    if exist "%RECTIFIER_RAW%" (
        echo === Step 4c: Flash rectifier model at 0x70600000 (optional) ===
        echo Rectifier source: "%RECTIFIER_RAW%"
        for %%I in ("%RECTIFIER_RAW%") do echo Rectifier source size: %%~zI bytes
        copy /y "%RECTIFIER_RAW%" "%RECTIFIER_BIN%" >nul
        if errorlevel 1 (
            echo WARNING: Could not stage rectifier model as .bin.
        ) else (
            "%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%RECTIFIER_BIN%" 0x70600000
            if errorlevel 1 (
                echo WARNING: Rectifier model flash failed.
            ) else (
                echo Rectifier model flashed at 0x70600000.
            )
        )
    ) else (
        echo === Step 4c: Skipping rectifier (not found) ===
    )

    if exist "%SOURCE_CROP_BOX_RAW%" (
        echo === Step 4d: Flash source-crop-box model at 0x70B00000 (optional) ===
        echo Source-crop-box source: "%SOURCE_CROP_BOX_RAW%"
        for %%I in ("%SOURCE_CROP_BOX_RAW%") do echo Source-crop-box source size: %%~zI bytes
        copy /y "%SOURCE_CROP_BOX_RAW%" "%SOURCE_CROP_BOX_BIN%" >nul
        if errorlevel 1 (
            echo WARNING: Could not stage source-crop-box model as .bin.
        ) else (
            "%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%SOURCE_CROP_BOX_BIN%" 0x70B00000
            if errorlevel 1 (
                echo WARNING: Source-crop-box model flash failed.
            ) else (
                echo Source-crop-box model flashed at 0x70B00000.
            )
        )
    ) else (
        echo === Step 4d: Skipping source-crop-box (not found) ===
    )

    if exist "%TIP_FOCUS_WEIGHTS_RAW%" (
        echo === Step 4e: Flash tip-focus weights at 0x70400000 (optional) ===
        echo Tip-focus weights source: "%TIP_FOCUS_WEIGHTS_RAW%"
        for %%I in ("%TIP_FOCUS_WEIGHTS_RAW%") do echo Tip-focus weights source size: %%~zI bytes
        copy /y "%TIP_FOCUS_WEIGHTS_RAW%" "%TIP_FOCUS_WEIGHTS_BIN%" >nul
        if errorlevel 1 (
            echo WARNING: Could not stage tip-focus weights as .bin.
        ) else (
            "%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%TIP_FOCUS_WEIGHTS_BIN%" 0x70400000
            if errorlevel 1 (
                echo WARNING: Tip-focus weights flash failed.
            ) else (
                echo Tip-focus weights flashed at 0x70400000.
            )
        )
    ) else (
        echo === Step 4e: Skipping tip-focus weights (not found) ===
    )
) else (
    echo === Step 4: Skipping model image flash (FLASH_MODEL not set) ===
)

if "%FLASH_MODEL%"=="1" (
    echo.
    echo === Step 4d: Extract model signatures for firmware update ===
    python "%REPO_ROOT%ml\scripts\extract_model_signature.py" "%SCALAR_RAW%" > "%SIG_REPORT_DIR%\scalar_signature.txt"
    if errorlevel 1 (
        echo ERROR: Scalar signature extraction failed.
        exit /b 1
    )
    python "%REPO_ROOT%ml\scripts\extract_model_signature.py" "%OBB_RAW%" > "%SIG_REPORT_DIR%\obb_signature.txt"
    if errorlevel 1 (
        echo ERROR: OBB signature extraction failed.
        exit /b 1
    )
    echo Scalar signature report: "%SIG_REPORT_DIR%\scalar_signature.txt"
    echo OBB signature report: "%SIG_REPORT_DIR%\obb_signature.txt"
) else (
    echo === Step 4d: Skipping model signature extraction ===
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
    "%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%APP_SIGN%" 0x70100000
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
