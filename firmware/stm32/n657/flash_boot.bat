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
set "CENTER_DETECTOR_RAW=%SCRIPT_DIR%st_ai_output\packages\heatmap_cd_v4s_80\st_ai_output\heatmap_cd_atonbuf.xSPI2.raw"
set "TIP_FOCUS_RAW=%SCRIPT_DIR%st_ai_output\packages\tip_focus_v18_int8_n6_npu\st_ai_output\tip_focus_v18_int8_atonbuf.xSPI2.raw"
REM Board bbox OBB deploy candidate flashed into the OBB slot at 0x71400000.
set "OBB_RAW=%SCRIPT_DIR%st_ai_output\packages\obb_box_board_bbox_deploy_candidate\st_ai_output\obb_box_board_bbox_deploy_candidate_atonbuf.xSPI2.raw"

if not exist "%CENTER_DETECTOR_RAW%" set "CENTER_DETECTOR_RAW=%REPO_ROOT%firmware\stm32\n657\st_ai_output\packages\heatmap_cd_v4s_80\st_ai_output\heatmap_cd_atonbuf.xSPI2.raw"
if not exist "%TIP_FOCUS_RAW%" set "TIP_FOCUS_RAW=%REPO_ROOT%firmware\stm32\n657\st_ai_output\packages\tip_focus_v18_int8_n6_npu\st_ai_output\tip_focus_v18_int8_atonbuf.xSPI2.raw"
if not exist "%OBB_RAW%" set "OBB_RAW=%REPO_ROOT%firmware\stm32\n657\st_ai_output\packages\obb_box_board_bbox_deploy_candidate\st_ai_output\obb_box_board_bbox_deploy_candidate_atonbuf.xSPI2.raw"

REM CubeProgrammer v2.21 does not accept .raw extension with -w; stage as .bin
set "CENTER_DETECTOR_BIN=%SCRIPT_DIR%Appli\Debug\center_detector_model_stage.bin"
set "TIP_FOCUS_BIN=%SCRIPT_DIR%Appli\Debug\tip_focus_v18_int8_n6_npu.bin"
set "OBB_BIN=%SCRIPT_DIR%Appli\Debug\obb_model_stage.bin"

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
if "%FLASH_MODEL%"=="1" if not exist "%CENTER_DETECTOR_RAW%" (
    echo WARNING: Center detector model not found (optional — OBB face-localizer provides the centre): "%CENTER_DETECTOR_RAW%"
)
if "%FLASH_MODEL%"=="1" if not exist "%TIP_FOCUS_RAW%" (
    echo ERROR: Tip-focus model not found: "%TIP_FOCUS_RAW%"
    exit /b 1
)
if "%FLASH_MODEL%"=="1" if not exist "%OBB_RAW%" (
    echo ERROR: OBB face-localizer model not found: "%OBB_RAW%"
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
"%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%FSBL_TRUSTED%" 0x70000000
if errorlevel 1 (
    echo ERROR: FSBL flash failed.
    exit /b 1
)

echo.
if "%FLASH_MODEL%"=="1" (
    if exist "%CENTER_DETECTOR_RAW%" (
        echo === Step 4a: Flash heatmap center detector at 0x70200000 (optional — joint model provides the centre) ===
        echo Heatmap center detector source: "%CENTER_DETECTOR_RAW%"
        for %%I in ("%CENTER_DETECTOR_RAW%") do echo Center detector source size: %%~zI bytes
        copy /y "%CENTER_DETECTOR_RAW%" "%CENTER_DETECTOR_BIN%" >nul
        if errorlevel 1 (
            echo ERROR: Could not stage heatmap center detector model as .bin.
            exit /b 1
        )
        "%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%CENTER_DETECTOR_BIN%" 0x70200000
        if errorlevel 1 (
            echo ERROR: Heatmap center detector model flash failed.
            exit /b 1
        )
        echo Heatmap center detector model flashed at 0x70200000.
    ) else (
        echo === Step 4a: Skipping center detector (not found — joint model provides the centre) ===
    )

    echo === Step 4b: Flash tip-focus UNet v18 model at 0x70400000 (required) ===
    echo Tip-focus source: "%TIP_FOCUS_RAW%"
    for %%I in ("%TIP_FOCUS_RAW%") do echo Tip-focus source size: %%~zI bytes
    copy /y "%TIP_FOCUS_RAW%" "%TIP_FOCUS_BIN%" >nul
    if errorlevel 1 (
        echo ERROR: Could not stage tip-focus UNet model as .bin.
        exit /b 1
    )
    "%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%TIP_FOCUS_BIN%" 0x70400000
    if errorlevel 1 (
        echo ERROR: Tip-focus UNet model flash failed.
        exit /b 1
    )
    echo Tip-focus UNet v18 model flashed at 0x70400000.

    echo === Step 4c: Flash board bbox OBB candidate at 0x71400000 (required) ===
    echo Board bbox source: "%OBB_RAW%"
    for %%I in ("%OBB_RAW%") do echo Board bbox source size: %%~zI bytes
    copy /y "%OBB_RAW%" "%OBB_BIN%" >nul
    if errorlevel 1 (
        echo ERROR: Could not stage OBB face-localizer model as .bin.
        exit /b 1
    )
    "%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%OBB_BIN%" 0x71400000
    if errorlevel 1 (
        echo ERROR: OBB face-localizer model flash failed.
        exit /b 1
    )
    echo Board bbox OBB model flashed at 0x71400000.
)

if "%FLASH_MODEL%"=="1" (
    echo.
    echo === Step 4d: Extract model signatures for firmware update ===
    if exist "%CENTER_DETECTOR_RAW%" (
        python "%SCRIPT_DIR%tools\extract_model_signature.py" "%CENTER_DETECTOR_RAW%" > "%SIG_REPORT_DIR%\heatmap_cd_signature.txt"
        if errorlevel 1 (
            echo ERROR: Heatmap center detector signature extraction failed.
            exit /b 1
        )
        echo Heatmap center detector signature report: "%SIG_REPORT_DIR%\heatmap_cd_signature.txt"
    )
    python "%SCRIPT_DIR%tools\extract_model_signature.py" "%TIP_FOCUS_RAW%" > "%SIG_REPORT_DIR%\tip_focus_v18_signature.txt"
    if errorlevel 1 (
        echo ERROR: Tip-focus UNet signature extraction failed.
        exit /b 1
    )
    echo Tip-focus signature report: "%SIG_REPORT_DIR%\tip_focus_v18_signature.txt"
    python "%SCRIPT_DIR%tools\extract_model_signature.py" "%OBB_RAW%" > "%SIG_REPORT_DIR%\obb_signature.txt"
    if errorlevel 1 (
        echo ERROR: OBB signature extraction failed.
        exit /b 1
    )
    echo OBB signature report: "%SIG_REPORT_DIR%\obb_signature.txt"
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
