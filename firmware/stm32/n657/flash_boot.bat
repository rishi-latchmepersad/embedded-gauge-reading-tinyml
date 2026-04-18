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
set "FSBL_BIN=%SCRIPT_DIR%FSBL\Debug\n657_FSBL.bin"
set "FSBL_TRUSTED=%SCRIPT_DIR%FSBL\Debug\FSBL_trusted.bin"
set "SCALAR_RAW=%SCRIPT_DIR%..\..\..\st_ai_output\atonbuf.xSPI2.raw"
set "RECTIFIER_RAW=%SCRIPT_DIR%..\..\..\st_ai_output\atonbuf.rectifier.xSPI2.raw"
REM CubeProgrammer v2.21 does not accept .raw extension with -w; stage as .bin
set "SCALAR_BIN=%SCRIPT_DIR%Appli\Debug\scalar_model_stage.bin"
set "RECTIFIER_BIN=%SCRIPT_DIR%Appli\Debug\rectifier_model_stage.bin"
set "APP_BIN=%SCRIPT_DIR%Appli\Debug\n657_Appli.bin"
set "APP_SIGN=%SCRIPT_DIR%Appli\Debug\n657_Appli_sign_new.bin"
set "FLASH_MODEL=1"
set "FLASH_APP=1"

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
if "%FLASH_MODEL%"=="1" if not exist "%RECTIFIER_RAW%" (
    echo ERROR: Rectifier model not found: "%RECTIFIER_RAW%"
    exit /b 1
)

echo.
echo === Step 2: Sign FSBL binary ===
"%SIGN%" -bin "%FSBL_BIN%" -nk -of 0x80000000 -t fsbl -hv 2.3 -o "%FSBL_TRUSTED%" -dump "%FSBL_TRUSTED%" -align
if errorlevel 1 (
    echo ERROR: FSBL signing failed.
    exit /b 1
)
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
    echo === Step 4a: Flash scalar model at 0x70200000 ===
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

    echo === Step 4b: Flash rectifier model at 0x70520000 ===
    copy /y "%RECTIFIER_RAW%" "%RECTIFIER_BIN%" >nul
    if errorlevel 1 (
        echo ERROR: Could not stage rectifier model as .bin.
        exit /b 1
    )
    "%PROG%" -c port=SWD mode=HOTPLUG -el "%ELDR%" -hardRst -w "%RECTIFIER_BIN%" 0x70520000
    if errorlevel 1 (
        echo ERROR: Rectifier model flash failed.
        exit /b 1
    )
    echo Rectifier model flashed at 0x70520000.
) else (
    echo === Step 4: Skipping model image flash (FLASH_MODEL not set) ===
)

if "%FLASH_APP%"=="1" (
    echo.
    echo === Step 5: Sign application binary ===
    "%SIGN%" -bin "%APP_BIN%" -nk -of 0x80000000 -t ssbl -hv 2.3 -o "%APP_SIGN%" -align
    if errorlevel 1 (
        echo ERROR: Signing failed.
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

