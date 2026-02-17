@echo off
setlocal

REM ==============================================================================
REM Purpose:
REM   Build and run host-based Unity tests using CMake.
REM ==============================================================================

if not exist build (
  mkdir build
)

cmake -S . -B build
if errorlevel 1 exit /b 1

cmake --build build
if errorlevel 1 exit /b 1

echo.
echo Running unit tests...
echo.
build\unit_tests.exe

endlocal
