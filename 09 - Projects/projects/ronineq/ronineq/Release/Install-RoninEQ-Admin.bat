@echo off
REM RoninEQ VST3 installer — RIGHT-CLICK this file and choose "Run as administrator".
REM Copies the staged bundle into the system VST3 folder that Cakewalk scans.

set SRC=%~dp0RoninEQ.vst3
set DEST=C:\Program Files\Common Files\VST3\RoninEQ.vst3

echo Installing RoninEQ VST3...
echo   From: %SRC%
echo   To:   %DEST%
echo.

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Not running as administrator.
    echo Right-click Install-RoninEQ-Admin.bat and choose "Run as administrator".
    pause
    exit /b 1
)

if not exist "%SRC%" (
    echo ERROR: Could not find "%SRC%"
    pause
    exit /b 1
)

robocopy "%SRC%" "%DEST%" /MIR /NFL /NDL /NJH /NJS
if %errorlevel% geq 8 (
    echo ERROR: Copy failed with code %errorlevel%.
    pause
    exit /b 1
)

echo.
echo SUCCESS: RoninEQ.vst3 installed.
echo Next: open Cakewalk, rescan plugins (Utilities -^> Cakewalk Plug-in Manager
echo -^> VST Audio Effects -^> Scan), then insert RoninEQ on a track.
pause
