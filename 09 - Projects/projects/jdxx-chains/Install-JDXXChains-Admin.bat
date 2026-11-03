@echo off
REM JDXX Chains — install the built VST3 to the system-wide VST3 folder.
REM MUST be run as Administrator (Program Files is protected).

setlocal
set "SRC=%~dp0build\JdxxChains_artefacts\Release\VST3\JDXX Chains.vst3"
set "DEST=C:\Program Files\Common Files\VST3\JDXX Chains.vst3"

if not exist "%SRC%" (
  echo [ERROR] Built plugin not found at:
  echo         %SRC%
  echo Run the Release build first:
  echo   cmake --build build --config Release
  exit /b 1
)

echo Installing JDXX Chains to system VST3 folder...
xcopy /E /I /Y "%SRC%" "%DEST%"
if errorlevel 1 (
  echo [ERROR] Copy failed. Are you running as Administrator?
  exit /b 1
)
echo [OK] Installed: %DEST%
echo Restart your DAW (or rescan plugins) to see "JDXX Chains".
endlocal
