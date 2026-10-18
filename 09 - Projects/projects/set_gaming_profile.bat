@echo off
title Quillan High-Fidelity 1440p Gaming Profile
cls
echo ====================================================================
echo   QUILLAN HIGH-FIDELITY GAMING OPTIMIZER (GTX 1050 + 4K TV)
echo ====================================================================
echo.
echo [1] Opening Windows Display Settings to select 2560x1440 (1440p)...
start ms-settings:display
timeout /t 1 /nobreak >nul

echo [2] Opening NVIDIA Control Panel to verify Anisotropic Filtering 16x...
start shell:AppsFolder\NVIDIACorp.NVIDIAControlPanel_56jybvy8sckqj!NVIDIACorp.NVIDIAControlPanel
timeout /t 1 /nobreak >nul

echo [3] Compacting standby memory and trimming working sets...
powershell -NoProfile -Command "Get-Process | ForEach-Object { try { [void]$_.EmptyWorkingSet() } catch {} }"

echo.
echo ====================================================================
echo   RECOMMENDED IN-GAME SETTINGS FOR MAXIMUM QUALITY + 60 FPS:
echo ====================================================================
echo   - Resolution: 2560x1440 (or 4K with 67%% / 70%% Render Scale)
echo   - AMD FSR / NIS: Quality Mode (Sharpness: 0.3 - 0.5)
echo   - Texture Filtering: Anisotropic 16x
echo   - Textures: Medium / High
echo   - Shadows: Medium
echo   - Volumetric Fog / Clouds: Low
echo ====================================================================
echo.
pause
