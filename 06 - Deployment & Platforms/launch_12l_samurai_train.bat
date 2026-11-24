@echo off
title Quillan-Ronin 12L Main Master Training (1.163B Params)
echo =========================================================================
echo   QUILLAN-RONIN 12L MAIN SOVEREIGN ENGINE - MASTER TRAINING LAUNCHER
echo   Target Size : 1.163 Billion Parameters (34 Experts)
echo   Corpus      : ~204.92 Million Tokens (Full Samurai Multi-Dataset)
echo   Optimizer   : Sovereign Muon-K2 + AdamW Fused
echo =========================================================================
echo.

set PYTHON_EXE=C:\02_QUILLAN\09 - Projects\projects\venv_oni_cu126\Scripts\python.exe
set TRAIN_SCRIPT=C:\02_QUILLAN\03 - Training & Model\scripts\train_master_samurai.py

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Could not find venv python at %PYTHON_EXE%
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%TRAIN_SCRIPT%" ^
    --model_size 12l ^
    --batch_size 1 ^
    --accum_steps 32 ^
    --seq_len 512 ^
    --lr 1.8e-4 ^
    --lr_muon 0.010 ^
    --max_steps 12500 ^
    --warmup_steps 500 ^
    --save_every 250 ^
    --probe_every 50

echo.
echo [DONE] Training process finished.
pause
