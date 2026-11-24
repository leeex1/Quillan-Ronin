@echo off
title Quillan-Ronin 6L Mini Master Training (562.8M Params)
echo =========================================================================
echo   QUILLAN-RONIN 6L MINI SOVEREIGN ENGINE - MASTER TRAINING LAUNCHER
echo   Target Size : 562.8 Million Parameters (34 Experts)
echo   Corpus      : ~204.92 Million Tokens (Full Samurai Multi-Dataset)
echo   Optimizer   : Sovereign Muon-K2 + AdamW Fused
echo   Hardware    : NVIDIA GeForce GTX 1050 (sm_61 CUDA 12.6)
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
    --model_size 6l ^
    --batch_size 1 ^
    --accum_steps 16 ^
    --seq_len 512 ^
    --lr 2.5e-4 ^
    --lr_muon 0.015 ^
    --max_steps 12500 ^
    --warmup_steps 250 ^
    --save_every 250 ^
    --probe_every 50 ^
    --device cuda

echo.
echo [DONE] Training process finished.
pause
