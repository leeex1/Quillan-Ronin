@echo off
REM ⚡ Build script for Quillan.cpp using LLVM Clang++ with AVX2 & FMA
set CLANG="C:\Program Files\LLVM\bin\clang++.exe"
set SRC="src\quillan_cli.cpp"
set OUT="quillan_engine.exe"

echo [QUILLAN.CPP] Compiling Native AVX2/FMA Engine...
%CLANG% -std=c++20 -mavx2 -mfma -O3 %SRC% -lpsapi -o %OUT%

if %ERRORLEVEL% EQU 0 (
    echo [OK] Successfully built %OUT%
) else (
    echo [ERROR] Build failed with exit code %ERRORLEVEL%
)
