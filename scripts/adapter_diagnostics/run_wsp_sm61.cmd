@echo off
REM Reversible GPU launcher for WSP probes on sm_61 (GTX 1050).
REM Uses isolated venv (torch 2.7.1+cu118) — does NOT touch system Python 3.14 / torch 2.12.
REM Usage: run_wsp_sm61.cmd [probe args...]
set VIPY=C:\02_QUILLAN\venv_sm61_test\Scripts\python.exe
set RUNNER=C:\02_QUILLAN\scripts\adapter_diagnostics\wsp_tcp_gpu_sm61.py
if not exist "%VIPY%" (echo [sm61] venv missing & exit /b 2)
echo [sm61] GPU probe via %VIPY%
"%VIPY%" -u "%RUNNER%" %*
