@echo off
REM Quillan-Ronin Unified OS Launcher
REM Order: Ollama (:11434) -> Sovereign Server (:8080) -> Collector Hub (:8777) -> Worker + Extension Bridge (:7777)
set REPO=C:\02_QUILLAN
set PY=C:\Users\Admin\.unsloth\studio\unsloth_studio\Scripts\python.exe
if not exist "%PY%" set PY=C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe
if not exist "%PY%" set PY=python

start "ollama-serve" /min ollama serve
timeout /t 5 /nobreak >nul
start "quillan-sovereign-server" /min "%PY%" "%REPO%\quillan_server.py" --host 127.0.0.1 --port 8080
start "quillan-collector-hub" /min "%PY%" "%REPO%\06 - Deployment & Platforms\quillan_collector.py"
start "quillan-worker" /min node "%REPO%\worker\server.js"
timeout /t 3 /nobreak >nul

echo Quillan OS stack launched. Endpoints:
echo   http://127.0.0.1:8080/health      (Sovereign API Server)
echo   http://127.0.0.1:8777/api/health  (Telemetry Collector & Dashboard)
echo   http://127.0.0.1:7777/api/state   (QuillanWorker & Chrome Extension Bridge)
echo   http://127.0.0.1:11434/api/version (Ollama Engine)
pause
