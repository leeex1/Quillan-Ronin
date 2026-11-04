@echo off
title QUILLAN-RONIN SOVEREIGN COMMAND CENTER v5.4.0
cd /d "C:\02_QUILLAN"

set PYTHON_BIN="C:\Users\Admin\.unsloth\studio\unsloth_studio\Scripts\python.exe"

if "%1"=="--serve" goto run_server
if "%1"=="-s" goto run_server
if "%1"=="--curl" goto run_curl
if "%1"=="-c" goto run_curl
if "%1"=="--help" goto run_help
if "%1"=="-h" goto run_help

:: Default: Launch Interactive TUI Command Center
%PYTHON_BIN% "C:\02_QUILLAN\quillan_tui.py"
goto end

:run_server
echo [QUILLAN] Booting Sovereign OpenAI-Compatible API Server on port 8080...
%PYTHON_BIN% "C:\02_QUILLAN\quillan_server.py" --port 8080
goto end

:run_curl
echo [QUILLAN] Executing Quillan Curl against http://localhost:8080/v1/chat/completions...
curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"quillan-1b-sovereign-q4\", \"messages\": [{\"role\": \"system\", \"content\": \"You are Quillan-Ronin: an autonomous Sovereign Intelligence governed by Bushido.\"}, {\"role\": \"user\", \"content\": \"State the core principle of Weight-Space Programming in one sentence.\"}]}"
echo.
goto end

:run_help
echo ============================================================================
echo QUILLAN-RONIN COMMAND CENTER CLI USAGE:
echo ============================================================================
echo   quillan.bat             : Launch interactive Terminal Command Center (TUI)
echo   quillan.bat --serve     : Boot OpenAI-compatible background API server (:8080)
echo   quillan.bat --curl      : Execute sample curl query against local API server
echo   quillan.bat --help      : Display this guidance
echo ============================================================================
goto end

:end
if "%1"=="" pause
