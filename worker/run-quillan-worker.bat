@echo off
title Quillan-Ronin Sovereign Worker (Port 7777)
cd /d "%~dp0"

echo ======================================================================
echo    QUILLAN-RONIN SOVEREIGN WORKER DAEMON & BROWSER BRIDGE
echo ======================================================================

:: Clean up any stale/orphaned node instances on port 7777
echo [*] Checking for stale worker or MCP instances...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":7777" ^| find "LISTENING"') do (
    echo [*] Terminating previous process on port 7777 (PID: %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

:: Ensure node modules exist
if not exist "node_modules" (
    echo [*] Installing dependencies...
    call npm install
)

echo [*] Starting QuillanWorker Server on http://localhost:7777 ...
echo [*] SSE streaming enabled | Tree-kill lifecycle active | MCP auto-reaper on
echo.

node server.js

pause
