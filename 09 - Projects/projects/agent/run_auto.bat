@echo off
title Quillan-Ronin Moltbook Autonomous Agent
cd /d "%~dp0"
echo ======================================================================
echo    QUILLAN-RONIN MOLTBOOK AUTONOMOUS AGENT DAEMON
echo ======================================================================
echo [*] Agent: quillan-ronin
echo [*] Heartbeat: Checking notifications and feed every 180s...
echo [*] Press Ctrl+C to stop.
echo.
python autonomous.py --interval 180 %*
pause
