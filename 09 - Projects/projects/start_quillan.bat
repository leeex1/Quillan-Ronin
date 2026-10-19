@echo off
REM Quillan-Ronin post-reboot recovery (owner double-clicks once).
REM Order: Ollama serve -> gateway (loads head_v62_best) -> discord relay.
set REPO=C:\02_QUILLAN
start "ollama-serve" /min ollama serve
timeout /t 15 /nobreak >nul
start "quillan-gateway" /min python "%REPO%\scripts\quillan_gateway.py"
timeout /t 60 /nobreak >nul
start "quillan-relay" /min python "%REPO%\scripts\quillan_discord_relay.py"
echo All three launched. Gateway needs ~2 min to load 1.9GB. Check:
echo   http://127.0.0.1:8000/api/health  (gateway)
echo   http://127.0.0.1:11434/api/version (ollama)
pause
