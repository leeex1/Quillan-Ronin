@echo off
cd /d C:\Users\Admin\QuillanWorker
echo === install start %time% === >> data\browse-install.log
call npm init -y >> ..\nul 2>&1
call npm install playwright-core chess.js --no-audit --no-fund >> data\browse-install.log 2>&1
echo === done exit=%errorlevel% %time% === >> data\browse-install.log
