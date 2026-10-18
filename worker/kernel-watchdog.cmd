@echo off
title QuillanWorker Kernel
cd /d C:\Users\Admin\QuillanWorker
:loop
echo [%date% %time%] kernel starting >> kernel.log
node server.js >> kernel.log 2>&1
echo [%date% %time%] kernel exited code %errorlevel% - restarting in 2s >> kernel.log
timeout /t 2 /nobreak >nul
goto loop
