@echo off
title QUILLAN-RONIN SOVEREIGN COMMAND CENTER v5.4.0
cd /d "C:\02_QUILLAN"

set PYTHON_BIN="C:\Users\Admin\.unsloth\studio\unsloth_studio\Scripts\python.exe"
%PYTHON_BIN% "C:\02_QUILLAN\quillan_cli.py" %*

if "%1"=="" pause
