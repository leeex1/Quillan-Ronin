@echo off
title Quillan System Overhead Optimizer
echo Launching PowerShell with Administrator privileges...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"\"%~dp0optimize_system_overhead.ps1\"\"'"
