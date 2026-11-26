@echo off
REM One-command rollback of the 2026-11-22 telemetry producer + guards.
REM Disables the producer and restores the three backed-up daemons.
schtasks /Change /TN "QuillanHostTelemetry" /DISABLE >nul 2>&1
echo Telemetry producer task disabled.
echo.
echo To restore source files manually, copy the *.bak-20261122_142057 files
echo over their originals in daemons\ and Validation-test-kit\backend\.
echo Then restart the governor and autoboost daemons.
