# Quillan PC Boost — safe + reversible only. Nothing destructive.
# Run before training: .\pc_boost.ps1
# Revert: power plan back to Balanced via `powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e`
$ErrorActionPreference = "SilentlyContinue"

Write-Output "=== QUILLAN PC BOOST (safe) ==="

# 1. Power plan -> High performance (already active on this box; idempotent)
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
Write-Output "[1] Power plan: High performance"

# 2. Temp cleanup, files older than 2 days only, locked files skipped
$n = 0
Get-ChildItem $env:TEMP -Recurse -Force -ErrorAction SilentlyContinue |
  Where-Object { -not $_.PSIsContainer -and $_.LastWriteTime -lt (Get-Date).AddDays(-2) } |
  ForEach-Object { try { Remove-Item $_.FullName -Force -ErrorAction Stop; $n++ } catch {} }
Write-Output "[2] Temp cleanup: $n files removed"

# 3. RAM report + floor check (8GB, same floor as hybrid_compute.json)
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
Write-Output "[3] RAM free: ${freeGB}GB $(if ($freeGB -lt 8) { '(BELOW 8GB FLOOR - close browser tabs)' } else { '(OK)' })"

# 4. Top RAM hogs (report only — never kills)
Write-Output "[4] Top RAM hogs (report only):"
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 5 Name,
  @{N='MB';E={[int]($_.WorkingSet64/1MB)}} | Format-Table | Out-String | Write-Output

# 5. GPU scheduling state (report only)
$hw = (Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -Name HwSchMode).HwSchMode
Write-Output "[5] HW GPU scheduling: $hw (2=on)"

# 6. Registry snapshot verify-only (f9ee audit 2026-09-03: warn, never write)
# Games Scheduling Category must be Medium so Priority 6 is honored
$g = Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile\Tasks\Games" -ErrorAction SilentlyContinue
$cat = $g.PSObject.Properties['Scheduling Category'].Value
if ($cat -ne 'Medium') { Write-Output ('[6] WARN Games Scheduling Category=' + $cat + ' (want Medium)') } else { Write-Output '[6] Games Scheduling Category=Medium OK, Priority 6 honored' }
$ac = (Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\GameDVR" -Name AppCaptureEnabled -ErrorAction SilentlyContinue).AppCaptureEnabled
if ($ac -ne 0) { Write-Output ('[6] WARN GameDVR AppCaptureEnabled=' + $ac + ' (want 0)') } else { Write-Output '[6] GameDVR AppCaptureEnabled=0 OK' }

Write-Output "=== DONE. Destructive steps (startup trims, Brave, OneDrive) need your approval ==="
