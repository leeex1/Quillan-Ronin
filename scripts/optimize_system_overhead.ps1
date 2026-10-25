<#
.SYNOPSIS
    Quillan System Overhead Optimization Script
    Disables Windows Search Indexing and adds Windows Defender exclusions
    to prevent high CPU consumption during model pre-training and IDE operations.
#>

$ErrorActionPreference = "SilentlyContinue"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   QUILLAN-RONIN SYSTEM OVERHEAD OPTIMIZER               " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Non-Elevated: Set NotContentIndexed attribute on C:\02_QUILLAN
Write-Host "`n[1/4] Applying NotContentIndexed flag to C:\02_QUILLAN..." -ForegroundColor Yellow
try {
    $quillanDir = "C:\02_QUILLAN"
    if (Test-Path $quillanDir) {
        attrib +I $quillanDir /S /D
        Write-Host "  -> Content indexing disabled on $quillanDir and subfolders." -ForegroundColor Green
    }
} catch {
    Write-Host "  -> Note: Attrib warning: $_" -ForegroundColor DarkGray
}

# 2. Check for Administrator Privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "`n[!] Windows Defender exclusions and WSearch service management require Administrator rights." -ForegroundColor Yellow
    Write-Host "    Requesting elevation (click 'Yes' on the UAC prompt)..." -ForegroundColor Cyan
    
    $scriptPath = $MyInvocation.MyCommand.Definition
    if (-not $scriptPath) {
        $scriptPath = "C:\02_QUILLAN\scripts\optimize_system_overhead.ps1"
    }
    
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Elevated"
    exit
}

# --- ELEVATED ACTIONS BELOW ---

Write-Host "`n[2/4] Configuring Windows Defender Exclusions..." -ForegroundColor Yellow
try {
    # Path exclusions
    Add-MpPreference -ExclusionPath "C:\02_QUILLAN"
    Add-MpPreference -ExclusionPath "C:\Users\Admin\.gemini"
    Write-Host "  -> Excluded folders: C:\02_QUILLAN, C:\Users\Admin\.gemini" -ForegroundColor Green

    # Process exclusion for Python runtime
    Add-MpPreference -ExclusionProcess "python.exe"
    Write-Host "  -> Excluded process: python.exe" -ForegroundColor Green

    # Extension exclusions for large model binaries and datasets
    Add-MpPreference -ExclusionExtension ".pt", ".bin", ".safetensors", ".parquet", ".arrow"
    Write-Host "  -> Excluded extensions: .pt, .bin, .safetensors, .parquet, .arrow" -ForegroundColor Green

    # Cap Defender scan CPU usage
    Set-MpPreference -ScanAvgCPULoadFactor 15
    Write-Host "  -> Capped Defender ScanAvgCPULoadFactor to 15%" -ForegroundColor Green
} catch {
    Write-Host "  -> Defender configuration error: $_" -ForegroundColor Red
}

Write-Host "`n[3/4] Stopping and Disabling Windows Search Indexer Service..." -ForegroundColor Yellow
try {
    Stop-Service -Name WSearch -Force
    Set-Service -Name WSearch -StartupType Manual
    Write-Host "  -> WSearch service stopped and set to Manual (will not hog CPU)." -ForegroundColor Green
} catch {
    Write-Host "  -> WSearch error: $_" -ForegroundColor Red
}

Write-Host "`n[4/4] Verification..." -ForegroundColor Yellow
$wsearch = Get-Service -Name WSearch
Write-Host "  -> Windows Search Service Status: $($wsearch.Status) ($($wsearch.StartType))" -ForegroundColor Cyan
Write-Host "`n[SUCCESS] System overhead optimization complete!" -ForegroundColor Green
Write-Host "Defender and SearchIndexer will no longer compete for CPU cycles." -ForegroundColor Green
Write-Host "You can close this window.`n" -ForegroundColor Gray

Start-Sleep -Seconds 5
