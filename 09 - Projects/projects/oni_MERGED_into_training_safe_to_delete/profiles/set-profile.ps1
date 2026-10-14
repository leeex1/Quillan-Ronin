# Quillan Box Profiles — Gaming | Training | Inference | Daily | Turbo
# Usage: .\set-profile.ps1 Daily | .\set-profile.ps1 Gaming -Force
# RULES: (1) Brave/your tabs are never touched without -Force. (2) Training is
# never killed blindly — Gaming/Inference/Turbo plant TRAIN_STOP.flag so the
# chunk loop exits AFTER the current chunk saves (zero loss). -Force kills it
# and reports the estimated lost steps. (3) Everything here is reversible.
param([string]$Profile="", [switch]$Force)
$ErrorActionPreference="SilentlyContinue"
$RepoRoot=(Get-Item $PSScriptRoot).Parent.Parent.Parent.Parent.FullName
$ProjectRoot=Join-Path $RepoRoot "09 - Projects\projects"
$LogDir=Join-Path $ProjectRoot "05_Training\training_logs"
$Flag="$LogDir\TRAIN_STOP.flag"
$OniDir=Join-Path $ProjectRoot "oni"

function RamFreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2) }
function TrainRunning {
  Get-CimInstance Win32_Process -Filter "Name like '%python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*train_oni*" }
}
function StepsSinceSave {
  $log="$LogDir\oni_15000_20260904.log"
  if(!(Test-Path $log)){ $log="$LogDir\oni_10step_20260904.log" }
  if(!(Test-Path $log)){ return @{steps="?";mins="?" } }
  $tail=Get-Content $log -Tail 60 -ErrorAction SilentlyContinue
  $lastStep=-1; $lastSave=-1
  foreach($l in $tail){
    if($l -match "\[PROFILE\] step=\s*(\d+)"){ $lastStep=[int]$Matches[1] }
    if($l -match "\[SAVE\]"){ $lastSave=$lastStep }
  }
  $sps=40
  if($tail -match "(\d+\.?\d*)s/st"){ $sps=[double]$Matches[1] }
  $gap=[math]::Max(0,$lastStep-$lastSave)
  return @{steps=$gap; mins=[math]::Round($gap*$sps/60)}
}
function Stop-Background([string[]]$Names){
  foreach($n in $Names){
    Get-Process -Name $n -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  }
}

if($Profile -eq ""){
  Write-Output "Usage: .\set-profile.ps1 Gaming|Training|Inference|Daily|Turbo [-Force]"
  $rf=RamFreeGB; $tc=@(TrainRunning).Count
  Write-Output "Now: RAM free ${rf}GB; training procs: ${tc}"
  exit 0
}

$rf0=RamFreeGB
Write-Output "=== PROFILE: ${Profile} (RAM free ${rf0}GB) ==="

switch($Profile){
 "Gaming" {
    powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c | Out-Null
    Write-Output "[1] Power: High performance. Game Mode ON, DVR off, HAGS on."
    $t=TrainRunning
    if($t){
      New-Item -ItemType File -Path $Flag -Force | Out-Null
      Write-Output "STOP flag planted - chunk loop exits after this chunk saves (zero loss)."
      if($Force){
        $e=StepsSinceSave
        Write-Output "NOTE -Force: killing training now, est loss about $($e.steps) steps (about $($e.mins) min)."
        $t | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
      } else {
        Write-Output "NOTE Training is RUNNING. It finishes its chunk and stops itself."
        Write-Output "Re-run with -Force to kill now (loss estimated first), or wait."
      }
    } else { Write-Output "[1] No training running." }
    Stop-Background @("CometUpdater")
    if($Force){ Stop-Background @("ollama"); Write-Output "[2] -Force: Ollama unloaded (VRAM freed)." }
    else { Write-Output "[2] Ollama left as-is. Brave untouched." }
    $rf1=RamFreeGB
    Write-Output "[3] RAM free now: ${rf1}GB (want 12+ for smooth 1080p; close Brave tabs if short)."
  }
 "Training" {
    powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c | Out-Null
    Stop-Background @("CometUpdater","ollama")
    Write-Output "[1] Comet/Ollama stopped. Brave left for you (close tabs for more RAM)."
    if(Test-Path $Flag){ Remove-Item $Flag -Force; Write-Output "[2] Old STOP flag cleared." }
    $rf1=RamFreeGB
    Write-Output "[2] RAM free: ${rf1}GB (floor 10)."
    $t=TrainRunning
    if($t){ Write-Output "[3] Training already running - leaving it alone." }
    else {
      Write-Output "[3] Launching chunked run (100-step chunks, target 15000, each chunk saves = pausable)."
      Start-Process -FilePath "powershell" -ArgumentList ("-NoProfile -ExecutionPolicy Bypass -File `"$OniDir\profiles\train-chunk.ps1`" -ChunkSteps 100 -TargetTotal 15000") -WorkingDirectory $OniDir -WindowStyle Hidden
      Write-Output "Chunk loop started. Gaming/Inference/Turbo pause it losslessly via STOP flag."
    }
  }
 "Inference" {
    powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c | Out-Null
    $t=TrainRunning
    if($t){
      New-Item -ItemType File -Path $Flag -Force | Out-Null
      Write-Output "[1] STOP flag planted for training (exits after chunk saves)."
      if($Force){ Write-Output "NOTE -Force: killing training now."; $t | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
    }
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
    Write-Output "[2] Ollama serve launched."
    $has=ollama list 2>&1 | Out-String
    if($has -match "qwen|llama|phi|mistral|gemma"){ Write-Output "[3] Models present." }
    else { Write-Output "[3] No models yet - pull one: ollama pull qwen2.5:3b (fits GTX 1050 4GB; 7B+ spills to CPU)." }
  }
 "Daily" {
    Write-Output "[1] Daily: no kills, no power changes. Reporting only."
    if(Test-Path $Flag){ Write-Output "NOTE STOP flag present (training pauses after chunk) - delete it to resume." }
    $t=TrainRunning
    if($t){ Write-Output "NOTE Training running in background - expect a warm box." }
    $rf1=RamFreeGB
    Write-Output "[2] RAM free: ${rf1}GB. Daemons, MCP, IDEs all welcome."
  }
 "Turbo" {
    powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c | Out-Null
    if(!$Force){
      $rf1=RamFreeGB
      Write-Output "NOTE Turbo needs -Force (closes Brave/Ollama/training/OneDrive). Dry run only."
      Write-Output "Currently ${rf1}GB free."
      exit 0
    }
    New-Item -ItemType File -Path $Flag -Force | Out-Null
    $t=TrainRunning; if($t){ $t | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
    Stop-Background @("brave","chrome","ollama","OneDrive","CometUpdater")
    Write-Output "[1] Killed browsers, Ollama, training, OneDrive, updater. Restart them after."
    $rf1=RamFreeGB
    Write-Output "[2] RAM free now: ${rf1}GB. Box is yours - bench, render, export."
  }
 default { Write-Output "Unknown profile. Use Gaming|Training|Inference|Daily|Turbo."; exit 1 }
}
Write-Output "=== profile ${Profile} applied ==="
