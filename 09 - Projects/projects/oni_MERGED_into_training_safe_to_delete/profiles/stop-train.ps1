# stop-train.ps1 — pause training. Default = lossless (STOP flag, chunk loop exits
# after current chunk saves). -Force = kill now + honest loss estimate.
# Usage: .\stop-train.ps1 | .\stop-train.ps1 -Force
param([switch]$Force)
$ErrorActionPreference="SilentlyContinue"
$LogDir="C:\02_QUILLAN\05_Training\training_logs"
$Flag="$LogDir\TRAIN_STOP.flag"
New-Item -ItemType File -Path $Flag -Force | Out-Null
Write-Output "STOP flag planted: chunk loop pauses after current chunk saves (zero loss)."
if($Force){
  $tp=Get-CimInstance Win32_Process -Filter "Name like '%python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*train_oni*" }
  if(!$tp){ Write-Output "No live train process (already paused/exited). Nothing killed."; exit 0 }
  $tp | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Write-Output "Killed live train proc(s). Lost = steps since last chunk SAVE (<= chunk size, see chunk log)."
}
