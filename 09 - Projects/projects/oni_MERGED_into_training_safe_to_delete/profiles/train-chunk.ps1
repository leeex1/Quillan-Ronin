# Chunked training loop - pausable with ZERO loss. Each chunk ends at an exact
# --steps boundary, and train_oni.py SAVES at every boundary. Planting
# TRAIN_STOP.flag (via set-profile Gaming/Inference/Turbo or stop-train.ps1)
# exits cleanly BETWEEN chunks: nothing after the last save is ever lost.
# Usage: .\train-chunk.ps1 -ChunkSteps 100 -TargetTotal 15000 [-NoRqgm]
param([int]$ChunkSteps=100, [int]$TargetTotal=15000, [switch]$NoRqgm)
$ErrorActionPreference="SilentlyContinue"
$RepoRoot=(Get-Item $PSScriptRoot).Parent.Parent.Parent.Parent.FullName
$ProjectRoot=Join-Path $RepoRoot "09 - Projects\projects"
$LogDir=Join-Path $ProjectRoot "05_Training\training_logs"
$Jsonl="$LogDir\oni_train_log.jsonl"
$Flag="$LogDir\TRAIN_STOP.flag"
$Py="C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe"
$Train=Join-Path $ProjectRoot "oni\train_oni.py"
$ChunkLog="$LogDir\chunk_loop.log"
$OniDir=Join-Path $ProjectRoot "oni"

function CurStep {
  # Source of truth order: chunk_state.json (written after every good chunk)
  # else checkpoint header (slow, once). NEVER jsonl max (append-mode history lies).
  $state="$LogDir\chunk_state.json"
  if(Test-Path $state){
    try { return [int](Get-Content $state -Raw | ConvertFrom-Json).next_start } catch {}
  }
  $ck=Join-Path $ProjectRoot "05_Training\checkpoints\checkpoints_oni\quillan_oni_latest.pt"
  if(Test-Path $ck){
    $s=C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe "$OniDir\profiles\ckpt-step.py" $ck 2>$null
    if($s -match "^\d+$"){ return [int]$s }
  }
  return 0
}
"[chunk] loop start $(Get-Date -Format 'yyyy-MM-dd HH:mm') target=$TargetTotal chunk=$ChunkSteps" | Out-File $ChunkLog -Append -Encoding utf8

while($true){
  if(Test-Path $Flag){
    "[chunk] STOP flag seen - paused losslessly. Delete flag + rerun to resume." | Out-File $ChunkLog -Append -Encoding utf8
    exit 0
  }
  $cur=CurStep
  if($cur -ge $TargetTotal){
    "[chunk] target $TargetTotal reached at step $cur. Done." | Out-File $ChunkLog -Append -Encoding utf8
    exit 0
  }
  $next=[math]::Min($cur+$ChunkSteps, $TargetTotal)
  "[chunk] launching: resume $cur -> $next" | Out-File $ChunkLog -Append -Encoding utf8
  $extra=""
  if($NoRqgm){ $extra=" --rqgm-disable" }
  $log="$LogDir\oni_chunk_$($cur)_$($next).log"
  # Proven volume on 28GB box: batch 1 / accum 1 (batch 2x4 dies silently in first fwd). Do not raise without a voltest.
  $p=Start-Process -FilePath $Py -ArgumentList ("-u `"$Train`" --steps $next --batch-size 1 --grad-accum 1 --device cpu --resume"+$extra) -WorkingDirectory $OniDir -RedirectStandardOutput $log -RedirectStandardError ($log+".err") -WindowStyle Hidden -PassThru
  $p.WaitForExit()
  $code=$p.ExitCode
  "[chunk] chunk exit code $code (log oni_chunk_$($cur)_$($next).log)" | Out-File $ChunkLog -Append -Encoding utf8
  if($code -and $code -ne 0){
    "[chunk] NONZERO exit - STOP flag left so box is safe; inspect log, delete flag, rerun." | Out-File $ChunkLog -Append -Encoding utf8
    New-Item -ItemType File -Path $Flag -Force | Out-Null
    exit $p.ExitCode
  }
  (@{next_start=$next} | ConvertTo-Json) | Out-File "$LogDir\chunk_state.json" -Encoding utf8
  Start-Sleep 5
}
