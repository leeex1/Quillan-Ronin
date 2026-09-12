# Quillan Overnight Watcher - light keepalive, no priority thrashing
$python = "C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe"
$script = "C:\02_QUILLAN\03 - Training & Model\scripts\train_frontier_capability.py"
$log = "C:\02_QUILLAN\logs\frontier_stdout.log"
$stdout = "C:\02_QUILLAN\logs\frontier_stdout.log"
$stderr = "C:\02_QUILLAN\logs\frontier_stderr.log"
$env:PYTHONUNBUFFERED="1"
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
while($true){
  Start-Sleep -Seconds 60
  $trainer = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -and $_.CommandLine -match "train_frontier" }
  if(-not $trainer){
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content $log "[WATCHER $ts] No trainer found -> resurrecting" -Encoding UTF8
    try {
      Start-Process -FilePath $python -ArgumentList "-u `"$script`"" -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden
      Add-Content $log "[WATCHER $ts] Resurrected PID" -Encoding UTF8
      # deprioritize nodes again
      Get-Process node -ErrorAction SilentlyContinue | ForEach-Object { try { $_.PriorityClass='BelowNormal' } catch {} }
    } catch { Add-Content $log "[WATCHER err $_]" -Encoding UTF8 }
  } else {
    # keep nodes BelowNormal, trainer Normal
    Get-Process node -ErrorAction SilentlyContinue | ForEach-Object { try { if($_.PriorityClass -ne 'BelowNormal'){ $_.PriorityClass='BelowNormal' } } catch {} }
    $tp = Get-Process -Id $trainer.ProcessId -ErrorAction SilentlyContinue; if($tp -and $tp.PriorityClass -ne 'Normal'){ try { $tp.PriorityClass='Normal' } catch {} }
  }
  # tail progress to watcher log? just light
}
