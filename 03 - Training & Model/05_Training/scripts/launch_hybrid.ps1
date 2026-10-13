# Hybrid Launcher with Toolkit 500x gains
# Uses ZeRO Stage3 offload + grad_checkpoint + expandable_segments
# GPU 4GB + RAM 28GB + NVMe offload
param(
    [switch]$CpuOnly,
    [int]$Steps=15000,
    [int]$NLayer=6,
    [int]$SeqLen=256,
    [string]$Device="cuda"
)
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:CUDA_LAUNCH_BLOCKING="0"
$logOut="C:\02_QUILLAN\training_logs\oni_flagship_gpu_console.log"
$logErr="C:\02_QUILLAN\training_logs\oni_flagship_gpu_err.log"
# Ensure offload dir exists
if(-not (Test-Path "C:\02_QUILLAN\offload")){ New-Item -ItemType Directory -Path "C:\02_QUILLAN\offload" | Out-Null }
$args="C:\02_QUILLAN\oni\train_oni.py --resume --steps $Steps --n-layer $NLayer --device $Device --batch-size 1 --grad-accum 8 --seq-len $SeqLen --lr 3e-4 --warmup 200 --eval-every 250 --save-every 500"
if($CpuOnly){ $args=$args -replace "--device cuda","--device cpu" -replace "--seq-len 256","--seq-len 512" -replace "--batch-size 1","--batch-size 2" -replace "--grad-accum 8","--grad-accum 4" }
Write-Host "Launching toolkit hybrid: $args"
Write-Host "Env: PYTORCH_CUDA_ALLOC_CONF=$env:PYTORCH_CUDA_ALLOC_CONF"
Start-Process -FilePath "C:\02_QUILLAN\venv_oni_gpu\Scripts\python.exe" -ArgumentList $args -WorkingDirectory "C:\02_QUILLAN" -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr
Write-Host "Launched, logs: $logOut"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*train_oni*" } | Select-Object ProcessId,CommandLine | Format-List | Out-String -Width 700 | Write-Host
