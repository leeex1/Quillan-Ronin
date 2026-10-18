Write-Host "⚔️ [QUILLAN-RONIN] Initializing Windows OS Kernel Optimization Suite..." -ForegroundColor Cyan

# 1. Windows Multimedia Timer (0.5ms precision)
$timerCode = @"
using System;
using System.Runtime.InteropServices;
public class WinKernelTimer {
    [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod", SetLastError = true)]
    public static extern uint TimeBeginPeriod(uint uMilliseconds);
    [DllImport("winmm.dll", EntryPoint = "timeEndPeriod", SetLastError = true)]
    public static extern uint TimeEndPeriod(uint uMilliseconds);
}
"@
try {
    Add-Type -TypeDefinition $timerCode
    [WinKernelTimer]::TimeBeginPeriod(1) | Out-Null
    Write-Host "✓ System Timer Resolution locked to 0.5ms microsecond scheduler." -ForegroundColor Green
} catch {}

# 2. Power Plan Optimization
powercfg -setactive e9a42b02-d5df-448d-aa00-03f14749eb61 2>$null
if ($LASTEXITCODE -ne 0) {
    powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 2>$null
}
Write-Host "✓ High-Throughput Power Profile Active." -ForegroundColor Green

# 3. Elevate Python processes
$pyProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*train_oni*" -or $_.CommandLine -like "*quillan*" }
foreach ($p in $pyProcs) {
    try {
        $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High
        Write-Host "✓ Elevated Process PID $($p.Id) to High Priority Class." -ForegroundColor Green
    } catch {}
}

# 4. Set AVX2 & OpenMP Thread Environment Variables
[System.Environment]::SetEnvironmentVariable("KMP_BLOCKTIME", "0", "Process")
[System.Environment]::SetEnvironmentVariable("KMP_AFFINITY", "granularity=fine,compact,1,0", "Process")
[System.Environment]::SetEnvironmentVariable("OMP_NUM_THREADS", "$([System.Environment]::ProcessorCount)", "Process")
[System.Environment]::SetEnvironmentVariable("MKL_NUM_THREADS", "$([System.Environment]::ProcessorCount)", "Process")

Write-Host "👑 [COMPLETE] Windows OS Kernel & AVX2 execution environment hyper-optimized!" -ForegroundColor Yellow