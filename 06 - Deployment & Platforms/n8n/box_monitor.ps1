#Requires -Version 5.1
<#
  Quillan box monitor -> posts process/service health to the local collector
  as agent events (agents view + activity feed). Stdlib only, localhost only.
  Runs forever every 60s. Transition-only logging (no feed spam).
  Stop: close this window / Ctrl+C.
#>
$ErrorActionPreference = "SilentlyContinue"
$COLLECTOR = "http://127.0.0.1:8777/api/events"
$PERIOD = 60
$prev = @{}

function Post-Event($agent, $type, $detail) {
    $body = @{ agent = $agent; type = $type; detail = $detail } | ConvertTo-Json -Compress
    try { Invoke-RestMethod -Uri $COLLECTOR -Method POST -Body $body -ContentType "application/json" -TimeoutSec 8 | Out-Null; return $true }
    catch { return $false }
}

function Port-Open($port) {
    return $null -ne (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Cpu-Pct {
    $c = Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average
    return [math]::Round($c.Average)
}
function Mem-Info {
    $os = Get-CimInstance Win32_OperatingSystem
    $total = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $free = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    return @{ Total = $total; Used = [math]::Round($total - $free, 1) }
}
function Gpu-Info {
    try {
        $o = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($o -match "(\d+)\s*,\s*(\d+)") { return @{ Util = $matches[1]; MemMB = $matches[2] } }
    } catch {}
    return $null
}

function Watch($agent, $isUp, $taskUp, $taskDown) {
    $key = $agent
    $was = $prev[$key]
    if ($isUp) {
        Post-Event $agent "status" ("online - " + $taskUp) | Out-Null
        if ($was -eq $false) { Post-Event $agent "log" ($agent + " recovered: " + $taskUp) | Out-Null }
        $prev[$key] = $true
    } else {
        Post-Event $agent "status" ("offline - " + $taskDown) | Out-Null
        if ($was -ne $false) { Post-Event $agent "log" ($agent + " DOWN: " + $taskDown) | Out-Null }
        $prev[$key] = $false
    }
}

Write-Output "box-monitor posting to $COLLECTOR every ${PERIOD}s. Ctrl+C to stop."
while ($true) {
    $anti = @(Get-Process -Name "Antigravity IDE" -ErrorAction SilentlyContinue).Count
    Watch "antigravity" ($anti -gt 0) ("IDE live, $anti procs") "IDE not running"

    $dev = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='node.exe'" |
        Where-Object { $_.CommandLine -like "*devin*" }).Count
    if ($dev -gt 0 -or $prev.ContainsKey("devin")) {
        Watch "devin" ($dev -gt 0) ("agent procs: $dev") "no devin procs"
    }

    Watch "n8n-server" (Port-Open 5678) "n8n up :5678" "port 5678 closed"
    Watch "quillan-collector" (Port-Open 8777) "collector up :8777" "port 8777 closed"
    Watch "quillan-hub" (Port-Open 8080) "hub up :8080" "port 8080 closed"
    Watch "validation-kit" (Port-Open 5000) "backend up :5000" "port 5000 closed"

    $mcp = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*09 - Projects\mcp\quillan*" }).Count
    Watch "mcp-kit" ($mcp -gt 0) ("$mcp quillan servers alive") "no quillan mcp servers"

    $tr = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*run_6l*" -or $_.CommandLine -like "*train*" }).Count
    if ($tr -gt 0 -or $prev.ContainsKey("quillan-training")) {
        Watch "quillan-training" ($tr -gt 0) ("training procs: $tr") "no training run"
    }

    $mem = Mem-Info; $gpu = Gpu-Info
    $gs = if ($gpu) { "GPU $($gpu.Util)% $($gpu.MemMB)MB" } else { "GPU n/a" }
    Post-Event "box" "status" ("online - CPU $(Cpu-Pct)% RAM $($mem.Used)/$($mem.Total)GB " + $gs) | Out-Null

    Start-Sleep -Seconds $PERIOD
}
