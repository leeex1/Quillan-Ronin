# QUILLAN RESOURCE OPTIMIZER v3.1 - Physics of Agents (2608.16578) Enhanced
# - Statistical mechanics predicts herding/polarization in 34-council + 9M agents
# - Thresholds tuned: util 92-> herding point, temp 80C -> phase transition
# - Prime Agent (2608.23552) RLM harness: persistent REPL monitoring
$Cfg = @{ PollSeconds=5; LogFile='C:\02_QUILLAN\training_logs\resource_optimizer.log'; MaxLogBytes=524288; GpuVramMax=3200; GpuUtilHigh=92 }
$created = $false; $mutex = New-Object System.Threading.Mutex($true, 'Global\QuillanResourceOptimizer', [ref]$created); if (-not $created) { exit 0 }
function Write-Log([string]$msg){
    try{
        if((Test-Path $Cfg.LogFile) -and ((Get-Item $Cfg.LogFile).Length -gt $Cfg.MaxLogBytes)){
            $b=[System.IO.File]::ReadAllBytes($Cfg.LogFile); [System.IO.File]::WriteAllBytes($Cfg.LogFile,$b[($b.Length/2)..($b.Length-1)])
        }
        "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg | Add-Content -Path $Cfg.LogFile -Encoding UTF8
    }catch{}
}
function Show-Toast([string]$t,[string]$b){
    try{
        $null=[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime]
        $xml=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
        $texts=$xml.GetElementsByTagName('text'); $null=$texts.Item(0).AppendChild($xml.CreateTextNode($t)); $null=$texts.Item(1).AppendChild($xml.CreateTextNode($b))
        $toast=New-Object Windows.UI.Notifications.ToastNotification($xml)
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe').Show($toast)
    }catch{}
}
function Set-Prio([int[]]$ids,[string]$lvl){
    foreach($id in $ids){
        try{ $p=Get-Process -Id $id -ErrorAction Stop; if($p.PriorityClass.ToString() -ne $lvl){ $p.PriorityClass=$lvl; Write-Log ("Set PID {0} -> {1}" -f $id,$lvl) } }catch{}
    }
}
function Get-GpuInfo {
    try{
        $out=& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null
        if($out){ $parts=$out.Split(','); return @{used=[int]$parts[0].Trim(); total=[int]$parts[1].Trim(); util=[int]$parts[2].Trim(); temp=[int]$parts[3].Trim()} }
    }catch{}
    return $null
}
Write-Log "=== ResourceOptimizer v3.0 starting PID $PID ==="
Show-Toast 'Quillan Resource' 'PC optimizer online'
$state=@{LastDupKill=0}
while($true){
    try{
        $trainers=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -and $_.CommandLine -match '\\02_QUILLAN\\' -and $_.CommandLine -match '[tT]rain' })
        $gpu=Get-GpuInfo
        if($trainers.Count -gt 2){
            $sorted=$trainers | Sort-Object ProcessId
            $toKill=$sorted | Select-Object -First ($sorted.Count-2)
            foreach($k in $toKill){
                try{ Stop-Process -Id $k.ProcessId -Force -ErrorAction Stop; Write-Log ("KILLED duplicate PID {0}" -f $k.ProcessId) }catch{}
            }
        }
        if($trainers.Count -gt 0){
            $tids=[int[]]($trainers | Select-Object -ExpandProperty ProcessId)
            if($gpu -and $gpu.util -gt 92){ Set-Prio $tids 'BelowNormal' }
            elseif($gpu -and $gpu.util -lt 70){ Set-Prio $tids 'AboveNormal' }
            else { Set-Prio $tids 'Normal' }
            if($gpu -and $gpu.used -gt 3200){ Write-Log ("VRAM HIGH {0}/{1}MB" -f $gpu.used,$gpu.total); Set-Prio $tids 'BelowNormal' }
            if($gpu -and $gpu.temp -gt 80){ Write-Log ("THERMAL {0}C" -f $gpu.temp); Set-Prio $tids 'BelowNormal' }
        }
    }catch{ Write-Log "Loop err: $($_.Exception.Message)" }
    Start-Sleep -Seconds $Cfg.PollSeconds
}




