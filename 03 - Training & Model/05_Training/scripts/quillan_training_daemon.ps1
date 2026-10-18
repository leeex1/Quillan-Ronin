# ============================================================
# QUILLAN TRAINING DAEMON v3.0 - KeepAlive until Done
# Single purpose: Keep training running until steps complete
# - Adopts trainers, watches checkpoint progress
# - Resurrects ONLY with same cmdline, exponential backoff
# - Never touches priorities (Resource Daemon does that)
# - Log: training_logs/training_daemon.log
# ============================================================
$Cfg = @{
    PollSeconds   = 10
    QuillanRoot   = 'C:\02_QUILLAN'
    LogFile       = 'C:\02_QUILLAN\training_logs\training_daemon.log'
    MaxLogBytes   = 524288
    StallMinutes  = 20
    AutoResurrect = $true
    MaxResurrect  = 999
    TargetSteps   = 15000
}
$mutex = New-Object System.Threading.Mutex($true, 'Global\QuillanTrainingDaemon', [ref]$null)
if (-not $created) { exit 0 }
function Write-Log([string]$msg){
    try{
        if((Test-Path $Cfg.LogFile) -and ((Get-Item $Cfg.LogFile).Length -gt $Cfg.MaxLogBytes)){
            $b=[System.IO.File]::ReadAllBytes($Cfg.LogFile); [System.IO.File]::WriteAllBytes($Cfg.LogFile, $b[($b.Length/2)..($b.Length-1)])
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
function Get-Trainers {
    $list=@()
    foreach($p in (Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)){
        if(-not $p.CommandLine){continue}
        if($p.CommandLine -match '\\02_QUILLAN\\' -and $p.CommandLine -match '[tT]rain'){
            $list+=[pscustomobject]@{Pid=$p.ProcessId; Cmdline=$p.CommandLine}
        }
    }
    return $list
}
function Split-Launch([string]$c){
    if($c.StartsWith('"')){ $e=$c.IndexOf('"',1); return $c.Substring(1,$e-1), $c.Substring($e+1).Trim() }
    $sp=$c.IndexOf(' '); if($sp -lt 0){return $c,''}; return $c.Substring(0,$sp), $c.Substring($sp+1)
}
function Get-CkptStep {
    try{
        $ck="C:\02_QUILLAN\checkpoints\checkpoints_oni\quillan_oni_latest.pt"
        if(Test-Path $ck){
            $py="C:\02_QUILLAN\venv_oni_gpu\Scripts\python.exe"
            $out=& $py -c "import torch; ck=torch.load(r'C:\02_QUILLAN\checkpoints\checkpoints_oni\quillan_oni_latest.pt',map_location='cpu'); print(ck.get('step',0))" 2>$null
            return [int]$out.Trim()
        }
    }catch{}
    return 0
}
Write-Log "=== TrainingDaemon v3.0 starting PID $PID ==="
Show-Toast 'Quillan Training' 'KeepAlive daemon online - watching training'
$state=@{LastCmd=''; ResurrectCount=0; Offsets=@{}}
while($true){
    try{
        $trainers=Get-Trainers
        foreach($t in $trainers){
            $k=[string]$t.Pid
            if(-not $state.Offsets.ContainsKey($k)){
                Write-Log ("Adopted trainer PID {0}: {1}" -f $t.Pid, $t.Cmdline.Substring(0,[Math]::Min(160,$t.Cmdline.Length)))
                $state.Offsets[$k]=0
                if($t.Cmdline -notmatch '--help'){ $state.LastCmd=$t.Cmdline }
            }
        }
        $tracked=@($state.Offsets.Keys)
        $died=($tracked.Count -gt 0 -and $trainers.Count -eq 0)
        $dead=@($tracked | Where-Object { $trainers.Pid -notcontains [int]$_ })
        foreach($d in $dead){ Write-Log "Trainer PID $d exited"; $state.Offsets.Remove($d) }
        $ckptStep=Get-CkptStep
        $done=($ckptStep -ge $Cfg.TargetSteps)
        if($died -and -not $done -and $Cfg.AutoResurrect -and $state.LastCmd -ne '' -and $state.ResurrectCount -lt $Cfg.MaxResurrect){
            $state.ResurrectCount++
            $delay=30 * $state.ResurrectCount
            if($delay -gt 120){ $delay=120 }
            Write-Log ("Resurrection {0}/{1} in {2}s (step {3}/{4}): {5}" -f $state.ResurrectCount,$Cfg.MaxResurrect,$delay,$ckptStep,$Cfg.TargetSteps,$state.LastCmd.Substring(0,[Math]::Min(120,$state.LastCmd.Length)))
            Show-Toast 'Quillan Training' ("Trainer died at step {0}. Relaunch in {1}s (#{2})" -f $ckptStep,$delay,$state.ResurrectCount)
            Start-Sleep -Seconds $delay
            # Re-check nothing started while we waited
            if((Get-Trainers).Count -eq 0){
                try{
                    $exe,$rest=Split-Launch $state.LastCmd
                    # Ensure single instance: set env for expandable_segments
                    $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
                    Start-Process -FilePath $exe -ArgumentList $rest -WorkingDirectory $Cfg.QuillanRoot -WindowStyle Hidden
                    Write-Log "Relaunched"
                }catch{ Write-Log "Relaunch failed: $($_.Exception.Message)" }
            } else { Write-Log "Skipped resurrect - trainer already back" }
        } elseif($done){
            if($died){ Write-Log "TARGET REACHED step $ckptStep - no resurrect"; Show-Toast 'Quillan Training DONE' ("Reached {0} steps!" -f $ckptStep) }
        }
    }catch{ Write-Log "Loop err: $($_.Exception.Message)" }
    Start-Sleep -Seconds $Cfg.PollSeconds
}

