# Ghost Process Janitor for MCP Servers
# Safely terminates orphaned and unattached MCP processes without touching active training or system daemons.

param(
    [switch]$Force = $false
)

Write-Host "Scanning for orphaned MCP processes..." -ForegroundColor Cyan

# PIDs to explicitly preserve
$protectedKeywords = @(
    "launch_sovereign_sft",
    "quillan_governor_daemon"
)

$mcpKeywords = @(
    "09 - Projects\mcp",
    "windows_mcp",
    "mempalace-mcp",
    "duckduckgo-mcp",
    "mcp-server",
    "server-sequential-thinking",
    "server-filesystem",
    "server-memory",
    "server-puppeteer",
    "mcp-local-rag",
    "gitnexus",
    "chrome-devtools-mcp",
    "@playwright\mcp"
)

$allProcs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' or Name = 'node.exe'"
$killed = 0
$freedMB = 0

# Active parent clients
$activeHosts = (Get-Process agy, opencode-cli -ErrorAction SilentlyContinue).Id

foreach ($proc in $allProcs) {
    $cmd = $proc.CommandLine
    if (-not $cmd) { continue }

    # Skip protected training / daemon jobs
    $isProtected = $false
    foreach ($pk in $protectedKeywords) {
        if ($cmd -like "*$pk*") { $isProtected = $true; break }
    }
    if ($isProtected) { continue }

    # Check if this is an MCP process
    $isMcp = $false
    foreach ($mk in $mcpKeywords) {
        if ($cmd -like "*$mk*") { $isMcp = $true; break }
    }
    if (-not $isMcp) { continue }

    # Check if orphaned (parent is dead or parent is not an active IDE host)
    $parentAlive = Get-Process -Id $proc.ParentProcessId -ErrorAction SilentlyContinue
    $isOrphan = ($null -eq $parentAlive) -or ($proc.ParentProcessId -notin $activeHosts)

    if ($isOrphan -or $Force) {
        try {
            $p = Get-Process -Id $proc.ProcessId -ErrorAction Stop
            $mem = [math]::Round($p.WorkingSet64 / 1MB, 1)
            $p | Stop-Process -Force -ErrorAction Stop
            $killed++
            $freedMB += $mem
            Write-Host "  [CLEANED] PID $($proc.ProcessId) ($($proc.Name)) - Freed ${mem}MB" -ForegroundColor DarkGray
        } catch {
            # process may have already exited
        }
    }
}

Write-Host "Cleanup complete: $killed ghost processes terminated. Freed ${freedMB}MB RAM." -ForegroundColor Green

