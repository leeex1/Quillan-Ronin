cd "C:\02_QUILLAN\09 - Projects\mcp"
$servers = Get-ChildItem -Directory | Where-Object { Test-Path "$_\server.py" }
foreach ($s in $servers) {
    $name = $s.Name
    $result = python -m py_compile "$s\server.py" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK $name"
    } else {
        Write-Host "FAIL $name: $result"
    }
}