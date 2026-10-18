# Repairs Qwen desktop app's forgotten MCP boot-load.
# Run this again if a Qwen update replaces app.asar (symptom: MCP servers dead after restart).
# Usage: powershell -ExecutionPolicy Bypass -File fix-qwen-mcp-boot.ps1

$ErrorActionPreference = 'Stop'
$asar = "C:\Program Files\Qwen\resources\app.asar"
$work = "$env:TEMP\qwen-asar-fix"

Write-Host '[1/5] extracting...'
npx -y @electron/asar extract $asar "$work\extracted"

$main = "$work\extracted\out\main\index.js"
$c = Get-Content $main -Raw

if ($c.Contains('settings.get("mcp_config")')) {
    Write-Host '[OK] already patched - nothing to do'
    exit 0
}

Write-Host '[2/5] injecting boot-load...'
$anchor = 'ipcMain.handle("mcp_client_update_config", mcpClientUpdateConfig);'
if (-not $c.Contains($anchor)) { throw 'anchor not found - app version changed, manual review needed' }
$inject = $anchor + @'

  try {
    const __saved = settings.get("mcp_config");
    if (__saved && typeof __saved === "object" && Object.keys(__saved).length) {
      console.log("[boot] restoring saved MCP config:", Object.keys(__saved).join(", "));
      mcpServer.setMCPServers(adaptConfig(__saved));
    }
  } catch (e) { console.error("[boot] MCP restore failed:", e); }
'@
$c = $c.Replace($anchor, $inject)
Set-Content $main $c -Encoding UTF8

Write-Host '[3/5] backing up current asar...'
Copy-Item $asar "$asar.pre-fix.bak" -Force

Write-Host '[4/5] repacking...'
npx -y @electron/asar pack "$work\extracted" $asar

Write-Host '[5/5] done. Restart Qwen - MCP servers will now survive restarts.'
