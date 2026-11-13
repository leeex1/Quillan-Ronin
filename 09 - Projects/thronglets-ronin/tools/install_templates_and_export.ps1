# Install Godot export templates (drop-in for the .tpz) and build dist/RoninThronglets.exe
$ErrorActionPreference = 'Stop'
$dest = "$env:APPDATA\Godot\export_templates\4.7.2.stable"
New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item 'C:\02_QUILLAN\09 - Projects\thronglets-ronin\tools\export_templates.tpz' "$env:TEMP\export_templates.zip" -Force
# .tpz is just a zip with a top-level 'templates' folder
Expand-Archive "$env:TEMP\export_templates.zip" "$env:TEMP\tpl_extract" -Force
$tpl = Get-ChildItem "$env:TEMP\tpl_extract" -Directory | Select-Object -First 1
Copy-Item (Join-Path $tpl.FullName '*') $dest -Recurse -Force
Write-Output "templates installed to $dest"
New-Item -ItemType Directory -Force 'C:\02_QUILLAN\09 - Projects\thronglets-ronin\dist' | Out-Null
& 'C:\02_QUILLAN\09 - Projects\thronglets-ronin\tools\Godot_v4.7.2-stable_win64.exe' --headless --path 'C:\02_QUILLAN\09 - Projects\thronglets-ronin\game' --export-release 'Windows Desktop' 'C:\02_QUILLAN\09 - Projects\thronglets-ronin\dist\RoninThronglets.exe'
Write-Output '--- dist ---'
Get-ChildItem 'C:\02_QUILLAN\09 - Projects\thronglets-ronin\dist' | Select-Object Name,Length