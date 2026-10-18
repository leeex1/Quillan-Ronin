import json, glob, shutil, datetime
stamp = '20261013'
targets = [
 r'C:\Users\Admin\.cursor\mcp_config.json',
 r'C:\Users\Admin\.codeium\windsurf\mcp_config.json',
 r'C:\Users\Admin\.gemini\config\mcp_config.json',
 r'C:\Users\Admin\.qwen\mcp_config.json',
 r'C:\Users\Admin\.devin\mcp_config.json',
]
for p in targets:
    try:
        d = json.load(open(p))
    except Exception as e:
        print('SKIP', p, str(e)[:80])
        continue
    servers = d.get('mcpServers', d)
    if 'MemPalace' not in servers:
        servers['MemPalace'] = {'command': 'mempalace-mcp'}
        print('ADD MemPalace ->', p)
    if 'GitNexus' not in servers:
        servers['GitNexus'] = {'command': 'npx', 'args': ['gitnexus', 'mcp']}
        print('ADD GitNexus ->', p)
    d['mcpServers'] = servers
    shutil.copy(p, p + '.bak-' + stamp)
    json.dump(d, open(p, 'w'), indent=2)
    print('WROTE', p)
