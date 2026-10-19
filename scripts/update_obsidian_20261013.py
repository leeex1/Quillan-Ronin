import json
a = json.load(open(r'C:\02_QUILLAN\.obsidian\app.json'))
a['excludedFolders'] = ['.git','.gitnexus','__pycache__','.pytest_cache','checkpoints','.venv-mazebench','.venv*','venv*','node_modules','mcp/*/node_modules','quillan_rag_db','_obsidian_backup']
a['templateFolder'] = '08 - Templates & Config'
json.dump(a, open(r'C:\02_QUILLAN\.obsidian\app.json','w'), indent=2)
print('app.json updated')
g = json.load(open(r'C:\02_QUILLAN\.obsidian\graph.json'))
g['colorGroups'] = [
 {'color':'#f2f0e8','query':'path:01 - Core Architecture'},
 {'color':'#a371f7','query':'path:02 - Knowledge Foundation'},
 {'color':'#49e0da','query':'path:03 - Training'},
 {'color':'#f6b84b','query':'path:04- quillan-sovereign-skills'},
 {'color':'#ff3d81','query':'path:05 - Creative Works'},
 {'color':'#3fb950','query':'path:06 - Deployment'},
 {'color':'#a371f7','query':'path:07 - Memory'},
 {'color':'#e6edf3','query':'path:08 - Templates'},
 {'color':'#f6b84b','query':'path:09 - Projects'},
 {'color':'#79c0ff','query':'path:10 - Formal Papers'},
 {'color':'#7ee787','query':'path:00 - Web Map'},
 {'color':'#ff9d5c','query':'tag:#ship'},
]
json.dump(g, open(r'C:\02_QUILLAN\.obsidian\graph.json','w'))
print('graph.json updated:', len(g['colorGroups']), 'groups')
