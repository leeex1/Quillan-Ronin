#!/usr/bin/env python3
"""Create directory index files listing every file with wikilinks."""
import os

ROOT = r'C:\02_QUILLAN'
skip_dirs = {'node_modules','__pycache__','.git','.gitnexus','venv','venv_cuda1050','.venv','.venv-cuda','checkpoints','.obsidian','_obsidian_backup'}
skip_ext = {'.pyc', '.exe', '.dll', '.so', '.pyd'}

# Major directories to index - canonical C:\02_QUILLAN layout 2026-10-13
dirs = ['00 - Web Map', '01 - Core Architecture', '02 - Knowledge Foundation',
        '03 - Training & Model', '04- quillan-sovereign-skills-and-swarm',
        '05 - Creative Works', '05 - Technical Implementation',
        '06 - Deployment & Platforms', '07 - Memory & LanceDB',
        '08 - Templates & Config', '09 - Projects', '10 - Formal Papers',
        'mcp', 'scripts', 'worker', 'assets', 'Misc']

for d in dirs:
    base = os.path.join(ROOT, d)
    if not os.path.exists(base):
        continue
    
    # Collect all files (depth-capped + entry cap: keeps graph hubs readable)
    entries = []
    for root, subdirs, files in os.walk(base):
        depth = os.path.relpath(root, base).count(os.sep)
        subdirs[:] = [x for x in subdirs if x not in skip_dirs]
        if depth > 1:
            subdirs[:] = []
            continue
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in skip_ext:
                continue
            rel = os.path.relpath(os.path.join(root, f), ROOT).replace('\\', '/')
            entries.append(f'  - [[{rel}]]')
            if len(entries) >= 2500:
                entries.append('  - *... truncated at 2500 links for graph performance; use search for deeper files*')
                break
        if len(entries) >= 2500:
            break
    
    if not entries:
        continue
    
    # Also add the Downloads/Papers directory
    if d == '10 - Formal Papers':
        papers_root = r'C:\Users\Admin\Downloads\Papers'
        if os.path.exists(papers_root):
            entries.append(f'\n### Downloads/Papers\n')
            for f in sorted(os.listdir(papers_root)):
                ext = os.path.splitext(f)[1].lower()
                if ext in skip_ext:
                    continue
                entries.append(f'  - [[C:/Users/Admin/Downloads/Papers/{f}]]')
    
    content = f'---\nfile_type: index\ndomain: {d}\n---\n\n# {d} File Index\n\nThis file indexes every file in `{d}/` for graph connectivity.\n\n- [[system prompts/Quillan-Samurai.md]]\n- [[00 - Meta/00 - Vault Index.md]]\n\n'
    content += '\n'.join(entries)
    
    idx_path = os.path.join(ROOT, f'{d} File Index.md')
    with open(idx_path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'Created: {d} File Index.md ({len(entries)} entries)')

print('\nDone!')
