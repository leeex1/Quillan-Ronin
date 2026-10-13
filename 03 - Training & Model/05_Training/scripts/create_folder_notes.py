#!/usr/bin/env python3
"""Create _Index.md folder notes in each major directory."""
import os

ROOT = r'C:\Users\Admin\Quillan-Ronin'
LINK = '\n- [[system prompts/Quillan-Samurai.md]]\n- [[00 - Meta/00 - Vault Index.md]]\n'

# Directories that need folder notes
dirs = ['scripts', '_dev', 'Quillan Knowledge files', 'Formal Papers', 
        'Skills', 'Platforms', 'system prompts', 'Audio Engineer', 
        'Book Series', 'testing', 'Misc', '00 - Meta', '00 - Templates',
        'training_data', 'training_logs', 'checkpoints',
        '_projects/Ronin-Saga-Neo-Eden', '_projects/Python-Monsters']

for d in dirs:
    base = os.path.join(ROOT, d)
    if not os.path.exists(base):
        print(f'SKIP (not found): {d}')
        continue
    
    # Build listing
    entries = []
    for f in sorted(os.listdir(base)):
        fp = os.path.join(base, f)
        if os.path.isdir(fp) and not f.startswith('.'):
            entries.append(f'  - **{f}/**')
        elif os.path.isfile(fp) and not f.startswith('.'):
            entries.append(f'  - [[{d}/{f}]]')
    
    # Create the _Index.md
    content = f'# {d}\n\n'
    content += f'Folder note for `{d}/` — {len(entries)} entries.\n{LINK}\n'
    content += '\n'.join(entries) + '\n'
    
    idx_path = os.path.join(base, '_Index.md')
    with open(idx_path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'Created: {d}/_Index.md ({len(entries)} entries)')

print('\nDone!')
