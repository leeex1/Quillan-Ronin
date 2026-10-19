#!/usr/bin/env python3
"""Create directory index files listing every file with wikilinks."""
import os

ROOT = r'C:\Users\Admin\Quillan-Ronin'
skip_dirs = {'node_modules','__pycache__','.git','venv','venv_cuda1050','.venv','.venv-cuda','__pycache__'}
skip_ext = {'.pyc', '.exe', '.dll', '.so', '.pyd'}

# Major directories to index
dirs = ['scripts', '_dev', 'checkpoints', 'training_data', 'training_logs',
        'Quillan Knowledge files', 'Formal Papers', 'Skills', 'Platforms',
        'system prompts', 'Audio Engineer', 'Book Series', 'testing',
        'Software Engineer', 'Misc', 'Media Template', '_config', '_projects',
        '00 - Meta', '00 - Templates']

for d in dirs:
    base = os.path.join(ROOT, d)
    if not os.path.exists(base):
        continue
    
    # Collect all files
    entries = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x not in skip_dirs]
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in skip_ext:
                continue
            rel = os.path.relpath(os.path.join(root, f), ROOT).replace('\\', '/')
            entries.append(f'  - [[{rel}]]')
    
    if not entries:
        continue
    
    # Also add the Downloads/Papers directory
    if d == 'Formal Papers':
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
