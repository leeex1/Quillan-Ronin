#!/usr/bin/env python3
"""Add wikilink footer to all .md files in sparse vault directories."""
import os

ROOT = r'C:\Users\Admin\Quillan-Ronin'
FOOTER = '\n- [[system prompts/Quillan-Samurai.md]]\n'

# All directories except already-connected ones
dirs = [
    'Skills', 'Audio Engineer', 'Book Series', 'Software Engineer',
    'Misc', 'testing', 'Formal Papers', 'Media Template', 'Platforms',
    'system prompts', 'training_logs',
    '_config', '_projects',
]

skip_dirs = {'node_modules', '__pycache__', '.git', '.obsidian', 'venv_cuda1050', '.venv', '.venv-cuda'}
total = 0
for d in dirs:
    base = os.path.join(ROOT, d)
    for root, dirs, files in os.walk(base):
        # Skip third-party/cache directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if not f.endswith('.md'):
                continue
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                content = fh.read()
            # Skip if already has the link
            if 'Quillan-Samurai.md' in content:
                continue
            # Skip if already has a link section at end
            content = content.rstrip() + FOOTER
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(content)
            total += 1
            if total % 50 == 0:
                print(f'  {total} files updated...')

print(f'Done! Updated {total} files with wikilink to Quillan-Samurai.md')
