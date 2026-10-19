#!/usr/bin/env python3
"""Add MOC backlinks so files cluster by topic, not just a star around the hub."""
import os

ROOT = r'C:\Users\Admin\Quillan-Ronin'
skip_dirs = {'node_modules','__pycache__','.git','venv','venv_cuda1050','.venv','.venv-cuda','_projects'}

# Map: directory prefix -> MOC wikilink to add
moc_links = {
    'Quillan Knowledge files': '[[00 - Meta/02 - Knowledge Foundation.md]]',
    'Skills': '[[00 - Meta/04 - Skills & Capabilities.md]]',
    'Audio Engineer': '[[00 - Meta/05 - Creative Works.md]]',
    'Book Series': '[[00 - Meta/05 - Creative Works.md]]',
    'Media Template': '[[00 - Meta/05 - Creative Works.md]]',
    'Platforms': '[[00 - Meta/06 - Deployment & Platforms.md]]',
    'system prompts': '[[00 - Meta/06 - Deployment & Platforms.md]]',
    'Formal Papers': '[[00 - Meta/02 - Knowledge Foundation.md]]',
    'Software Engineer': '[[00 - Meta/01 - Core Architecture.md]]',
    'Misc': '[[00 - Meta/00 - Vault Index.md]]',
    '_dev': '[[00 - Meta/03 - Training & Model.md]]',
    '_config': '[[00 - Meta/04 - Skills & Capabilities.md]]',
}

total = 0
for dir_prefix, moc_link in moc_links.items():
    base = os.path.join(ROOT, dir_prefix)
    if not os.path.exists(base):
        continue
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if not f.endswith('.md'):
                continue
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                content = fh.read()
            # Skip if already has this MOC link (check by full path)
            if moc_link in content:
                continue
            content = content.rstrip() + f'\n- {moc_link}\n'
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(content)
            total += 1

print(f'Added MOC backlinks to {total} files')
