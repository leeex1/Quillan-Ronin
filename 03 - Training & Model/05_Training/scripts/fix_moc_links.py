#!/usr/bin/env python3
"""Add correct MOC links to all _Index.md files."""
import os

ROOT = r'C:\Users\Admin\Quillan-Ronin'

# Directory -> correct MOC link
moc_links = {
    'Audio Engineer': '[[00 - Meta/05 - Creative Works.md]]',
    'Book Series': '[[00 - Meta/05 - Creative Works.md]]',
    'Formal Papers': '[[00 - Meta/02 - Knowledge Foundation.md]]',
    'Platforms': '[[00 - Meta/06 - Deployment & Platforms.md]]',
    'Quillan Knowledge files': '[[00 - Meta/02 - Knowledge Foundation.md]]',
    'Skills': '[[00 - Meta/04 - Skills & Capabilities.md]]',
    'Software Engineer': '[[00 - Meta/01 - Core Architecture.md]]',
    '_dev': '[[00 - Meta/03 - Training & Model.md]]',
    'checkpoints': '[[00 - Meta/03 - Training & Model.md]]',
    'scripts': '[[00 - Meta/03 - Training & Model.md]]',
    'system prompts': '[[00 - Meta/06 - Deployment & Platforms.md]]',
    'testing': '[[00 - Meta/03 - Training & Model.md]]',
    'training_data': '[[00 - Meta/03 - Training & Model.md]]',
    'training_logs': '[[00 - Meta/03 - Training & Model.md]]',
    'Main images': '[[00 - Meta/05 - Creative Works.md]]',
    'Media Template': '[[00 - Meta/05 - Creative Works.md]]',
    'Misc': '[[00 - Meta/00 - Vault Index.md]]',
    '_config': '[[00 - Meta/04 - Skills & Capabilities.md]]',
    '_projects': '[[00 - Meta/00 - Vault Index.md]]',
}

count = 0
for d, moc in moc_links.items():
    idx_path = os.path.join(ROOT, d, '_Index.md')
    if not os.path.exists(idx_path):
        # Create it with basic content
        entries = []
        base = os.path.join(ROOT, d)
        if os.path.exists(base):
            for f in sorted(os.listdir(base)):
                if f == '_Index.md':
                    continue
                fp = os.path.join(base, f)
                if os.path.isdir(fp):
                    entries.append(f'  - **{f}/**')
                else:
                    entries.append(f'  - [[{d}/{f}]]')
        content = f'# {d}\n\n- [[system prompts/Quillan-Samurai.md]]\n- {moc}\n- [[00 - Meta/00 - Vault Index.md]]\n\n' + '\n'.join(entries) + '\n'
        with open(idx_path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        print(f'CREATED: {d}/_Index.md')
        count += 1
        continue

    with open(idx_path, 'r', encoding='utf-8') as fh:
        content = fh.read()
    
    # Check if MOC link is already present (by MOC number)
    moc_num = moc.split('/')[1].split(' -')[0].strip()
    if f'[[00 - Meta/{moc_num}' in content:
        print(f'OK:      {d}/_Index.md (has MOC {moc_num})')
        continue
    
    # Add MOC link
    content = content.rstrip() + f'\n- {moc}\n'
    with open(idx_path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'FIXED:   {d}/_Index.md (added MOC {moc_num})')
    count += 1

print(f'\nDone! {count} files updated.')
