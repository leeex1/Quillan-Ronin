#!/usr/bin/env python3
"""Create remaining _Index.md folder notes."""
import os

ROOT = r'C:\Users\Admin\Quillan-Ronin'

dirs = {
    'Main images': 'Quillan branding, logos, diagrams, AI-generated artwork',
    'Media Template': 'Content generation templates for social media, branding, marketing',
    'Software Engineer': 'Development prompts, SWE orchestrator documentation',
}

for d, desc in dirs.items():
    base = os.path.join(ROOT, d)
    if not os.path.exists(base):
        print(f'SKIP: {d}')
        continue
    
    entries = []
    for f in sorted(os.listdir(base)):
        fp = os.path.join(base, f)
        if os.path.isdir(fp):
            entries.append(f'  - **{f}/**')
        else:
            entries.append(f'  - [[{d}/{f}]]')
    
    content = f'# {d}\n\n{desc}\n\n'
    content += '- [[system prompts/Quillan-Samurai.md]]\n'
    content += '- [[00 - Meta/00 - Vault Index.md]]\n'
    content += '- [[00 - Meta/05 - Creative Works.md]]\n\n'
    content += '\n'.join(entries) + '\n'
    
    path = os.path.join(base, '_Index.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'Created: {d}/_Index.md ({len(entries)} entries)')

# Also note that venv_cuda1050 is now empty
venv_path = os.path.join(ROOT, 'venv_cuda1050', '_Index.md')
with open(venv_path, 'w', encoding='utf-8') as fh:
    fh.write('# venv_cuda1050\n\nEmpty — virtual environment was removed during disk cleanup.\n')
    fh.write('- [[system prompts/Quillan-Samurai.md]]\n')
print('Created: venv_cuda1050/_Index.md (empty, deleted)')

print('\nDone!')
