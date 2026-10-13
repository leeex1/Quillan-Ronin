#!/usr/bin/env python3
"""Generate complete media index for Quillan-generated content."""
import os

ROOT = r'C:\Users\Admin\Quillan-Ronin'
DOWNLOAD = r'C:\Users\Admin\Downloads'
LINK = '\n- [[system prompts/Quillan-Samurai.md]]\n- [[00 - Meta/05 - Creative Works.md]]\n- [[Audio Engineer/_Index.md]]\n'

media_sources = {
    'Downloads/audio': ('Audio (Tracks, Freestyles)', '.mp3', '.wav', '.flac', '.m4a', '.ogg'),
    'Downloads/iamges': ('Images (AI-Generated Art)', '.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp'),
    'Downloads/vids': ('Videos (Music Videos, Clips)', '.mp4', '.mov', '.avi', '.mkv', '.webm'),
    'Main images': ('Vault Media (Branding, Logos)', '.png', '.jpg', '.jpeg', '.svg', '.gif'),
}

sections = []
for folder, (title, *exts) in media_sources.items():
    exts = exts[0]
    base = os.path.join(DOWNLOAD if folder.startswith('Downloads') else ROOT, folder.split('/')[-1])
    if not os.path.exists(base):
        base = os.path.join(ROOT, folder)
    if not os.path.exists(base):
        continue
    
    files = []
    for root, dirs, dirfiles in os.walk(base):
        for f in sorted(dirfiles):
            if any(f.lower().endswith(e) for e in exts):
                files.append(f)
    
    if not files:
        continue
    
    # Build full paths for wikilinks
    link_base = folder.replace('Downloads', 'C:/Users/Admin/Downloads')
    entries = []
    for f in files:
        entries.append(f'  - [[{link_base}/{f}]]')
    
    sections.append(f'## {title} ({len(files)} files)\n' + '\n'.join(entries))

content = '# Quillan-Generated Media Index\n\n'
content += f'Index of Quillan-generated audio, image, and video content.\n{LINK}\n\n'
content += '\n\n'.join(sections) + '\n'

path = os.path.join(ROOT, 'Quillan Generated Media Index.md')
with open(path, 'w', encoding='utf-8') as fh:
    fh.write(content)

# Count total
total = sum(1 for l in content.split('\n') if l.strip().startswith('- [['))
print(f'Created: Quillan Generated Media Index.md ({total} files indexed)')
