import os, json, datetime
root = r'C:\02_QUILLAN'
sectors = ['00 - Web Map','01 - Core Architecture','02 - Knowledge Foundation','03 - Training & Model','04- quillan-sovereign-skills-and-swarm','05 - Creative Works','05 - Technical Implementation','06 - Deployment & Platforms','07 - Memory & LanceDB','08 - Templates & Config','09 - Projects','10 - Formal Papers','mcp','scripts','worker','assets','Misc']
skip = {'.git','.gitnexus','.obsidian','_obsidian_backup','__pycache__','node_modules','checkpoints','.venv-mazebench','quillan_rag_db'}
total = 0
rows = []
for s in sectors:
    p = os.path.join(root, s)
    if not os.path.isdir(p):
        rows.append((s, 0, 0, 0))
        continue
    fc = dc = cc = 0
    for dp, dn, fn in os.walk(p):
        dn[:] = [d for d in dn if d not in skip and not d.startswith('venv')]
        for f in fn:
            fc += 1
            e = os.path.splitext(f)[1].lower()
            if e in ('.py','.js','.ts','.tsx','.jsx','.cpp','.h','.rs','.go'):
                cc += 1
            if e in ('.md','.txt','.json','.yaml','.yml'):
                dc += 1
    total += fc
    rows.append((s, fc, cc, dc))
now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
lines = ['# SOvereign Second Brain \u2014 Master System Topology Map','='*66,'*Autonomous Documentation Regenerated after restructure | '+now+'*','',
 '## 1. Executive System Overview','- **Total Workspace Artifacts Mapped**: **{:,} files** (top-level sectors, venv/checkpoint excluded)'.format(total),
 '- **Central Anchor Nucleus**: **Quillan C0 Core Sovereign Throne**','- **Canonical Root**: `C:\\02_QUILLAN` (old `C:\\Users\\Admin\\Quillan-Ronin` stub archived to `C:\\05_ARCHIVE\\Legacy\\Quillan-Ronin-stub-20261013`)','',
 '## 2. Sovereign Subsystem Sectors','| Sector | File Count | Code | Docs |','| :--- | :--- | :--- | :--- |']
for s, fc, cc, dc in rows:
    lines.append('| `{}` | {:,} | {} | {} |'.format(s, fc, cc, dc))
lines += ['','## 3. Restructure Notes (2026-10-13)',
 '- Staged ~3,169 git moves with rename detection (`09 - Projects/projects/05_Training/*` -> `03 - Training & Model/05_Training/*`, etc.)',
 '- Fixed `.gitnexusignore` (was ignore-everything Metamon stub -> canonical source+docs allowlist); re-ran `gitnexus analyze --force`',
 '- Updated `.obsidian/app.json` excludes + `graph.json` to 12 color groups matching canonical sectors',
 '- Regenerated 17 `* File Index.md` maps via `scripts/create_indexes.py` (ROOT now `C:\\02_QUILLAN`)',
 '- Patched `mcp/quillan-mcp-config.json` paths `09 - Projects/mcp` -> `mcp/` (JSON-validated); all 16 servers `py_compile` OK',
 '- MemPalace 3.10.0 installed via `uv tool`; palace init + mine on `C:\\02_QUILLAN`',
 '']
open(os.path.join(root,'SYSTEM_TOPOLOGY_MAP.md'),'w',encoding='utf-8').write('\n'.join(lines))
print('wrote SYSTEM_TOPOLOGY_MAP.md total=', total)
audit = {'timestamp': datetime.datetime.utcnow().timestamp(), 'time_utc': now, 'total_files_audited': total,
 'sectors': [{'name': s, 'files': fc, 'code': cc, 'docs': dc} for s,fc,cc,dc in rows],
 'notes': 'post-restructure audit 2026-10-13; misplaced-file pass superseded by git rename staging'}
json.dump(audit, open(os.path.join(root,'ORGANIZATION_AUDIT.json'),'w'), indent=2)
print('wrote ORGANIZATION_AUDIT.json')
