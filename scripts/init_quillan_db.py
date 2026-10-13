#!/usr/bin/env python3
"""Init + seed C:\02_QUILLAN\quillan.db — SHARED working memory for ALL agents on this box.
Run:  python C:\02_QUILLAN\scripts\init_quillan_db.py
Idempotent: safe to re-run; seeds use INSERT OR REPLACE / INSERT OR IGNORE.
Conventions: every row carries WHO wrote it (agent) and WHEN. Append, don't overwrite history.
Once created, enable the SQLite MCP server so every harness (opencode/cursor/claude/...)
gets query tools over this DB. See AGENT_MEMORY_PROTOCOL.md next to the vault root docs.
"""
import sqlite3, datetime
DB = r'C:\02_QUILLAN\quillan.db'
now = datetime.datetime.now().isoformat(timespec='seconds')
con = sqlite3.connect(DB)
c = con.cursor()
c.executescript('''
CREATE TABLE IF NOT EXISTS agents(agent TEXT PRIMARY KEY, role TEXT, last_seen TEXT);
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, agent TEXT, started TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY, session_id INTEGER, agent TEXT, topic TEXT, decision TEXT, decided_at TEXT);
CREATE TABLE IF NOT EXISTS todo(id INTEGER PRIMARY KEY, item TEXT, status TEXT, owner TEXT, updated_at TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS attempts_log(id INTEGER PRIMARY KEY, agent TEXT, tried_at TEXT, action TEXT, result TEXT, how_failed TEXT, lesson TEXT);
CREATE TABLE IF NOT EXISTS memory_notes(topic TEXT PRIMARY KEY, body TEXT, updated_at TEXT, author TEXT);
CREATE TABLE IF NOT EXISTS file_index(path TEXT PRIMARY KEY, sector TEXT, indexed_at TEXT);
CREATE TABLE IF NOT EXISTS mcp_health(server TEXT PRIMARY KEY, status TEXT, checked_at TEXT, note TEXT);
''')
c.execute('INSERT OR REPLACE INTO agents(agent, role, last_seen) VALUES (?,?,?)',
          ('opencode', 'primary cleanup + memory keeper', now))
notes = {
'repo_canonical': ('Canonical root C:\02_QUILLAN. Old C:\Users\Admin\Quillan-Ronin stub archived to C:\05_ARCHIVE\Legacy\Quillan-Ronin-stub-20261013 (held 1.5GB ckpt). C:\02_QUILLAN\mcp is a JUNCTION to 09 - Projects\mcp, so both spellings resolve to the same files.', 'opencode'),
'sectors': ('17 top-level sectors: 00 - Web Map, 01 - Core Architecture, 02 - Knowledge Foundation, 03 - Training & Model, 04- quillan-sovereign-skills-and-swarm, 05 - Creative Works, 05 - Technical Implementation, 06 - Deployment & Platforms, 07 - Memory & LanceDB, 08 - Templates & Config, 09 - Projects, 10 - Formal Papers, mcp, scripts, worker, assets, Misc. ~75.8k files (venv/checkpoints excluded).', 'opencode'),
'git_state': ('Branch main ahead 18. ~3169 moves staged with rename detection (09-Projects/projects/05_Training -> 03-Training & Model etc). LEFT UNCOMMITTED pending user review.', 'opencode'),
'index_state': ('GitNexus 02_QUILLAN 0 nodes: .gitnexusignore was Metamon stub ignoring everything; replaced with canonical allowlist + venv excludes, needs analyze --force re-run. MemPalace 3.10.0, 02_quillan wing ~8k drawers (server currently read-only: peer writer holds lock). QuillanRAG 360 chunks healthy. ThinkingEngine 368 skills healthy. LocalRAG live but 0 docs, needs ingest or stays off.', 'opencode'),
'mcp_policy': ('REMOVED: CodeMemory, CocoIndex (dead since 2026-09-04), ComfyUI (too heavy for this box), brave-search (WebSearch covers). DISABLED-OK: SQLite until this DB exists, GitHub until token set, ThreeJSDevtools (needs dev server+tab), LocalRAG optional (QuillanRAG covers). ADDED to all 8 harness configs: MemPalace + GitNexus.', 'opencode'),
'how_we_use_this_db': ('Session start: read memory_notes + recent sessions/decisions/todo/attempts. As you work: INSERT decisions, todo updates, attempts (especially FAILURES with how_failed + lesson), mcp_health changes. Always set agent=<your-name>. Append history, never rewrite it.', 'opencode'),
}
for topic, (body, author) in notes.items():
    c.execute('INSERT OR REPLACE INTO memory_notes(topic, body, updated_at, author) VALUES (?,?,?,?)', (topic, body, now, author))
c.execute('INSERT OR IGNORE INTO sessions(id, agent, started, summary) VALUES (1,?,?,?)',
          ('opencode', '2026-10-13', 'Restructure cleanup: staged git moves, fixed gitnexusignore, archived stub, installed MemPalace, patched 8 MCP configs, regenerated 17 file indexes + Obsidian + topology.'))
todos = [
    ('Review + commit staged 3169 git moves', 'open', 'user', 'Rename-detected staging done 2026-10-13; needs human review before commit.'),
    ('Re-run gitnexus analyze --force in C:\02_QUILLAN', 'open', 'any-agent', 'Cancelled by restart; fixed ignore file on disk. Verify nodes > 0 after.'),
    ('LocalRAG: ingest C:\02_QUILLAN or confirm stays off', 'open', 'any-agent', 'Server live, 0 docs. QuillanRAG already covers retrieval.'),
    ('Set GITHUB_TOKEN + enable GitHub MCP', 'open', 'user', 'Needs human-created PAT; then env in harness configs.'),
    ('Run this init script + enable SQLite MCP', 'doing', 'user', 'One command, then flip enabled:true + restart harness.'),
    ('Watch MemPalace mine completion (peer writer lock)', 'doing', 'any-agent', 'Wing at ~8k drawers; diary writes rejected until peer releases.'),
]
for item, status, owner, note in todos:
    if c.execute('SELECT id FROM todo WHERE item=?', (item,)).fetchone() is None:
        c.execute('INSERT INTO todo(item, status, owner, updated_at, notes) VALUES (?,?,?,?,?)', (item, status, owner, now, note))
attempts = [
    ('opencode', 'gitnexus analyze', 'fail', '0 nodes: .gitnexusignore was Metamon stub (/* + !Metamon-Survival-Rogue) ignoring everything.', 'Always read the ignore file when an index comes back empty.'),
    ('opencode', 'gitnexus analyze --force', 'fail', 'Choked on 09-Projects/projects/venv_oni_cu126 torch files; worker timeout, sequential fallback; then killed by restart.', 'Excluded venv_oni*/**, re-run needed.'),
    ('opencode', 'spider_bridge.py --help', 'fail', 'It is a stdio MCP server: --help starts serving and hangs. Expected behavior, not a bug.', 'Probe MCP servers with timed jobs + init handshake, not --help.'),
    ('opencode', 'opencode.jsonc comment-strip validation', 'fail', 'Naive // regex ate the https:// inside the $schema string.', 'Strip comments only outside string literals (placeholder method).'),
    ('opencode', 'mempalace_diary_write', 'fail', 'Rejected: peer MCP writer active, server read-only for mutations.', 'quillan.db is the writable layer until lock releases.'),
    ('opencode', 'LocalRAG status check', 'ok', '', 'Live with 0 docs; needs ingest.'),
    ('opencode', 'SQLite MCP enable', 'blocked', 'quillan.db is 0 bytes; server exits on open.', 'This script creates a valid seeded DB.'),
]
for agent, action, result, how_failed, lesson in attempts:
    if c.execute('SELECT id FROM attempts_log WHERE agent=? AND action=? AND tried_at=?', (agent, action, now)).fetchone() is None:
        c.execute('INSERT INTO attempts_log(agent, tried_at, action, result, how_failed, lesson) VALUES (?,?,?,?,?,?)',
                  (agent, now, action, result, how_failed, lesson))
health = [
    ('QuillanRAG', 'healthy', '360 chunks, NIM embed, live'),
    ('ThinkingEngine', 'healthy', '368 skills, HEALTHY audit'),
    ('SamuraiEngine', 'healthy', 'spawns clean via stdio'),
    ('RoninBuddy', 'healthy', 'tools live in session'),
    ('MemPalace', 'healthy', '3.10.0, 8k+ drawers; mutations read-only until peer lock releases'),
    ('GitNexus', 'degraded', 'MCP live but 02_QUILLAN index is 0 nodes; force re-run pending'),
    ('LocalRAG', 'empty', 'connected, 0 docs; needs ingest or leave off'),
    ('SQLite', 'blocked', '0-byte db; this script unblocks it'),
    ('GitHub', 'blocked', 'needs GITHUB_TOKEN env'),
    ('CodeMemory', 'removed', 'dead since 2026-09-04'),
    ('CocoIndex', 'removed', 'needs flow setup; RAG covers'),
    ('ComfyUI', 'removed', 'too heavy for this box'),
    ('brave-search', 'removed', 'bogus URL; WebSearch covers'),
    ('ThreeJSDevtools', 'standby', 'needs dev server + open tab; enable per-session only'),
]
for server, status, note in health:
    c.execute('INSERT OR REPLACE INTO mcp_health(server, status, checked_at, note) VALUES (?,?,?,?)', (server, status, now, note))
con.commit()
print('quillan.db ready:', DB,
      '| notes:', c.execute('SELECT COUNT(*) FROM memory_notes').fetchone()[0],
      '| todo:', c.execute('SELECT COUNT(*) FROM todo').fetchone()[0],
      '| attempts:', c.execute('SELECT COUNT(*) FROM attempts_log').fetchone()[0])
con.close()
