#!/usr/bin/env python3
"""Init + seed C:/02_QUILLAN/quillan.db - SHARED working memory for ALL agents on this box.
Run:  python C:/02_QUILLAN/scripts/init_quillan_db.py
Idempotent: safe to re-run; seeds use INSERT OR REPLACE / existence checks.
Conforms to C5-ECHO agent spec (01 - Core Architecture/agents/c5-echo.agent.md):
verbatim recall, verified-vs-reconstruction labels, contradictions flagged, never invent.
"""
import sqlite3, datetime
DB = r'C:\02_QUILLAN\quillan.db'
now = datetime.datetime.now().isoformat(timespec='seconds')
con = sqlite3.connect(DB)
c = con.cursor()
c.executescript('''
CREATE TABLE IF NOT EXISTS agents(agent TEXT PRIMARY KEY, role TEXT, last_seen TEXT);
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, agent TEXT, started TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY, session_id INTEGER, agent TEXT, topic TEXT, decision TEXT, verified TEXT DEFAULT 'verified', decided_at TEXT);
CREATE TABLE IF NOT EXISTS todo(id INTEGER PRIMARY KEY, item TEXT, status TEXT, owner TEXT, updated_at TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS attempts_log(id INTEGER PRIMARY KEY, agent TEXT, tried_at TEXT, action TEXT, result TEXT, how_failed TEXT, lesson TEXT);
CREATE TABLE IF NOT EXISTS memory_notes(topic TEXT PRIMARY KEY, body TEXT, verified TEXT DEFAULT 'verified', updated_at TEXT, author TEXT);
CREATE TABLE IF NOT EXISTS contradictions(id INTEGER PRIMARY KEY, agent TEXT, flagged_at TEXT, claim_a TEXT, claim_b TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS file_index(path TEXT PRIMARY KEY, sector TEXT, indexed_at TEXT);
CREATE TABLE IF NOT EXISTS mcp_health(server TEXT PRIMARY KEY, status TEXT, checked_at TEXT, note TEXT);
''')
c.execute('INSERT OR REPLACE INTO agents(agent, role, last_seen) VALUES (?,?,?)',
          ('opencode', 'primary cleanup + memory keeper', now))
c.execute('INSERT OR REPLACE INTO agents(agent, role, last_seen) VALUES (?,?,?)',
          ('c5-echo', 'Council memory node: verbatim recall, verified-vs-reconstruction labels, contradiction flags. Keeper of this DB per AGENT_MEMORY_PROTOCOL.md.', now))
notes = {
'repo_canonical': (r'Canonical root C:\02_QUILLAN. Old stub archived to C:/05_ARCHIVE/Legacy. C:\02_QUILLAN\mcp is a JUNCTION to 09 - Projects/mcp, so both spellings resolve.', 'verified', 'opencode'),
'sectors': ('17 top-level sectors, ~75.8k files (venv/checkpoints excluded).', 'verified', 'opencode'),
'git_state': ('PUSHED 488b2b03 verified over API; GitHub mirrors local consolidated tree.', 'verified', 'opencode'),
'index_state': ('GitNexus 02_QUILLAN needs analyze --force. MemPalace 8k drawers. QuillanRAG healthy. LocalRAG OFF (redundant).', 'verified', 'opencode'),
'mcp_policy': ('REMOVED CodeMemory/CocoIndex/ComfyUI/brave-search. SQLite unblocked by this DB. GitHub needs token.', 'verified', 'opencode'),
'echo_keeper': ('Per C5-ECHO spec: verbatim recall; label verified vs reconstruction; flag contradictions; never invent.', 'verified', 'opencode'),
'how_we_use_this_db': ('Session start: read notes+sessions+decisions+todo+attempts. Append as you work with agent=<name>.', 'verified', 'opencode'),
}
for topic, (body, verified, author) in notes.items():
    c.execute('INSERT OR REPLACE INTO memory_notes(topic, body, verified, updated_at, author) VALUES (?,?,?,?,?)', (topic, body, verified, now, author))
c.execute('INSERT OR IGNORE INTO sessions(id, agent, started, summary) VALUES (1,?,?,?)',
          ('opencode', '2026-10-13', 'Restructure cleanup + push 488b2b03 verified.'))
decisions = [
    ('git', 'Stage-review-push cycle; push verified at 488b2b03.', 'verified'),
    ('memory', 'quillan.db is shared agent working memory.', 'verified'),
    ('memory', 'C5-ECHO is designated keeper; schema conforms to c5-echo.agent.md.', 'verified'),
]
for topic, decision, verified in decisions:
    if c.execute('SELECT id FROM decisions WHERE topic=? AND decision=?', (topic, decision)).fetchone() is None:
        c.execute('INSERT INTO decisions(session_id, agent, topic, decision, verified, decided_at) VALUES (1,?,?,?,?,?)', ('opencode', topic, decision, verified, now))
con.commit()
print('notes:', c.execute('SELECT COUNT(*) FROM memory_notes').fetchone()[0],
      '| todos:', c.execute('SELECT COUNT(*) FROM todo').fetchone()[0],
      '| agents:', c.execute('SELECT COUNT(*) FROM agents').fetchone()[0])
con.close()
