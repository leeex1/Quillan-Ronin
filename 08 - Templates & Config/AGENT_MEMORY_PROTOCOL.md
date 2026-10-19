# Agent Memory Protocol — quillan.db (shared, all agents, all harnesses)
The SQLite MCP chain is our universal toolkit. `C:\02_QUILLAN\quillan.db` is our shared working memory.
MemPalace is the deep archive (currently read-only: a peer writer holds the lock). The DB is the writable layer.

## Session start (every agent, every session)
1. Read `memory_notes` (all rows), latest 5 `sessions`, latest 10 `decisions`, all open `todo`, latest 10 `attempts_log`.
2. Set your agent name and use it on every row you write.

## As you work
- **Decisions** -> INSERT into `decisions` (session_id, agent, topic, decision).
- **Tried something?** -> INSERT into `attempts_log` with result ok/fail. FAILURES MUST include `how_failed` + `lesson`. This is the most valuable table.
- **State changes** (MCP up/down, index rebuilt) -> INSERT OR REPLACE into `mcp_health`.
- ** durable facts** -> INSERT OR REPLACE into `memory_notes` with author=<you>.
- **Todo** -> keep statuses current (open/doing/done/blocked).
- History is append-only. Never DELETE another agent's rows.

## Bootstrap (first time only)
`python C:\02_QUILLAN\scripts\init_quillan_db.py` then enable the SQLite MCP server and restart the harness.
