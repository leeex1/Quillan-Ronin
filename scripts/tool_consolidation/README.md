# `scripts/tool_consolidation/`

Safe subset of PASS 2 tool consolidation (audit #886 / coordinator brief #896).

## Files

| File | Purpose |
|---|---|
| `namespace_registry.py` | Single source of truth: 17 session namespaces, 9 route profiles, all 54 legacy tool specs, preserved-name tables. |
| `canonical.py` | The three canonical tools: `mcp_session_replay`, `mcp_session_list`, `mcp_route`. Pure/read-only. |
| `shims.py` | Namespaced shim for every legacy name + `resolve(name)` / `is_legacy_name(name)`. |
| `verify.py` | Empirical verifier: every legacy name resolves; counts match audit; dangerous strings intact; canonical tools execute. Exit 0/1. |
| `DEPRECATION.md` | 90-day / one-minor-release policy and migration table. |

## Quick start

```bat
python C:\02_QUILLAN\scripts\tool_consolidation\verify.py
```

```python
from tool_consolidation import resolve, mcp_route

resolve("audio_session_replay")("sid")           # warns; namespace frozen to 'audio'
mcp_route("council", {"k": 1}, source="rag")     # canonical
```

## Facts (verified against files, not assumed)

- Zero exact duplicate tools across the 19 original servers (audit #886).
- `session_replay` x17 (13 byte-identical), `session_list` x17, cross-MCP route templates x9 (orochi's differs → excluded), 3 pure aliases.
- `prism_decompose` has two different signatures → kept namespaced, never merged.
- 6 thinking_engine `name=` overrides preserved, including typos `Genuis_level` / `adversarial_dual_council_diliberation`.
- Live MCP tool count stays **539**; this package adds new modules and touches no live config.
