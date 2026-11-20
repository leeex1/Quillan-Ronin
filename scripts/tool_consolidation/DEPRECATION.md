# Tool Consolidation — Deprecation Notice (PASS 2)

**Scope:** audit #886 / coordinator brief #896 — safe subset only.
**Status:** NEW shim modules only. No existing `server.py`, facade, or live MCP config was edited.

## Policy

| Field | Value |
|---|---|
| Deprecated since | `5.4.0-ONI` |
| Window | **90 days** AND **one minor release**, whichever is LATER |
| Earliest removal target | `5.5.0-ONI` (not before the 90-day mark) |
| Runtime signal | `DeprecationWarning` emitted by every merged shim |
| Live tool count | **unchanged (539)** — shims are not registered in the facades |

## What is consolidated (43 legacy names → 3 canonical interfaces)

| Legacy family | Count | Canonical interface | Shim namespace argument |
|---|---:|---|---|
| `*_session_replay` | 17 | `mcp_session_replay(session_id, namespace=...)` | original server namespace |
| `*_session_list` | 17 | `mcp_session_list(namespace=...)` | original server namespace |
| cross-MCP `*_route` templates | 9 | `mcp_route(target_mcp, payload, source=...)` | original server namespace |

Per-namespace data (session dirs, route note text, default targets) is preserved in
`namespace_registry.py`, so a shim is behaviorally identical to the tool it replaces.

## What is deliberately NOT consolidated

| Name(s) | Reason | Treatment |
|---|---|---|
| `prism_decompose` | Same exposed name, **different signature**: `quillan_prism.prism_decompose(task, context=None)` vs `quillan_thinking_engine.prism_decompose(text)` | Two namespaced callables `prism_decompose@quillan_prism` and `prism_decompose@quillan_thinking_engine`. **No bare `prism_decompose` shim exists.** |
| `Deep_reasoning`, `deep_reasoning`, `Genuis_level`, `genius_level`, `adversarial_dual_council_diliberation`, `adversarial_dual_council_deliberation` | Six `@mcp.tool(name=...)` overrides; call sites bind to these exact strings | Preserved verbatim. The typos `Genuis_level` and `adversarial_dual_council_diliberation` are **intentional and must not be corrected**. |
| `deep_reasoning_alias`, `genius_level_alias`, `adversarial_dual_council_deliberation_alias` | Pure aliases to the strings above | Preserved; alias targets keep the typo'd spellings. |
| `orochi_route` | Different description/semantics ("Orochi evolved artifacts"), not part of the byte-identical 9-template family | Not merged. |
| `spider_route_findings`, `skill_route`, `complexity_route` | Different signatures/semantics | Not merged. |

## Migration

```python
from tool_consolidation import resolve, mcp_route, mcp_session_list, mcp_session_replay

# Old (deprecated): audio_session_replay("abc")
resolve("audio_session_replay")("abc")          # warns, namespace frozen
mcp_session_replay("abc", namespace="audio")    # canonical

# Old (deprecated): rag_route("council", {"q": 1})
resolve("rag_route")("council", {"q": 1})       # warns, source frozen
mcp_route("council", {"q": 1}, source="rag")    # canonical
```

## Honest accounting

- Consolidatable names per audit #886: **~43** (17 + 17 + 9; the 3 aliases and 6 overrides are preserved, not merged).
- Shim-layer interface reduction: **43 → 3 = 40**.
- Live MCP tool count: **539 → 539**. Cutover to the canonical tools is a separate, opt-in config change for the next minor release. **No claim of 536 → 50 is made.**
