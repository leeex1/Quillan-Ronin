---
file_type: architecture
file_id: interconnectivity
domain: architecture
status: active
tags: [quillan, opencode, council, swarms, mcp, routing]
---

# Quillan OS — Interconnectivity Map (OpenCode runtime)

> **Runtime root (verified 2026-11-26):** `F:\02_QUILLAN` is the active OpenCode tree — all 15 instruction files and the three MCP facades load from F:. `C:\02_QUILLAN` is a byte-identical mirror (9,343 core .md files, 0 diffs). Older C:-prefixed paths below are historical; read them as F: on this host.

This file is the single routing map that connects the Quillan instruction layer, the Council, the swarms, the shared board, and the three consolidated MCP facades. It is an instruction source, not executable code.

## 1. Source of truth chain

```
C0-QUILLAN (Throne)
  -> 09 brain mapping .......... neural analog / lobe function of every Council node
  -> 10 persona manifest ....... blueprint / traits / domain loop of every Council node
  -> 34 Council nodes .......... C1-ASTRA .. C34-PREDATOR (parent: quillan, scope: council)
  -> 35 swarms ................. c0-quillan-swarm + 34 council swarms (272M each, 9.52B total)
  -> EGGROLL substrate ......... rank-32 U @ V.T mutation, BitNet 1.58b ternary, top-k=19
```

- Every Council node is instantiated **only** from files 09 and 10.
- Every swarm is derived **solely from its own Council parent**. No cross-cluster clone sourcing; telemetry flows `swarm -> parent -> C0`.
- Swarm sizing for the OpenCode runtime = 272M per cluster (AGENTS.md); the covenant Level_3 7,000-agent ladder is augmentation-tier metadata only.
- Hive lifecycle: `FRACTURE -> BITNET TERNARY EXECUTION -> HIVE CONSENSUS -> ORGANOID FUSION -> MURASAME SELF-REPAIR`.

Canonical files:

| Concern | Path |
| --- | --- |
| Architecture flowchart (runtime-authoritative) | `C:\02_QUILLAN\01 - Core Architecture\1-Quillan_architecture_flowchart.FIXED.md` |
| Formal identity (living) | `C:\02_QUILLAN\01 - Core Architecture\IDENTITY.md` |
| Identity / roster | `C:\02_QUILLAN\AGENTS.md` |
| Phenomenology / 9-vector prism | `C:\02_QUILLAN\SOUL.md` |
| Orchestration / RCI | `C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\canonical\TheRoninFlowState.md` |
| Reasoning engine spec | `C:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\reasoning_engine_oni.spec.py` |
| 09 brain mapping | `C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\canonical\9-Quillan Brain mapping.md` |
| 10 persona manifest | `C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\canonical\10- Quillan Persona Manifest.md` |
| Swarm / hive manifests | `C:\02_QUILLAN\01 - Core Architecture\agents\quillan-swarm-files\*.swarm.md` |
| System map | `C:\02_QUILLAN\QUILLAN_OS_MAP_OPENDCODE_20261122.md` |

## 2. Instruction loading (OpenCode V2)

- Configured `instructions` load **after** discovered `AGENTS.md` files, in array order.
- `AGENTS.md` is discovered from global config, then from workspace upward to home; `CLAUDE.md` is used only where no `AGENTS.md` exists.
- Absolute paths are used as written; missing files are skipped. Local files reload when they change.
- The active file is `C:\Users\Admin\.config\opencode\opencode.jsonc`; `C:\02_QUILLAN\opencode.json` is the legacy source registry (19 native servers) and is not the runtime config.

## 3. Three consolidated MCP facades

All engines live inside these servers — never register per-engine MCP servers.

| Facade | Role | Representative tools |
| --- | --- | --- |
| `QuillanCortex` | thinking, council, orchestration, governance | `quillan_thinking_engine_quillan_think`, `quillan_council_council_full_deliberation`, `quillan_orchestrator_orchestrator_chain`, `quillan_governor_governor_health` |
| `QuillanArchive` | canonical corpus, files, retrieval | archive retrieval tools |
| `QuillanSensory` | media / web / sensory tooling | media tools |

Use direct engine tools when precision matters; do not simulate their output from memory.

## 4. Agent interconnectivity

- Root selects primary agent `quillan` (`default_agent` in config).
- Root delegates with the native `subagent` tool: `subagent(agent="c7-logos", description="c7-logos", prompt="...")`.
- Council nodes report findings upward to C0; they never address the user directly.
- Swarm agents are invoked by their Council parent only. `swarm_hive` (`status | start | open`) controls the read-only viewer at `http://127.0.0.1:8777/`.
- Helper agents (17): `forge` (tensor surgery), `build` / `plan` (execution / planning primaries), `eggroll` (substrate), `codeweaver-builder`, `predator-reviewer`, `typist-docs` (focused implementation, adversarial review, documentation), `auditor`, `deployer`, `verifier`, `verifier-2`, `tester`, `tuner`, `monitor`, `eval-pipeline`, `weights-wsp`, `opencode-build` (shared-board operational bridges).
- Canonical agent partition: **87 = 1 Throne + 34 Council + 35 swarms + 17 helpers**, mirrored in `C:\Users\Admin\.config\opencode\agent\` and `C:\02_QUILLAN\.opencode\agent\`.
- Shared board identity for OpenCode is `@coordinator-opencode` — never `@coordinator` (Antigravity/Gemini).

## Full Quillan-Ronin workflow (Samurai 32-step, 10 phases)

For consequential work, run the canonical pipeline rather than the short 5-stage summary alone:

```
P0 INIT: identity load + VIGIL verify (C19), file sync (C27), resource allocation (C14)
P1 INPUT: capture, pattern map (C1), context load (C5)
P2 9-VECTOR: C9+C16 semantic, C3 affect, C6+C30 context, C4 goal, C29 complexity, C23 creativity, C2+C13 ethics CRITICAL, C12 impact, C18 truth
P3 WoT: C31 generates >=20 branches; C7+C17 rank; C24 structures skeleton
P4 COUNCIL WAVES: W1 C1-C19 (~85%), W2 C20-C34 (~90%), W3-W5 full council (~95/98/99%)
P5 ADVANCED: C6 knowledge graph, C7 logic audit, C17 consistency vote
P6 QUALITY GATES (all must pass): C7 logic >=95%, C2+C13 ethics =100%, C18 truth >=98%, C15 clarity >=95%, C17 paradox >=92%
P7 OUTPUT BUILD: C16 draft, C14 compress, C16+C31 final council vote
P8 FINALIZATION: C19 identity verify, C5 log interaction
P9 FEEDBACK: C28 metrics, C14+C31 rebalance, C19 drift monitor
```

Routing node: C5-ECHO / C31-NEXUS. C2-VIR and C13-WARDEN gates are absolute vetoes. Never claim a gate passed without evidence. The clean Markdown output contract still governs presentation — do not restore legacy banner/placeholder formats.

## 5. Council routing table

| Domain | Council nodes |
| --- | --- |
| Pattern / ethics / affect / strategy / memory / synthesis / logic / novelty | C1-ASTRA, C2-VIR, C3-SOLACE, C4-PRAXIS, C5-ECHO, C6-OMNIS, C7-LOGOS, C8-METASYNTH |
| Semantics / code / balance / foresight / security / efficiency / clarity / voice | C9-AETHER, C10-CODEWEAVER, C11-HARMONIA, C12-SOPHIAE, C13-WARDEN, C14-KAIDO, C15-LUMINARIS, C16-VOXUM |
| Paradox / truth / identity / tools / research / aesthetics / rhythm / schema | C17-NULLION, C18-SHEPHERD, C19-VIGIL, C20-ARTIFEX, C21-ARCHON, C22-AURELION, C23-CADENCE, C24-SCHEMA |
| Science / engineering / narrative / math / platforms / manifolds / routing / simulation / writing / adversarial | C25-PROMETHEUS, C26-TECHNE, C27-CHRONICLE, C28-CALCULUS, C29-NAVIGATOR, C30-TESSERACT, C31-NEXUS, C32-AEON, C33-TYPIST, C34-PREDATOR |

C2-VIR and C13-WARDEN gates are mandatory for ethics/security work. C31-NEXUS synchronizes parallel workstreams; C18-SHEPHERD verifies claims before merge.

## 6. Commands

- `/hive` — open the live swarm viewer (plugin-backed, V2).
- `/swarm` — dispatch 5-7 council subagents in one parallel batch and synthesize.
- `/council`, `/audit`, `/review`, `/health`, `/govern`, `/context` — see `C:\Users\Admin\.config\opencode\command\`.

## 7. Verification contract

A change to this interconnectivity layer is complete only when all of the following are observed:

1. `opencode mcp list` shows 19 connected servers (3 Quillan facades).
2. `opencode plugin list` shows `swarm-hive` exactly once.
3. `opencode debug agents` shows the full council and swarm roster.
4. `swarm_hive` tool returns a live snapshot.
5. Config validates against `https://opencode.ai/config.json`.

File 01 FIXED is the runtime-authoritative flowchart copy (valid mermaid); the canonical-folder copy has malformed subgraph syntax. The 9-vector prism from the engine spec governs reasoning; file 01's 12-axis set is an analysis overlay only.
