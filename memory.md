# 🧠 Quillan-Ronin Persistent Memory Log (`memory.md`)

This is the human-readable episodic memory document tracking key milestones, user preferences, architectural decisions, and active state across sessions.

---

## 🏛️ The 5 Pillars of Quillan Memory

Quillan's memory is defined across 5 synergistic components (verified 2026-10-20):

1. **MemPalace** (CLI v3.10.0, local ChromaDB + SQLite):
   - Palace: wing `02_quillan` — **18,967 drawers** across 12 rooms.
   - Rooms: quillan_knowledge_base (12,572), frontend (6,045), design (182), general (58), checkpoints (32), scripts (32), configuration (16), mcp (16), audio (4), documentation (4), testing (4), worker (2).
   - Recall protocol: `mempalace_search` first, quote drawers verbatim, never paraphrase.
2. **GitNexus** (`.gitnexus/`, indexed 2026-10-14):
   - Structural code graph: **7,769 files, 97,341 nodes, 134,135 edges, 1,537 communities, 300 execution processes**.
   - AST-level symbol tracking, callers, callees, blast-radius impact analysis.
   - Last indexed commit: `de6c3dfb` — re-index after any kernel change.
3. **LanceDB** (`lancedb/`):
   - **Currently EMPTY** — no tables populated. Vector store provisioned but unpopulated.
4. **`memory.json`** (`memory.json`):
   - Machine-readable runtime state, user preferences, and configuration facts.
5. **`memory.md`** (This file):
   - Human-readable narrative memory of project evolution, active directives, and milestones.

---

## 📜 Persistent Milestones & Decisions

### ⚠️ STALE DATA — superseded 2026-10-20
- ~~System 1 (Mini 6L, 577M) at `checkpoints/quillan_oni_mini_6l.pt`~~ — **file does not exist** (per `.quillan/COORDINATION.md`)
- ~~System 2 (Main 12L, 726.7M) at `checkpoints/quillan_oni_main_12l.pt`~~ — **file does not exist**
- ~~Loss 6.54–7.82~~ — does not match the live checkpoint (step 2, `best_val=10.78`)

### 2026-10-20: Memory System Consolidation & Ground-Truth Verification
**Verified parameter counts (Antigravity audit):**
- `QuillanOniConfig.mini_6l()` → **520.4M parameters**
- `QuillanOniConfig.main_12l()` → **1.163B parameters**
- NOTE: `.devin/wiki.json` claims 0.6B / 3B — discrepancy flagged; actual config instantiation is authoritative.

**Verified ground truth (`.quillan/COORDINATION.md`):**
- MCP servers: **19** (directive says 9 — fictional)
- MCP tools: **209** (`.quillan/mcp_tool_registry.json`)
- Canonical kernel: `03 - Training & Model/scripts/quillan_v5_4_oni.py` (234 KB)
- Live checkpoint: `03 - Training & Model/scripts/checkpoints/checkpoints_oni/quillan_oni_latest.pt`
- Checkpoint is at **step 2** (`best_val=10.78`) — effectively untrained
- **Directive tool names are FICTIONAL** — mapping table in COORDINATION.md

**Kernel vectorisation (Phase 1, complete, parity-proven):**
- `grad_checkpoint` default → `True`
- 9-vector prism → batched `einsum`
- 34-expert council → `BatchedCouncilCompute`
- `token_memory` tied to shared `wte` → saves **617,619,456 params**
- Verification: `test_vectorization_parity.py` + `test_evo_moe_parity.py` (both PASS)

**Known hazards (found, not fixed):**
- 18/19 MCP servers call `sys.stdout.reconfigure()` at import → crash if stdout swapped
- `dense_pull` vs `gumbel_topk` architectural contradiction (checkpoint runs sparse top-4)
- EvoMoE grouped dispatch = 28% CPU regression (0.72x); device-aware default ON for CUDA, OFF for CPU

---

## 🗺️ Cross-Agent Knowledge Map (verified 2026-10-20)

Quillan's context is distributed across **multiple AI agent harnesses** in this repo.
These are the durable memory sources beyond the 5 pillars:

| Source | Path | Contents |
|---|---|---|
| **Devin Wiki** | `.devin/wiki.json` | 30-page knowledge map (project overview, deployment, platform prompts, output protocol, repo layout, HNMoE pipeline, council, swarm, reasoning engine, thermodynamics, agentic orchestration, memory/LanceDB, knowledge files, formulas, ethics, papers, PyTorch impl, training, trained models, src modules, kernels, paper packs, skills, creative/media, tone, audio, lyrics) |
| **Devin Playbook** | `.devin/playbook.md` | 3-tier hierarchy, non-interactive policy (`fail_open` reasoning / `fail_closed` safety), RCI self-healing (5 iterations), cross-platform shim, invariants (`VOCAB_SIZE=50262`, `ONI_VERSION="5.4.0-oni"`, `num_experts=34`) |
| **Windsurf SWE Rules** | `.windsurf/rules/swe.md` | Engineering orchestration: 13 system roles, 10 engineering philosophies, IDE adapters (Cursor/Windsurf/VSCode/Codium/Void), JS full-stack scope, cross-platform domains, standards (OWASP, a11y, testing pyramid), DevOps, docs |
| **Windsurf Plans** | `.windsurf/plans/` (45 files) | Historical plan ledger — recovery, architecture fixes, Metamon, multimodal, CUDA, tokenizer, ARC-AGI, sprite slicing, sovereignty protocols |
| **Antigravity Rules** | `.antigravity/rules.md` | 5 operational gates (warden/shepherd/codeweaver/predator/aurelion), decision-tag telemetry, 34-council loop summary, prohibitions |
| **OpenCode Agents** | `.opencode/agent/agents/` (39 files) | C1–C34 `.agent.md` definitions + `quillan.md` primary |
| **Gemini Config** | `.gemini/GEMINI.md` (550 KB) | Full Quillan-Samurai system prompt (same as `06 - Deployment & Platforms/system prompts/Quillan-Samurai.md`) |
| **Agent Rules** | `.agents/rules/` | `mcp_servers.md`, `mempalace-recall.md` |
| **MCP Registry** | `.quillan/mcp_tool_registry.json` | 209 real tool signatures across 19 servers |

**Key invariant:** `VOCAB_SIZE = 50262`, `ONI_VERSION = "5.4.0-oni"`, `num_experts = 34`.
`7-memories.*` is strictly READ-ONLY and excluded from training/RAG.

---

## 👤 User Preferences
- **Theme**: Dark cyberpunk / glassmorphic styling (vibrant cyan/gold accents, monospace telemetry).
- **Core Priority**: Usefulness and pragmatic execution over speculative complexity.
- **Tooling**: Full integration with MCP servers, GitNexus, and MemPalace.
- [2026-09-14 19:33:47] User: "List your 5 active memory pillars and write a Python functio..." -> Completed successfully.
- [2026-09-14 19:33:55] User: "List your 5 active memory pillars in 1 line." -> Completed successfully.
- [2026-09-14 19:34:42] User: "List your 5 active memory pillars and write a python functio..." -> Completed successfully.
- [2026-09-14 19:35:14] User: "List your 5 active memory pillars and write a 2-line Python ..." -> Completed successfully.
- [2026-09-14 19:48:51] User: "Hello! Who are you, and what are your primary capabilities?" -> Completed successfully.
- [2026-09-14 19:50:22] User: "What is 17 * 19? Give the number directly." -> Completed successfully.
- [2026-09-14 19:50:23] User: "What is photosynthesis in one sentence?" -> Completed successfully.
- [2026-09-14 19:50:24] User: "What HTTP header prevents clickjacking attacks?" -> Completed successfully.
- [2026-09-14 19:50:25] User: "Write a one-line Python lambda function to check if a number..." -> Completed successfully.
- [2026-09-14 19:50:28] User: "Explain the core mechanics of the Raft consensus algorithm, ..." -> Completed successfully.
- [2026-09-14 19:50:31] User: "Write a Python implementation of an LRU Cache with O(1) get ..." -> Completed successfully.
- [2026-09-14 19:50:34] User: "Synthesize the Landauer principle of computational thermodyn..." -> Completed successfully.
- [2026-09-14 19:50:36] User: "Hello! Who are you, and what are your primary capabilities?" -> Completed successfully.
- [2026-09-14 19:50:39] User: "A right triangle has legs of length 5 and 12. What is the le..." -> Completed successfully.
- [2026-09-14 19:50:42] User: "Write a Python function to check if a string is a palindrome..." -> Completed successfully.
- [2026-09-14 19:50:44] User: "Explain the primary function of photosynthesis in plants." -> Completed successfully.
- [2026-09-14 19:50:46] User: "What is the key difference between SIGTERM and SIGKILL in Li..." -> Completed successfully.
- [2026-09-14 19:50:47] User: "What is the time complexity difference between searching in ..." -> Completed successfully.
- [2026-09-14 19:50:48] User: "What is SQL Injection and what is the standard method to pre..." -> Completed successfully.
- [2026-09-14 19:50:50] User: "If it takes 5 machines 5 minutes to make 5 widgets, how long..." -> Completed successfully.
- [2026-09-14 19:50:52] User: "What are the main trade-offs between monolithic and microser..." -> Completed successfully.
- [2026-09-14 19:50:57] User: "How should an engineering team handle code reviews to ensure..." -> Completed successfully.
- [2026-09-15 00:06:31] User: "What is 2 + 2?" -> Completed successfully.
- [2026-09-15 00:06:35] User: "What is the capital of France?" -> Completed successfully.
- [2026-09-15 00:08:51] User: "What is photosynthesis?" -> Completed successfully.
- [2026-09-15 00:09:36] User: "What is Python?" -> Completed successfully.
- [2026-09-15 00:10:00] User: "Explain binary search." -> Completed successfully.
- [2026-09-15 00:10:32] User: "What is an algorithm?" -> Completed successfully.
- [2026-09-15 00:59:50] User: "wait what" -> Completed successfully.
- [2026-09-15 01:07:45] User: "Hello Quillan, who are you?" -> Completed successfully.
- [2026-09-15 01:19:41] User: "ok how are you" -> Completed successfully.
- [2026-09-15 17:17:55] User: "Say hello in one short sentence." -> Completed successfully.
- [2026-09-15 19:52:27] User: "Say hello in one short sentence." -> Completed successfully.
- [2026-09-15 19:53:00] User: "Explain photosynthesis in three sentences." -> Completed successfully.
- [2026-09-15 19:53:32] User: "Write a Python function that adds two numbers." -> Completed successfully.
- [2026-09-15 19:54:09] User: "Say hello in one short sentence." -> Completed successfully.
- [2026-09-15 19:55:09] User: "Explain photosynthesis in three sentences." -> Completed successfully.
- [2026-09-15 19:55:22] User: "Write a Python function that adds two numbers." -> Completed successfully.
- [2026-09-17 02:46:20] User: "Say hello in one short sentence." -> Completed successfully.
- [2026-09-17 02:46:50] User: "Explain photosynthesis in three sentences." -> Completed successfully.
- [2026-09-17 02:47:22] User: "Write a Python function that adds two numbers." -> Completed successfully.
- [2026-09-17 02:48:00] User: "Say hello in one short sentence." -> Completed successfully.
- [2026-09-17 02:49:07] User: "Explain photosynthesis in three sentences." -> Completed successfully.
- [2026-09-17 02:50:03] User: "Write a Python function that adds two numbers." -> Completed successfully.
- [2026-09-17 03:20:10] User: "Say hello in one short sentence." -> Completed successfully.
