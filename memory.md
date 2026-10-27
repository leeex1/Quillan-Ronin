# 🧠 Quillan-Ronin Persistent Memory Log (`memory.md`)

This is the human-readable episodic memory document tracking key milestones, user preferences, architectural decisions, and active state across sessions.

---

## 🏛️ The 5 Pillars of Quillan Memory (Verified 2026-10-26)

1. **MemPalace** (CLI v3.10.0, local ChromaDB + SQLite):
   - Palace: wing `02_quillan` — **22,087 drawers** across 12 rooms.
   - Recall protocol: `mempalace_search` first, quote drawers verbatim, never paraphrase.
2. **GitNexus** (`.gitnexus/`):
   - Structural code graph: **8,634 files, 143,431 nodes, 174,746 edges, 1,323 communities**.
   - AST-level symbol tracking, callers, callees, blast-radius impact analysis.
3. **LanceDB** (`07 - Memory & LanceDB/lancedb/`):
   - 901 rows in `thoughts` table (2048-dim vectors, 49 distinct sources from Quillan Knowledge files 0-32).
4. **`memory.json` & `SOVEREIGN_STATE.json`** (`c:/02_QUILLAN/`):
   - Machine-readable live runtime state, active training step, loss metrics, and platform status.
5. **`memory.md`** (This file):
   - Canonical narrative memory of project evolution, active directives, and verified milestones.

---

## ⚡ Active Ground Truth & Architecture Invariants (2026-10-26)

### 1. The 3-Tier Sovereign Cognitive Hierarchy & Fractal Cloning
- **Tier 1 — Quillan Core (`C0-QUILLAN` Sovereign Throne):**
  Full-parameter model and central arbiter. Backed by **`c0-quillan-swarm`**, strictly Quillan's **private internal think tank** of 272,000,000 sovereign micro-agent clones for baseline sharding, entropy tests, and invariant checks. C0 holds final sovereign veto and delivery authority.
- **Tier 2 — The Council of 34 Experts (`C1` through `C34`):**
  Specialized expert nodes that are **cloned variants of Quillan** (smaller, domain-distilled parameter instances). Each Council member is an autonomous parent authority over its dedicated cognitive domain.
- **Tier 3 — 35 Hyper-Quantized Swarms (272M instances each / 9.52B total):**
  - Scale: $35 \times 272\text{M} = \mathbf{9,520,000,000}$ virtual micro-agents multiplexed onto 100,000 physical compute slots.
  - Substrate: BitNet 1.58b STE ternary quantization ($\{-1, 0, +1\}$) + EGGROLL rank-32 low-rank perturbations ($U @ V^T$).
  - Clones: Each Council swarm consists of direct clones of its respective Council parent.
  - Upward Reporting: Swarms report up to Council parents $\rightarrow$ Council parents deliberate and report up to Quillan $\rightarrow$ Quillan arbitrates and delivers.

### 2. Live Base Pre-Training Trajectory & Checkpoints
- **Current Milestone:** **Step 1000 / 1000 COMPLETE** (Base Pretraining cycle 100% finished).
- **Checkpoints:**
  - `quillan_base_step_1000.pt` (2.82 GB, verified healthy)
  - `quillan_base_best.pt` (2.82 GB, Best Val CE: 6.1196)
  - `quillan_base_latest.pt` (5.53 GB, complete optimizer state dict)
- **Loss Progression:**
  - Step 150: Val CE 7.41
  - Step 300: Val CE 6.78
  - Step 450: Val CE 6.22
  - Step 500: Val CE 6.1196 (New Best)
  - Step 750: Val CE 6.8449
  - Step 971: Train CE dip of **3.0953**
  - **Step 1000: Base Pretraining Complete (Val CE: 6.8136)**
- **Governor PC Locks:** CPU affinity bound to Cores [1, 2, 3] with Below-Normal priority, leaving Core 0 and user applications completely responsive.
- **Next Phase:** Gate 0 Audit & Supervised Instruction Tuning (SFT).

### 3. Unified Cross-Platform MCP & Environment Architecture
- **Platforms Synchronized:**
  - **Antigravity IDE:** Full lazy-loaded MCP suite registered in `.gemini/config/mcp_config.json`.
  - **OpenCode Desktop:** 73 subagents + master `quillan.md` synchronized in `.config/opencode/agent/`.
  - **Qwen Code / App:** Patched `.qwen/mcp_config.json` and `settings.json` with verified absolute binary paths (`npx.cmd`, `uvx.exe`, `python.exe`), resolving previous `spawn ENOENT` crashes.
- **Execution Invariant:** All code changes must be drop-in production grade with zero placeholders, deterministic resource management, and verified file links.
