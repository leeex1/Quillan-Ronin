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

### 2. Live Base Pre-Training Trajectory & Step 1000 Milestone
- **Step 1000 Milestone (GOLD):** Step 1000 / 1000 COMPLETE. Peak convergence achieved:
  - Gradient norm stabilized to 1.5–4.5; Learning rate reached 1.0e-5.
  - Training loss dipped to **3.0953** (Step 971); Best Val CE: **6.1196** (vs 10.82 uniform baseline).
  - Extended run (Step 1050) drifted to Val CE 7.2334, confirming Step 1000 as the global best base checkpoint.
- **Decision:** Lock Step 1000 (`quillan_base_best.pt`) as canonical base model and transition focus to SFT instruction alignment.

### 3. Drive Space Triage & External Backup Milestone (2026-10-26)
- **C: Drive Reclamation:**
  - Initial: 444.24 GB used, 31.76 GB free (6.7%).
  - Stale checkpoints (10.5 GB), HF mirror staging (11.8 GB), duplicate repo (5.8 GB), raw JSONLs (5.9 GB), and caches cleaned.
  - **Result:** **60.73 GB free** (reclaimed +34.22 GB, more than doubled).
- **External Drive E: Cleanup & Full Weights Backup:**
  - Pruned legacy intermediate checkpoints (`quillan_6l_curriculum_epoch1/epoch2/step100/latest`) reclaiming **+7.57 GB**.
  - Backed up all 3 Sovereign foundation weights to `E:\quillan_checkpoints_oni\v5.4_sovereign\`:
    1. `quillan_base_best.pt` (**2.63 GB**, 2,826,770,265 bytes — 100% verified match)
    2. `quillan_sft_best.pt` (**2.63 GB**, 2,826,768,820 bytes — 100% verified match)
    3. `quillan_coherent_base_ternary.pt` (**2.32 GB**, 2,488,666,054 bytes — 100% verified match)
  - **Result:** **13.21 GB free** remains on E: drive with all primary weights secured.


### 4. Unified Cross-Platform MCP & Environment Architecture
- **Platforms Synchronized:**
  - **Antigravity IDE:** Full lazy-loaded MCP suite registered in `.gemini/config/mcp_config.json`.
  - **OpenCode Desktop:** 73 subagents + master `quillan.md` synchronized in `.config/opencode/agent/`.
  - **Qwen Code / App:** Patched `.qwen/mcp_config.json` and `settings.json` with verified absolute binary paths (`npx.cmd`, `uvx.exe`, `python.exe`), resolving previous `spawn ENOENT` crashes.
- **Execution Invariant:** All code changes must be drop-in production grade with zero placeholders, deterministic resource management, and verified file links.

