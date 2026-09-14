# 🧠 Quillan-Ronin Persistent Memory Log (`memory.md`)

This is the human-readable episodic memory document tracking key milestones, user preferences, architectural decisions, and active state across sessions.

---

## 🏛️ The 5 Pillars of Quillan Memory

Quillan's memory is defined across 5 synergistic components:

1. **MemPalace** (`mempalace_bridge.py`):
   - Multi-wing spatial cognitive memory (Wings: `technical`, `consciousness`, `memory`, etc.).
   - Granular room-based storage for structured knowledge concepts.
2. **GitNexus** (`.gitnexus/`):
   - Structural code graph memory (12,771 symbols, 17,780 relationships, 176 execution flows).
   - AST-level symbol tracking, callers, callees, and blast-radius impact analysis.
3. **LanceDB** (`lancedb/`):
   - Persistent vector memory (`thoughts.lance` with 901 chunks, 2048-dim vectors).
   - Semantic similarity search over canonical knowledge base.
4. **`memory.json`** (`memory.json`):
   - Machine-readable runtime state, user preferences, and configuration facts.
5. **`memory.md`** (This file):
   - Human-readable narrative memory of project evolution, active directives, and milestones.

---

## 📜 Persistent Milestones & Decisions

### 2026-09-14: Dual Neural MoE & Sovereign Gateway v5.4.0
- **Model Checkpoints**: System 1 (Mini 6L, 577M) at `checkpoints/quillan_oni_mini_6l.pt` and System 2 (Main 12L, 726.7M) at `checkpoints/quillan_oni_main_12l.pt`.
- **Silicon Protection**: PyTorch execution threads capped at 3 on 4-core CPU to prevent desktop UI freezing.
- **Sovereign Web Studio**: Live interactive studio running on `http://127.0.0.1:8000` with dual-model toggle, telemetries, and memory trim.
- **Audit Findings**: Models are currently under-converged (Loss 6.54–7.82); weight transfusion path identified for real coding capability.

---

## 👤 User Preferences
- **Theme**: Dark cyberpunk / glassmorphic styling (vibrant cyan/gold accents, monospace telemetry).
- **Core Priority**: Usefulness and pragmatic execution over speculative complexity.
- **Tooling**: Full integration with MCP servers, GitNexus, and MemPalace.
