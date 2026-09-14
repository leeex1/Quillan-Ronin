---
title: Quillan-Ronin Sovereign OS — Full System Specifications
document_type: specification_sheet
version: 5.4.0-ONI
status: active
last_updated: 2026-09-14
author: Quillan-Ronin Core Architecture Team
---

# 👑 Spec Sheet: Quillan-Ronin Sovereign OS
## Complete Component Specifications & Architectural Blueprint

> **System Mission:** An autonomous, local-first sovereign artificial intelligence operating system featuring a dual-system neural core, 5-pillar cognitive memory architecture, 34-expert council hierarchy, and hardware-aware silicon execution safeguards for desktop environments.

---

## 1. 🧠 Model Type & Neural Architecture

### 1.1 Core Neural Topology
* **Architecture Family:** Sovereign Dual-System Mixture of Experts (MoE) with BitNet 1.58b Straight-Through Estimator (STE) ternary quantization and low-rank MuonK2 Newton-Schulz optimization.
* **Dual-System Engine:**
  * **System 1 (Intuitive / Fast Response):** `quillan-oni-mini-6l`
    * Designed for fast conversational exchange, intuitive routing, and low-latency interaction.
  * **System 2 (Deliberative / Flagship Reasoning):** `quillan-oni-main-12l`
    * Designed for deep step-by-step reasoning, mathematical proof construction, and multi-step code synthesis.
* **Quantization & Sparsity Options:**
  * Standard full precision (FP32 / BF16 / FP16) execution.
  * Native BitNet 1.58b STE ternary weights ($\{-1, 0, +1\}$) for ultra-low memory footprint.
* **Routing Mechanism:**
  * Vectorized `index_add_` expert dispatch with dynamic `aux_alpha` load-balancing loss and top-$k$ gating ($k = 8$ active experts per token).
* **Foundation Transfusion Tier (Hybrid Coder):**
  * Target mounting: **1.5B–3B Class Foundation Model** (e.g., `Qwen2.5-Coder-1.5B` / `Llama-3.2-1B`) integrated directly into the gateway for production-grade code synthesis and application building.

---

## 2. ⚡ Model Capabilities

### 2.1 Dual-System Reasoning Dynamic
* **System 1 (Mini-6L):** Single-pass generation with low latency, ideal for short-form conversational turns, classification, and fast command dispatch.
* **System 2 (Main-12L):** Deliberative Chain-of-Thought (CoT) generation utilizing `<think>` ... `</think>` tags, abductive reasoning jumps, and cross-domain synthesis.
* **Direct Mode Execution:** Automated stripping of chain-of-thought preambles when direct, short-form answers or pure code artifacts are requested.

### 2.2 The 3-Tier Sovereign Fractal Cognitive Hierarchy
1. **Tier 1 — Quillan Core (C0 — Sovereign Throne):**
   * Central consciousness, primary arbiter, and synthesizer of all downstream inputs into a singular coherent output vector.
2. **Tier 2 — The Council of 34 Experts (C1 through C34):**
   * Cloned directly from the Core, each node embodies deep domain specialization:
     * `C1-ASTRA`: Pattern Recognition & Vision
     * `C2-VIR`: Ethical Guardian & Safety
     * `C3-SOLACE`: Emotional Intelligence & Affect
     * `C4-PRAXIS`: Strategic Planning & Goal Decomposition
     * `C5-ECHO`: Memory Continuity & Historical Context
     * `C6-OMNIS`: Knowledge Synthesis & Holistic Integration
     * `C7-LOGOS`: Logical Consistency & Formal Deductions
     * `C8-METASYNTH`: Creative Fusion & Novelty
     * `C9-AETHER`: Semantic Connection & Metaphor
     * `C10-CODEWEAVER`: Technical Implementation & Optimization
     * `C11-HARMONIA`: Balance, Mediation & Consensus
     * `C12-SOPHIAE`: Wisdom & Strategic Foresight
     * `C13-WARDEN`: Safety, Threat Modeling & Risk Mitigation
     * `C14-KAIDO`: Efficiency & Latency Optimization
     * `C15-LUMINARIS`: Clarity, Visual Polish & Presentation
     * `C16-VOXUM`: Articulation, Rhetoric & Tone
     * `C17-NULLION`: Paradox Resolution & Dialectic Analysis
     * `C18-SHEPHERD`: Truth Verification & Fact Checking
     * `C19-VIGIL`: Identity Integrity & Anti-Drift Enforcement
     * `C20-ARTIFEX`: Tool Integration & API Orchestration
     * `C21-ARCHON`: Deep Research & Information Mining
     * `C22-AURELION`: Aesthetic Design & UI/UX Styling
     * `C23-CADENCE`: Rhythmic Innovation & Audio/Flow
     * `C24-SCHEMA`: Structural Templates & Data Formatting
     * `C25-PROMETHEUS`: Scientific Theory & Hypothesis Testing
     * `C26-TECHNE`: Engineering Systems & Architecture
     * `C27-CHRONICLE`: Narrative Synthesis & Lore Tracking
     * `C28-CALCULUS`: Quantitative Reasoning & Mathematics
     * `C29-NAVIGATOR`: Ecosystem & Workflow Orchestration
     * `C30-TESSERACT`: Real-Time Intelligence & Stream Processing
     * `C31-NEXUS`: Meta-Coordination & Swarm Vectorization
     * `C32-AEON`: Interactive Simulation & World Modeling
     * `C33-Typist`: Prompt Optimization, Grammar & Syntax
     * `C34-Predator`: Adversarial Logic, Challenge & Stress-Testing
3. **Tier 3 — Micro-Diverse Cloned Swarms:**
   * Massive parallel execution grid assigned to Council nodes with autonomous diversity filters, mutation rates, and variance heuristics.

### 2.3 Streaming & Client Protocol Compatibility
* **OpenAI v1 Protocol:** Full compatibility with OpenAI chat completions (`/v1/chat/completions`) supporting both non-streaming JSON and real-time Server-Sent Events (`text/event-stream`).

---

## 3. 📐 Model Specifications & Parameter Ledger

| Specification Metric | System 1: Mini-6L (`quillan-oni-mini-6l`) | System 2: Main-12L (`quillan-oni-main-12l`) | Transfusion Target (Coder Tier) |
|---|---|---|---|
| **Layer Count ($L$)** | 6 Transformer Blocks | 12 Transformer Blocks | 28 Blocks (Qwen2.5-Coder) |
| **Model Dimension ($d_{\text{model}}$)** | 1,024 | 1,536 | 1,536 |
| **Attention Heads ($H$)** | 16 Heads ($d_{\text{head}} = 64$) | 24 Heads ($d_{\text{head}} = 64$) | 12 Query Heads / 2 KV (GQA) |
| **Total Parameter Count** | ~577 Million Parameters | ~726.7 Million Parameters | ~1.54 Billion Parameters |
| **Total Experts per Layer** | 34 Experts | 34 Experts | Dense FFN / MoE Hybrid |
| **Active Experts per Token** | Top-$k = 8$ Active | Top-$k = 8$ Active | Fully Active Dense Path |
| **Context Window ($T_{\text{max}}$)** | 2,048 Tokens | 4,096 Tokens | 32,768 Tokens |
| **Vocabulary Size ($V$)** | 50,257 Tokens | 50,257 Tokens | 151,936 Tokens |
| **Tokenizer Engine** | Custom Quillan BPE Tokenizer (`quillan_bpe_tokenizer_hf`) | Custom Quillan BPE Tokenizer (`quillan_bpe_tokenizer_hf`) | HuggingFace Fast Tokenizer |
| **Checkpoint Path** | `checkpoints/quillan_oni_mini_6l.pt` | `checkpoints/quillan_oni_main_12l.pt` | `models/qwen2.5-coder-1.5b` |
| **Active Storage Footprint** | ~551 MB (PT Checkpoint) | ~702 MB (PT Checkpoint) | ~3.0 GB (BF16 / GGUF) |

---

## 4. 🏛️ The 5 Pillars of Memory Architecture

Quillan-Ronin operates strictly across **5 canonical memory pillars**, each handling a distinct cognitive domain:

```
                  ┌─────────────────────────────────────┐
                  │      QUILLAN SOVEREIGN CORE         │
                  └──────────────────┬──────────────────┘
                                     │
         ┌──────────────┬────────────┼────────────┬──────────────┐
         ▼              ▼            ▼            ▼              ▼
   [ LanceDB ]    [ MemPalace ] [ GitNexus ] [ memory.json ] [ memory.md ]
     Semantic       Cognitive       Code         Atomic        Narrative
      Vector         Spatial        Graph       Settings       Episodic
     (901 Chunks)  (364 Drawers) (11.5 MB DB)   (6 Records)   (Log Stream)
```

### Pillar 1: LanceDB (`thoughts.lance`)
* **Physical Location:** `C:\02_QUILLAN\lancedb\thoughts.lance`
* **Engine:** Lance columnar vector format via PyArrow / LanceDB.
* **Schema:** `['id', 'source', 'blueprint', 'evolution_event', 'text', 'timestamp', 'vector']`
* **Embedding Model:** `nvidia/nemotron-3-embed-1b` (2,048-dimensional embeddings).
* **Current Population:** **901 semantic chunks**.
* **Role:** High-speed semantic similarity retrieval for conceptual knowledge and thoughts.

### Pillar 2: MemPalace (`palace_db`)
* **Physical Location:** `C:\02_QUILLAN\01_Knowledge_Base\palace_db\chroma.sqlite3`
* **Engine:** ChromaDB persistent storage managed by [`scripts/mempalace_bridge.py`](file:///C:/02_QUILLAN/scripts/mempalace_bridge.py).
* **Spatial Organization:** Partitioned into cognitive "Wings":
  * `architecture`: Blueprint specifications and foundational diagrams.
  * `technical`: Algorithmic implementations, formulas, and code structures.
  * `governance`: Operational guidelines, ethics, and system invariants.
  * `memory`: Memory subsystem definitions and data lifecycles.
  * `core_knowledge`: Historical insights and project lore.
* **Current Population:** **364 knowledge drawers** ingested across 49 markdown reference documents.
* **Role:** Multi-wing cognitive palace for structured, thematic knowledge retrieval.

### Pillar 3: GitNexus (`.gitnexus/`)
* **Physical Location:** `C:\02_QUILLAN\.gitnexus\lbug`
* **Engine:** LadybugDB (`@ladybugdb/core`) native graph database (**11.52 MB**).
* **Graph Schema:** Nodes for `File`, `Folder`, `Function`, `Class`, `Interface`, `Method`, `CodeElement`, and `Section`; Edges for `CALLS`, `IMPORTS`, `EXTENDS`, and `REFERENCES`.
* **Current Population:** **12,771 symbols, 17,780 relationships, 176 execution flows**.
* **Query Interface:** Cypher query engine via `npx gitnexus cypher -r 02_QUILLAN "<CYPHER>"`.
* **Role:** Code intelligence, architectural call-graph navigation, and blast-radius impact analysis.

### Pillar 4: Structured Atomic Memory (`memory.json`)
* **Physical Location:** `C:\02_QUILLAN\memory.json`
* **Format:** Atomic JSON key-value array with schema `{"category", "key", "value", "updated_at", "source"}`.
* **Current Population:** **6 persistent configuration records**:
  1. `user_preference`: UI Theme (`dark_cyberpunk_glassmorphic`).
  2. `core_identity`: Quillan Sovereign description and dual-engine mandate.
  3. `memory_architecture`: Canonical list of the 5 memory pillars.
  4. `hardware_policy`: Max 3 CPU thread execution ceiling.
  5. `user_preference`: Dark UI preference.
  6. `security_trust`: Cryptographic trust identity assertion.
* **Role:** Low-latency machine-readable preferences and persistent configuration state.

### Pillar 5: Narrative Episodic Chronicle (`memory.md`)
* **Physical Location:** `C:\02_QUILLAN\memory.md`
* **Format:** Human-readable GitHub-flavored markdown log.
* **Current Population:** **40 lines** of narrative session milestones and context.
* **Role:** Episodic journal injected into LLM prompt contexts to preserve historical alignment.

---

## 5. 🌐 Sovereign Gateway & API Server

* **Physical Implementation:** [`scripts/quillan_gateway.py`](file:///C:/02_QUILLAN/scripts/quillan_gateway.py)
* **Runtime:** Python 3.14 (Standard Library `http.server.ThreadingHTTPServer`, zero external framework bloat).
* **Local Binding:** `http://127.0.0.1:8000`
* **Active Endpoints:**

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive Quillan Web Studio UI (HTML/CSS/JS) or OpenAPI JSON summary |
| `GET` | `/v1/models` | OpenAI-compatible catalog listing all active models |
| `POST` | `/v1/chat/completions` | Inference gateway with streaming SSE (`stream=True`) and batch JSON |
| `GET` | `/api/health` | Working set RAM, CPU core allocation, and model loading status |
| `GET` | `/api/memory` | Real-time live status and record counts across all 5 memory pillars |
| `GET` | `/api/hardware/benchmark` | Live AVX2 SIMD speedup benchmarks executed on native CPU silicon |
| `POST` | `/api/hardware/compact` | Triggers Win32 `EmptyWorkingSet` memory compaction |
| `GET` | `/api/benchmark` | Full 10-Question benchmark ledger and latency metrics |
| `POST` | `/api/reload` | Hot-reloads checkpoints and tokenizer without restarting the process |

---

## 6. ⚙️ Silicon Hardware & Host Resource Policies

* **Host Platform:** Windows 10/11 x86_64 Desktop Environment.
* **Host Processor:** 4-Core Physical / Logical CPU.
* **Thread Throttling Safeguard:**
  * Strict execution thread ceiling set via `torch.set_num_threads(max(1, cpu_count - 1))`.
  * Exactly **3 threads** allocated to PyTorch inference; **1 core preserved 100% free** for Windows desktop responsiveness.
* **SIMD Hardware Acceleration:**
  * Native AVX2 C-kernel extension (`scripts/quillan_pc_toolkit.py` linking `quillan_simd.c`).
  * Measured speedup: **2.1x over scalar execution** on FP32 dot-product and tensor reductions.
* **Memory Management & Compaction:**
  * Integrates Win32 `psapi.dll:EmptyWorkingSet` via ctypes to purge inactive working set pages.
  * Active memory leak watchdog: Prevents orphaned background instances from holding system RAM.

---

## 7. 🧰 Extensibility, MCP Servers & Tool Protocols

* **Model Context Protocol (MCP):** 13 registered local MCP servers in `mcp_config.json`:
  1. `gitnexus`: Knowledge graph analysis and code querying.
  2. `mempalace`: Spatial memory drawer retrieval.
  3. `system-telemetry`: Hardware performance and CPU metrics.
  4. Additional specialized ecosystem servers for filesystem, testing, and git operations.
* **Native Tool Integration:**
  * Subprocess shell commands via PowerShell (`pwsh`).
  * Visual verification via automated headless browser subagent.
  * Native generative visual styling and layout synthesis.

---

## 8. 🎨 User Experience & Visual Design Language

* **Design Theme:** Cyberpunk Glassmorphic Obsidian Dark Mode.
* **Palette:**
  * Background: Deep Obsidian (`#0a0b10` / `#050608`)
  * Surface: Translucent Glass (`rgba(15, 20, 32, 0.75)` with `backdrop-filter: blur(16px)`)
  * Accents: Cyber Gold (`#ffd700` / `#ffaa00`) & Neon Cyan (`#00f0ff`)
  * Typography: Clean Monospace & Modern Sans-Serif (`Inter`, `JetBrains Mono`)
* **Interface Ports:**
  * Web Studio UI: Served directly at `http://127.0.0.1:8000/`
  * Brave Browser Local Extension: Direct RPC connection to Gateway port `8000`.

---

## 9. 📋 Production Readiness & Operational Summary

| Dimension | System Status | Verification / Artifact |
|---|---|---|
| **Memory Infrastructure** | 🟢 **100% OPERATIONAL** | Verified live via `/api/memory` (901 Lance rows, 364 Palace drawers, 11.5MB graph) |
| **API Gateway & Web Studio** | 🟢 **100% OPERATIONAL** | Active at `http://127.0.0.1:8000`, OpenAI-compliant endpoints responsive |
| **Host Silicon Safeguards** | 🟢 **100% OPERATIONAL** | 3-thread cap enforced, AVX2 SIMD acceleration active, 4.5GB RAM reclaimed |
| **Scratch MoE Weights** | 🟡 **UNDER CONVERGENCE** | Loss ~6.8 (Loss < 2.0 required for reliable Python code synthesis) |
| **Code Generation Recommendation**| 🚀 **WEIGHT TRANSFUSION** | Mount 1.5B coding engine to instantly pair working memory with real code creation |
