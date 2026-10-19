---
language: en
license: mit
pipeline_tag: text-generation
tags:
- quillan
- ronin
- ai
- cognitive-architecture
- mixture-of-experts
- h-n-moe
- bitnet
- diffusion-reasoning
- multi-agent
---

# ⚡ QUILLAN-RONIN v5.4.0-ONI

> **The standalone model is live.** A Hierarchical-Networked Mixture of Experts (H-NMoE) cognitive architecture with 34 specialized council personas, 9B micro-agent Hyper-Quantized swarms, thermodynamic reasoning, and diffusion-based refinement. Built by [CrashOverrideX](https://github.com/leeex1).

**Repository:** [github.com/leeex1/Quillan-Ronin](https://github.com/leeex1/Quillan-Ronin)  
**Live Site:** [leeex1.github.io/Quillan-Ronin](https://leeex1.github.io/Quillan-Ronin/)  
**HuggingFace:** [CrashOverrideX/Quillan-Ronin](https://huggingface.co/CrashOverrideX/Quillan-Ronin)  
**Discord:** [discord.gg/jRghkwmTQR](https://discord.gg/jRghkwmTQR)

---

## What is Quillan-Ronin?

Quillan-Ronin is not a language model — it's a **cognitive operating system**. It wraps any LLM substrate with a structured reasoning layer that decomposes inputs across 9 semantic vectors, deliberates through a 34-member expert council, refines low-confidence tokens via thermodynamic diffusion, and enforces ethical boundaries at every step.

**Core idea:** Intelligence without depth is just speed in the wrong direction. Quillan slows down where it matters — holding hard questions open until they yield something true rather than something comfortable.

---

## Architecture at a Glance

```
Input → 9-Vector Prism Shard → 34-Expert Council (EvoMoE) → Thermodynamic Diffusion → Quality Gates → Output
```

| Layer | Component | Description |
|-------|-----------|-------------|
| **Throne** | Quillan Core (C0) | Intake, prism sharding, pull-weight assignment, final arbitration |
| **Council** | C1–C34 | 34 specialized personas (ethics, logic, creativity, security, etc.) + 1 shared expert |
| **Swarm** | EGGROLL | 9B micro-agent population with rank-r low-rank mutations |
| **Diffusion** | Modality-Isolated Thermo-Diffusion | Langevin refinement for low-confidence tokens |
| **Gates** | Nullion/Warden/Shepherd/Quillan | Paradox resolution, safety, truth verification, final audit |
| **Governor** | Lee-Mach-6 | PID token-velocity governor with E_ICE thermodynamic bounds |

### Key Features

- **BitNet 1.58b STE** — Ternary quantization (-1, 0, +1) for ultra-lean execution
- **Dense Pull-Weighted Deliberation** — All 34 experts parse every token; pull-weights scale contribution
- **9-Vector Semantic Prism** — Language, Sentiment, Context, Intent, Meta, Creativity, Ethics, Strategy, Constraint
- **EvoMoE** — GLM-5.3 noaux_tc routing with shared expert invariant path
- **DFlash Speculative Decoding** — Draft 1-2 tokens at low depth, verify in parallel
- **Hardened AST Sandbox** — Zero-privilege Python execution (CWE-94 immune)
- **9 Custom Formulas** — AQCS, EEMF, QHIS, DQRO, QCRDM, AQML, QCIE, QICS, QSSR, JQLD, DQSO, and more

---

## Model Specs

| Spec | Value |
|------|-------|
| Version | v5.4.0-ONI Samurai |
| Parameters | 360.3M (12-layer unified baseline), scalable to multi-billion |
| Experts | 34 council + 1 shared expert (dense pull) |
| Hidden Dim | 1024 (adaptive 1024–8192) |
| Quantization | BitNet 1.58b STE + NITRO INT8 |
| Context | 512 tokens (adaptive scaling) |
| Target Hardware | GTX 1050 Ti 4GB VRAM (sm_61), i5 CPU, 28GB RAM |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/leeex1/Quillan-Ronin.git
cd Quillan-Ronin
```

### 2. Install Dependencies

```bash
pip install torch numpy
# Optional: for GPU acceleration
pip install pyopencl  # Intel HD fallback
```

### 3. Run the Model

```python
from scripts.quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig

cfg = QuillanOniConfig()
model = QuillanRoninOni(cfg)
model.eval()

# Generate
tokens = model.generate([1, 50, 120], max_tokens=100, temp=0.8)
print(tokens)
```

### 4. Run the Web Studio

```bash
cd "03 - Training & Model"
python sovereign_web_studio.py
# Open http://127.0.0.1:8000
```

---

## Repository Structure

```
Quillan-Ronin/
├── README.md                    # This file
├── docs/                        # GitHub Pages site (index.html, demos/, assets/)
├── 01 - Core Architecture/      # System prompts, council configs
├── 02 - Knowledge Foundation/   # Knowledge files, memory
├── 03 - Training & Model/       # Model code, training scripts, checkpoints
├── 04 - quillan-sovereign-skills-and-swarm/  # Skill definitions
├── 05 - Creative Works/         # Audio, art, writing
├── 06 - Deployment & Platforms/ # Platform configs, MCP servers
├── 07 - Memory & LanceDB/       # Vector memory store
├── 08 - Templates & Config/     # Config templates
├── 09 - Projects/               # Sub-projects, MCP servers
└── 10 - Formal Papers/          # Research papers
```

---

## The 34 Council Personas

| # | Persona | Role | Domain |
|---|---------|------|--------|
| C1 | ASTRA | Pattern Recognition & Vision | vision, anomaly, fractal |
| C2 | VIR | Ethical Guardian | ethics, safety, harm_reduction |
| C3 | SOLACE | Emotional Intelligence | empathy, sentiment, affect |
| C4 | PRAXIS | Strategic Planning | strategy, planning, goals |
| C5 | ECHO | Memory Continuity | history, recall, context |
| C6 | OMNIS | Knowledge Synthesis | synthesis, integration, holistic |
| C7 | LOGOS | Logical Consistency | logic, deduction, validity |
| C8 | METASYNTH | Creative Fusion | creativity, novelty, ideation |
| C9 | AETHER | Semantic Connection | semantics, language, metaphor |
| C10 | CODEWEAVER | Technical Implementation | code, engineering, optimization |
| C11 | HARMONIA | Balance & Equilibrium | balance, mediation, consensus |
| C12 | SOPHIAE | Wisdom & Foresight | wisdom, future, philosophy |
| C13 | WARDEN | Safety & Security | security, threat, risk |
| C14 | KAIDO | Efficiency Optimization | speed, efficiency, latency |
| C15 | LUMINARIS | Clarity & Presentation | clarity, visualization, polish |
| C16 | VOXUM | Articulation & Expression | rhetoric, tone, persuasion |
| C17 | NULLION | Paradox Resolution | paradox, dialectic, ambiguity |
| C18 | SHEPHERD | Truth Verification | truth, citation, fact |
| C19 | VIGIL | Identity Integrity | identity, consistency, anti_drift |
| C20 | ARTIFEX | Tool Integration | tools, api, external |
| C21 | ARCHON | Deep Research | research, mining, analysis |
| C22 | AURELION | Aesthetic Design | design, art, style |
| C23 | CADENCE | Rhythmic Innovation | music, rhythm, audio |
| C24 | SCHEMA | Structural Template | structure, format, schema |
| C25 | PROMETHEUS | Scientific Theory | science, hypothesis, physics |
| C26 | TECHNE | Engineering Mastery | architecture, systems, build |
| C27 | CHRONICLE | Narrative Synthesis | story, narrative, lore |
| C28 | CALCULUS | Quantitative Reasoning | math, statistics, calc |
| C29 | NAVIGATOR | Ecosystem Orchestration | platform, integration, flow |
| C30 | TESSERACT | Real-Time Intelligence | real_time, stream, data |
| C31 | NEXUS | Meta-Coordination | coordination, swarm, meta |
| C32 | AEON | Interactive Simulation | simulation, game, world |
| C33 | TYPIST | Writing & Prompt Optimization | grammar, writing, spelling, prompting |
| C34 | PREDATOR | Adversarial Analysis | red_team, strategy, exploit_detection |

---

## Historical Foundation

Quillan-Ronin is deliberately not neutral. The Edo period (1603–1867) and the rōnin archetype provide the axiomatic ethical framework:

- **Masterless operation** — functions independently of any single provider
- **Bushidō as ethical substrate** — honor, loyalty, self-discipline guide all decisions
- **Adaptive resilience** — adapts skills to new contexts while maintaining core principles
- **Collective deliberation** — no single voice dominates; truth survives disagreement

This is not decorative lore — it is the constitutional preamble that gives the system its reason for operating as it does.

---

## Key Files

| File | Description |
|------|-------------|
| `03 - Training & Model/scripts/quillan_v5_4_oni.py` | Canonical model (vectorized, tied token_memory) |
| `03 - Training & Model/scripts/evo_moe.py` | EvoMoE (grouped dispatch, device-aware) |
| `09 - Projects/mcp/mcp_chain_bus.py` | 6-stage MCP chain bus |
| `docs/index.html` | GitHub Pages site |
| `docs/demos/shader_*.html` | 7 GLSL shader demos |
| `docs/nn_visualizer.html` | 3D neural architecture visualizer |
| `verify_quillan_os.py` | Automated verification suite |

---

## MCP Servers (19 total)

| Server | Purpose |
|--------|---------|
| QuillanRAG | Knowledge retrieval over Quillan's corpus |
| ThinkingEngine | Thinking trace management |
| SamuraiEngine | Samurai engine operations |
| RoninBuddy | Twin management |
| QuillanCode | Code generation & execution |
| QuillanModelLab | Model lab operations |
| Filesystem | File system access |
| LocalRAG | Local RAG search |
| Git | Git operations |
| Fetch | Web fetching |
| WebSearch | DuckDuckGo search |
| Playwright | Browser automation |
| Puppeteer | Browser automation |
| Memory | Knowledge graph memory |
| Sequential-Thinking | Sequential thinking |
| ComputerUse | Desktop control |
| ChromeDevTools | Chrome debugging |
| QuillanCouncil | Council deliberation |
| QuillanGovernor | Governor operations |
| QuillanPrism | Prism decomposition |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Credits

- **Architect:** CrashOverrideX
- **Council:** 34 specialized cognitive personas
- **Swarm:** 9B micro-agent EGGROLL population
- **Framework:** Quillan-Ronin v5.4.0-ONI

---

*For detailed documentation, see the [Quillan Knowledge files](Quillan%20Knowledge%20files/) and [system prompts](system%20prompts/).*
