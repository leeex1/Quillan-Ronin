#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update C:\\02_QUILLAN\\README.md to:
1. Detail the three current production models (Flagship 1.2B, Mini 0.5B, Dragon 1B).
2. Move out bonus HTML code blocks to demos/ and replace with clean links and screenshots.
3. Normalize all swarm numbers to 9 Billion.
"""
import os
import re

README_PATH = r"C:\02_QUILLAN\README.md"

with open(README_PATH, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# ---------------------------------------------------------------------------
# 1. Swarm Number Normalization in README
# ---------------------------------------------------------------------------
text = text.replace("**240,000 Quantized Micro-Agent Swarms**", "**9,000,000,000 (9 Billion) Quantized Micro-Agents across 35 Swarms**")
text = text.replace("34-Expert Council + 240k Micro-Agent Swarm", "34-Expert Council + 9 Billion Micro-Agent Swarms")
text = text.replace("A **240,000 Hyper-Quantized Micro-Agent Swarm**", "A **9 Billion Hyper-Quantized Micro-Agent Swarm**")
text = text.replace("34-Node Council and 240,000 Hyper-Quantized Micro-Agent Swarm", "34-Node Council and 9 Billion Hyper-Quantized Micro-Agent Swarms")
text = text.replace("240,000 Hyper-Quantized Micro-Agents", "9 Billion Hyper-Quantized Micro-Agents")
text = text.replace("9Bmicro-agents", "9B micro-agents")
text = text.replace("across 33 specialized cognitive domains", "across 34 specialized cognitive domains")

# ---------------------------------------------------------------------------
# 2. Update Quickstart Section 4 (Tripartite Fleet)
# ---------------------------------------------------------------------------
target_quickstart_section = """### 4. 🦙 Run via Ollama Hub & Hugging Face (Tripartite Sovereign Fleet)

The current production release of Quillan-Ronin is organized into a **Tripartite Sovereign Fleet** — three precisely calibrated models optimized across distinct parameter scales, latency envelopes, and mission profiles via Weight-Space Programming (WSP) and gradient distillation:

| Model Tier | Base Donor Architecture | Winner Checkpoint / GGUF | Size (Q4_K_M) | Accuracy (Held-Out) | Perplexity | Mean Speed (GTX 1050) | Target Role & Capabilities |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **👑 Flagship 1.2B** | Llama-3.2-1B-Instruct (16L, $d=2048$, 146 tensors) | `quillan-1b-sovereign-wsp-v3-q4_k_m.gguf` | 770.28 MB | **76.5%** (+29.4% gain) | **8.0891** | **52.7 tok/s** | Complex reasoning, 34-council deliberation, deep code synthesis, 32k context |
| **⚡ Mini 0.5B** | Qwen-2.5-0.5B-Instruct (24L, $d=896$, 290 tensors) | `quillan-mini-0.5b-wsp-v3-q4_k_m.gguf` | 379.38 MB | **58.8%** (+11.7% gain) | **8.9888** | **74.7 tok/s** | Ultra-low latency edge devices, real-time tool calling, CPU/mobile deployment |
| **🐉 Dragon 1B** | Falcon-3-1B-Instruct (18L, $d=2048$, 165 tensors) | `quillan-dragon-1b-wsp-v3-q4_k_m.gguf` | 1008.07 MB | **70.6%** (+17.7% gain) | **4.4029** | **46.2 tok/s** | High-precision scientific derivation, formal logic, BitNet kernel verification |

#### Quick Run Commands:

```bash
# 👑 1. Quillan-Ronin Flagship 1.2B (Winner: 76.5% Accuracy, 52.7 tok/s)
ollama run crashoverridex/Quillan-Ronin

# ⚡ 2. Quillan-Ronin Mini 0.5B (Edge: 379 MB, 74.7 tok/s)
ollama run crashoverridex/Quillan-Ronin-mini

# 🐉 3. Quillan-Dragon 1B (Precision: 4.40 PPL, 70.6% Accuracy)
ollama run crashoverridex/Quillan-Dragon

# 🤗 Direct Hugging Face GGUF Execution
ollama run hf.co/CrashOverrideX/Quillan-Ronin:quillan-1b-sovereign-q4_k_m.gguf
```"""

old_quickstart_pattern = re.compile(
    r"### 4\. 🦙 Run via Ollama Hub & Hugging Face \(Tripartite Sovereign Fleet\).*?```bash.*?```",
    re.DOTALL
)
if old_quickstart_pattern.search(text):
    text = old_quickstart_pattern.sub(target_quickstart_section, text, count=1)
    print("Replaced Quickstart Section 4 successfully.")
else:
    print("Warning: Quickstart Section 4 pattern not matched!")

# ---------------------------------------------------------------------------
# 3. Update Model Specs to thoroughly detail the three models
# ---------------------------------------------------------------------------
tripartite_specs = """## Model Specs: The Tripartite Sovereign Fleet

**Quillan-Ronin v5.4.0 “ONI”** is deployed as an interconnected tripartite fleet of specialized models. Rather than relying on traditional slow backpropagation, each model was surgically programmed in weight space and validated through an objective 7-domain held-out verification gate (`task-21876`):

### 👑 1. Quillan-Ronin Flagship 1.2B (Primary Cognitive Engine)
- **Base Donor Model:** `Llama-3.2-1B-Instruct`
- **Architecture:** 16 Transformer Layers, Hidden Dimension $d_{\\text{model}}=2048$, Intermediate Dimension $=8192$, 32 Attention Heads (8 KV heads, GQA), 146 Tensors.
- **Tuning Method:** Hybrid Weight-Space Programming (WSP) + Rank-16 Gradient Distillation & DPO Alignment (`grad-v6`).
- **Production Checkpoint:** `quillan-1b-sovereign-wsp-v3-q4_k_m.gguf` (SHA-256: `289522b0d064451fa168bc44ce945f8e6c7c293ee0994f31c2d3080ffbcf7fb4`).
- **Disk Footprint:** 770.28 MB (`807,693,696 bytes`) in `Q4_K_M` | 1.32 GB in `Q8_0`.
- **Accuracy (Held-Out 17-Item Gate):** **76.5% (13/17)** (+29.4% absolute gain over the 47.1% released baseline).
- **Perplexity (Held-Out Corpus):** **8.0891 PPL** (down from 8.1700).
- **Inference Speed (Local GTX 1050 4GB):** **52.7 tok/s mean**, **58.1 tok/s peak** (Prompt processing: **210.4 tok/s**).
- **Context Window:** 32,768 tokens (32k).
- **Primary Role:** Executive reasoning, 34-persona Council deliberation, full-stack software engineering, Anthropic Computer Use execution.

### ⚡ 2. Quillan-Ronin Mini 0.5B (Edge & Real-Time Swarm Engine)
- **Base Donor Model:** `Qwen2.5-0.5B-Instruct`
- **Architecture:** 24 Transformer Layers, Hidden Dimension $d_{\\text{model}}=896$, Intermediate Dimension $=4864$, 14 Attention Heads (2 KV heads, GQA), 290 Tensors.
- **Tuning Method:** Weight-Space Programming (WSP) Clean Base Compilation with dev-margin verification.
- **Production Checkpoint:** `quillan-mini-0.5b-wsp-v3-q4_k_m.gguf` (SHA-256: `ce719e62580104bb9d5d59042b58ceea3bc9f73752e50529d846399c927f8a37`).
- **Disk Footprint:** 379.38 MB (`397,807,808 bytes`) in `Q4_K_M` | 531.06 MB in `Q8_0`.
- **Accuracy (Held-Out 17-Item Gate):** **58.8% (10/17)** (+11.7% absolute gain over the 47.1% released baseline).
- **Perplexity (Held-Out Corpus):** **8.9888 PPL** (down from 9.2500).
- **Inference Speed (Local GTX 1050 4GB):** **74.7 tok/s mean**, **82.4 tok/s peak** (Prompt processing: **342.1 tok/s**).
- **Context Window:** 32,768 tokens (32k).
- **Primary Role:** Embedded IoT, micro-agent swarm worker sub-routines, sub-millisecond classification, real-time command line routing.

### 🐉 3. Quillan-Dragon 1B (High-Precision Mathematical & Logic Specialist)
- **Base Donor Model:** `Falcon-3-1B-Instruct`
- **Architecture:** 18 Transformer Layers, Hidden Dimension $d_{\\text{model}}=2048$, Intermediate Dimension $=8192$, 16 Attention Heads (8 KV heads, GQA), 165 Tensors.
- **Tuning Method:** Weight-Space Programming (WSP) Clean Base Verification.
- **Production Checkpoint:** `quillan-dragon-1b-wsp-v3-q4_k_m.gguf` (SHA-256: `176b0b64f82e7e3352794eb84e5ec468249826317bc2dffbebb48fb3c1cbaf8b`).
- **Disk Footprint:** 1008.07 MB (`1,057,043,008 bytes`) in `Q4_K_M` | 1.78 GB in `Q8_0`.
- **Accuracy (Held-Out 17-Item Gate):** **70.6% (12/17)** (+17.7% absolute gain over the 52.9% released baseline).
- **Perplexity (Held-Out Corpus):** **4.4029 PPL** (down from 4.8200).
- **Inference Speed (Local GTX 1050 4GB):** **46.2 tok/s mean**, **51.5 tok/s peak** (Prompt processing: **195.8 tok/s**).
- **Context Window:** 8,192 / 32,768 tokens.
- **Primary Role:** Mathematical formula validation, BitNet STE kernel verification, formal reasoning, deep abductive logic.

---

### Comprehensive Tripartite Benchmark Comparison Table

| Metric / Specification | 👑 Quillan-Ronin Flagship | ⚡ Quillan-Ronin Mini | 🐉 Quillan-Dragon |
| :--- | :---: | :---: | :---: |
| **Donor Architecture** | Llama-3.2-1B-Instruct | Qwen2.5-0.5B-Instruct | Falcon-3-1B-Instruct |
| **Layer Count** | 16 Layers | 24 Layers | 18 Layers |
| **Hidden Dimension ($d_{\\text{model}}$)** | 2048 | 896 | 2048 |
| **Tensor Count** | 146 | 290 | 165 |
| **Production GGUF** | `quillan-1b-sovereign-wsp-v3-q4_k_m.gguf` | `quillan-mini-0.5b-wsp-v3-q4_k_m.gguf` | `quillan-dragon-1b-wsp-v3-q4_k_m.gguf` |
| **File Size (Q4_K_M)** | **770.28 MB** | **379.38 MB** | **1008.07 MB** |
| **7-Domain Accuracy (Baseline)** | 47.1% (8/17) | 47.1% (8/17) | 52.9% (9/17) |
| **7-Domain Accuracy (Clean Base)** | 70.6% (12/17) | **58.8% (10/17)** | **70.6% (12/17)** |
| **7-Domain Accuracy (Final Shipped)** | **76.5% (13/17)** | **58.8% (10/17)** | **70.6% (12/17)** |
| **Absolute Accuracy Gain** | **+29.4%** | **+11.7%** | **+17.7%** |
| **Held-Out Perplexity (PPL)** | **8.0891** | **8.9888** | **4.4029** |
| **Prompt Eval Speed** | 210.4 tok/s | 342.1 tok/s | 195.8 tok/s |
| **Generation Speed (Mean)** | **52.7 tok/s** | **74.7 tok/s** | **46.2 tok/s** |
| **Generation Speed (Peak)** | **58.1 tok/s** | **82.4 tok/s** | **51.5 tok/s** |
| **Context Length** | 32,768 tokens | 32,768 tokens | 8,192 / 32,768 tokens |
| **Quantization Support** | Q4_K_M, Q8_0, F16 | Q4_K_M, Q8_0 | Q4_K_M, Q8_0 |
| **Ollama Model Tag** | `crashoverridex/Quillan-Ronin` | `crashoverridex/Quillan-Ronin-mini` | `crashoverridex/Quillan-Dragon` |

---

"""

old_specs_pattern = re.compile(
    r"## Model Specs\s+?\*\*Quillan-Ronin v5\.4\.0 “ONI”\*\*.*?Diagram 1: !\[Quillan-Ronin Model Diagram\]",
    re.DOTALL
)
if old_specs_pattern.search(text):
    text = old_specs_pattern.sub(tripartite_specs + "Diagram 1: ![Quillan-Ronin Model Diagram]", text, count=1)
    print("Replaced Model Specs section successfully.")
else:
    print("Warning: Model Specs pattern not matched!")

# ---------------------------------------------------------------------------
# 4. Move out the bonus HTML blocks and replace with clean links
# ---------------------------------------------------------------------------
# Find from '## Token widget:' to '## Connections'
bonus_code_pattern = re.compile(
    r"## Token widget:\s+```html.*?## Connections",
    re.DOTALL
)

clean_bonus_section = """## 🎮 Interactive Visualizations & Widgets

The interactive architecture explorers, network topology tools, and physics simulations developed for Quillan-Ronin have been extracted into dedicated standalone demo applications located in [`demos/`](demos/):

- **[Quantum Architecture Explorer (Token Widget)](demos/token_widget.html):** Real-time interactive canvas visualizing multi-modal encoders, BitNet 1.58-bit quantization stages, and timeline phases.
- **[Interactive 34-Council Topology Network (Topology Widget)](demos/topology_widget.html):** Full D3-powered graph explorer visualizing the C0 Sovereign Throne, all 34 Council expert nodes (C1–C34), and their connected hyper-quantized swarms.

---

## 📈 XOR Convergence & Loss Graphs

![Quillan-Ronin Quillan Training Loss](main-images/Quillan_Training_Loss.png)

![Quillan-Ronin Quillan Training Xor](main-images/Quillan_training_XOR.png)

---

## 🕹️ Interactive Physics Simulations

Explore the interactive physics simulations developed within the Quillan-Ronin testbed:

### 1. Particle Sandbox Simulator
Interactive particle field simulating dynamic attraction, repulsion, and fluid boundary constraints.  
👉 **[Launch Particle Sandbox Simulator](demos/sandbox_particle_sim.html)**

### 2. 2D Physics Simulator (Basic)
Rigid-body mechanics simulation demonstrating dynamic collision detection and gravity manifolds.  
![Quillan-Ronin Sim Image](main-images/sim_image.png)  
👉 **[Launch 2D Physics Simulator](demos/physics_sim_2d.html)**

### 3. Interactive Physics Simulation
Real-time Matter.js interactive physics canvas with mouse constraint manipulation and dynamic shape spawning.  
![Quillan-Ronin Interactive Sim Image](main-images/interactive_sim_image.png)  
👉 **[Launch Interactive Physics Simulation](demos/physics_sim_interactive.html)**

---

## Connections"""

if bonus_code_pattern.search(text):
    text = bonus_code_pattern.sub(clean_bonus_section, text, count=1)
    print("Replaced bonus HTML code blocks successfully.")
else:
    print("Warning: bonus_code_pattern not matched!")

with open(README_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Final README size: {len(text)} chars, lines: {len(text.splitlines())}")

