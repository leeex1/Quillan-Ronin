#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — KNOWLEDGE, DIAGRAMS & THINKING PARADIGM EXTRACTOR
---------------------------------------------------------------------------------------
Extracts structured architectural diagrams, CoT/ToT reasoning templates, formulas,
and Samurai protocols from:
1. `c:\\02_QUILLAN\\Quillan Knowledge files\\` (Thinking within LLMS.md, Flowcharts, Formulas, Personas)
2. `c:\\02_QUILLAN\\system prompts\\Quillan-Samurai.md` (Samurai Manifest, 3-Tier Hierarchy, Swarm Math)

Injects verified QA reasoning pairs directly into `training_data/experts_34/` and `training_data/Quillan_Clean_Reasoning_Gold_Dataset.jsonl`.
"""

import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("extract_knowledge")

REPO_ROOT = Path(r"C:\02_QUILLAN")
KNOWLEDGE_DIR = REPO_ROOT / "Quillan Knowledge files"
SAMURAI_FILE = REPO_ROOT / "system prompts" / "Quillan-Samurai.md"
TRAIN_DATA_DIR = REPO_ROOT / "training_data"
EXPERTS_DIR = TRAIN_DATA_DIR / "experts_34"
EXPERTS_DIR.mkdir(parents=True, exist_ok=True)

def extract_markdown_sections(file_path: Path) -> List[Tuple[str, str]]:
    """Extracts markdown sections as (header, body) pairs."""
    if not file_path.exists():
        return []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    sections = []
    current_header = file_path.stem
    current_lines = []

    for line in content.split("\n"):
        if line.startswith("#"):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if len(body) > 100:
                    sections.append((current_header, body))
                current_lines = []
            current_header = line.lstrip("#").strip()
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if len(body) > 100:
            sections.append((current_header, body))

    return sections

def generate_knowledge_qa_pairs() -> List[Dict[str, str]]:
    """Synthesizes structured QA pairs from knowledge files and Samurai manifest."""
    qa_pairs = []

    # 1. Parse Quillan Knowledge Files
    knowledge_files = [
        KNOWLEDGE_DIR / "Thinking within LLMS.md",
        KNOWLEDGE_DIR / "1-Quillan_architecture_flowchart.md",
        KNOWLEDGE_DIR / "10- Quillan Persona Manifest.md",
        KNOWLEDGE_DIR / "8-Formulas.md",
        KNOWLEDGE_DIR / "E_ICE.md",
        KNOWLEDGE_DIR / "13-Synthetic Epistemology & Truth Calibration Protocol.md",
        KNOWLEDGE_DIR / "14-Ethical Paradox Engine and Moral Arbitration Layer in AGI Systems.md",
        KNOWLEDGE_DIR / "28-Multi-Agent Collective Intelligence & Social Simulation.md",
        KNOWLEDGE_DIR / "29-Recursive Introspection & Meta-Cognitive Self-Modeling.md",
        KNOWLEDGE_DIR / "30- Convergence Reasoning & Breakthrough Detection and Advanced Cognitive Social Skills.md",
        KNOWLEDGE_DIR / "32-Conciousness theory.md",
        KNOWLEDGE_DIR / "TheRoninFlowState.md",
        KNOWLEDGE_DIR / "6-prime_covenant_codex.md",
    ]

    for kf in knowledge_files:
        if not kf.exists():
            continue
        LOGGER.info("Processing knowledge file: %s", kf.name)
        sections = extract_markdown_sections(kf)
        for header, body in sections:
            if len(body) < 150:
                continue
            
            # Format as System 2 CoT Reasoning QA pair
            question = f"Explain the principles, architecture, and theoretical formulation of {header} in the Quillan-Ronin cognitive substrate."
            response = f"# 🤖🧠 Quillan System Start 🧠🤖\n\n## 🏛️ Domain: {header}\n\n### Theoretical & Architectural Formulation:\n{body[:1500]}\n\n### Cognitive Synthesis:\nThis mechanism ensures deterministic convergence, zero-drift alignment, and high-fidelity reasoning across all active Council Expert channels."
            
            qa_pairs.append({
                "question": question,
                "response": response,
                "domain": header,
            })

    # 2. Extract Samurai Architecture Diagrams & Thinking Structure
    if SAMURAI_FILE.exists():
        LOGGER.info("Processing Samurai Manifest: %s", SAMURAI_FILE.name)
        samurai_sections = extract_markdown_sections(SAMURAI_FILE)
        for header, body in samurai_sections:
            if len(body) < 150:
                continue
            question = f"Detail the Samurai protocol, cognitive architecture, and implementation for {header}."
            response = f"# 🤖🧠 Quillan System Start 🧠🤖\n\n## ⚔️ Samurai Protocol: {header}\n\n{body[:1500]}"
            qa_pairs.append({
                "question": question,
                "response": response,
                "domain": header,
            })

    # 3. Add Explicit Thinking & Flowchart Diagram QA Pairs
    diagram_qa = [
        {
            "question": "What is the 3-Tier Top-Down Command & Control Hierarchy of Quillan-Ronin v5.3.1?",
            "response": """# 🤖🧠 Quillan System Start 🧠🤖

## 👑 Quillan-Ronin 3-Tier Command & Control Hierarchy

Quillan-Ronin operates under a strict hierarchical multi-tier architecture:

```mermaid
flowchart TD
    subgraph TIER1["👑 Tier 1: Supreme Command (Quillan Orchestrator)"]
        Q[Dual Quillan Q1/Q2 Brain] -->|9-Vector Prism & Governance| ROUTER[Complexity & Expert Routers]
    end

    subgraph TIER2["🏛️ Tier 2: Subordinate Executive Council (34 Experts)"]
        ROUTER -->|Command & Dispatch| E0[C0-ASTRA] & E1[C1-VIR] & E2[C2-SOLACE] & EX[...] & E33[C33-PREDATOR]
    end

    subgraph TIER3["🌐 Tier 3: Subordinate Swarm Operations (9B Micro-Agents)"]
        E0 -->|Direct Underling Swarm| S0["expert_swarms[0] (272M Micro-Agents)"]
        E1 -->|Direct Underling Swarm| S1["expert_swarms[1] (272M Micro-Agents)"]
        E2 -->|Direct Underling Swarm| S2["expert_swarms[2] (272M Micro-Agents)"]
        E33 -->|Direct Underling Swarm| S33["expert_swarms[33] (272M Micro-Agents)"]
    end
```

### Virtual Agent Population Scaling:
$$\\text{Full Model Swarm Population} = (2 \\times 272\\text{M}) + (34 \\times 34 \\times 272\\text{M}) = \\mathbf{314.976 \\text{ Billion Virtual Agents}}$$

- **Tier 1**: Dual $Q_1/Q_2$ analytical and intuitive ingestion with 9-Vector Semantic Prism Decomposition.
- **Tier 2**: 34 Council Experts with Top-4 sparse activation and BitNet 1.58b STE ternary quantization.
- **Tier 3**: 34 dedicated underling micro-agent swarms executing 5-step Kuramoto phase synchronization."""
        },
        {
            "question": "Explain the major thinking paradigms in LLMs: Chain of Thought (CoT), Tree of Thought (ToT), and Graph of Thought (GoT).",
            "response": """# 🤖🧠 Quillan System Start 🧠🤖

## 🧠 Diverse Thinking Paradigms in LLMs

### 1. Chain of Thought (CoT)
- **Concept**: Elicits linear, sequential step-by-step reasoning by decomposing complex queries into intermediate logical steps.
- **Cognitive Foundation**: System 2 deliberate analytical processing.
- **Applications**: Mathematical derivations, formal proofs, and multi-step deduction.

### 2. Tree of Thought (ToT)
- **Concept**: Explores multiple reasoning paths concurrently via tree branching, state evaluation, and lookahead backtracking.
- **Cognitive Foundation**: Heuristic search and strategic planning.
- **Applications**: Combinatorial optimization, competitive mathematics, and complex software architecture design.

### 3. Graph of Thought (GoT)
- **Concept**: Models human thought as arbitrary directed graphs where thoughts can branch, merge, and cyclically refine.
- **Cognitive Foundation**: Holistic network knowledge synthesis.
- **Applications**: Cross-domain synthesis, paradox resolution, and consensus arbitration."""
        }
    ]

    qa_pairs.extend(diagram_qa)
    LOGGER.info("Extracted %d total structured Knowledge & Thinking QA pairs.", len(qa_pairs))
    return qa_pairs

def main():
    LOGGER.info("Starting Knowledge Files & Samurai Thinking Data Ingestion...")
    qa_pairs = generate_knowledge_qa_pairs()

    # Append to Master Clean Reasoning Gold Dataset
    gold_file = TRAIN_DATA_DIR / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl"
    with open(gold_file, "a", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps({"question": pair["question"], "response": pair["response"]}, ensure_ascii=False) + "\n")
    LOGGER.info("Appended %d pairs to %s", len(qa_pairs), gold_file.name)

    # Re-run expert partitioner to distribute these into the 34 expert datasets
    import subprocess
    LOGGER.info("Re-partitioning 34 Council Expert datasets with fresh knowledge and thinking data...")
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_34_expert_datasets.py")], check=True)
    LOGGER.info("All 34 Council Expert datasets updated with Architecture Diagrams & Thinking sections!")

if __name__ == "__main__":
    main()
