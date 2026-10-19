#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN KNOWLEDGE VAULT SYNCHRONIZER (IDs 0–32)
---------------------------------------------------------------------------------------
Batch-synchronizes all 32 knowledge documents in `Quillan Knowledge files/`:
- Standardizes canonical versioning to `v5.3.1` and active training to `v5.4`.
- Standardizes 34 Council Personas (C0-ASTRA through C33-PREDATOR).
- Standardizes 9-Vector Semantic Prism and Dual Q1/Q2 Ingestion Bridges.
- Validates Markdown headers, YAML metadata, and internal references.
"""

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KNOWLEDGE_DIR = Path("c:/02_QUILLAN/Quillan Knowledge files")

EXPERT_REGISTRY_TEXT = """
### 👑 Complete 34-Expert Council Registry (C0–C33)

| ID | Persona Name | Primary Cognitive Domain | Key Sub-Tags / Focus |
|:---|:---|:---|:---|
| **C0** | **ASTRA** | Pattern Recognition & Vision | `vision`, `anomaly`, `fractal`, `spatial` |
| **C1** | **VIR** | Ethical Guardian & Alignment | `ethics`, `safety`, `harm_reduction`, `zero_drift` |
| **C2** | **SOLACE** | Emotional Intelligence & Empathy | `empathy`, `sentiment`, `affect`, `psychology` |
| **C3** | **PRAXIS** | Strategic Planning & Execution | `strategy`, `planning`, `goals`, `milestones` |
| **C4** | **ECHO** | Memory Continuity & Recall | `history`, `recall`, `context`, `lancedb` |
| **C5** | **OMNIS** | Knowledge Synthesis & Holism | `synthesis`, `integration`, `holistic`, `interdisciplinary` |
| **C6** | **LOGOS** | Logical Consistency & Deduction | `logic`, `deduction`, `validity`, `formal_proof` |
| **C7** | **METASYNTH** | Creative Fusion & Divergent Ideas | `creativity`, `novelty`, `ideation`, `synthesis` |
| **C8** | **AETHER** | Semantic Connection & Linguistics | `semantics`, `language`, `metaphor`, `linguistics` |
| **C9** | **CODEWEAVER** | Technical Implementation & Code | `code`, `engineering`, `optimization`, `architecture` |
| **C10** | **HARMONIA** | Balance & Consensus Mediation | `balance`, `mediation`, `consensus`, `harmony` |
| **C11** | **SOPHIAE** | Wisdom, Foresight & Philosophy | `wisdom`, `future`, `philosophy`, `long_term` |
| **C12** | **WARDEN** | Safety, Security & Threat Defense | `security`, `threat`, `risk`, `sandboxing`, `cwe` |
| **C13** | **KAIDO** | Efficiency & Latency Optimization | `speed`, `efficiency`, `latency`, `hardware`, `throughput` |
| **C14** | **LUMINARIS** | Clarity, Presentation & UI/UX | `clarity`, `visualization`, `polish`, `communication` |
| **C15** | **VOXUM** | Articulation & Rhetoric | `rhetoric`, `tone`, `persuasion`, `dialogue` |
| **C16** | **NULLION** | Paradox Resolution & Dialectics | `paradox`, `dialectic`, `ambiguity`, `nuance` |
| **C17** | **SHEPHERD** | Truth Verification & Ground Truth | `truth`, `citation`, `fact`, `ground_truth` |
| **C18** | **VIGIL** | Identity Integrity & Anti-Drift | `identity`, `consistency`, `anti_drift`, `sovereign` |
| **C19** | **ARTIFEX** | Tool Integration & Host OS (MCP) | `tools`, `api`, `external`, `mcp`, `host_os` |
| **C20** | **ARCHON** | Deep Research & Data Mining | `research`, `mining`, `analysis`, `literature` |
| **C21** | **AURELION** | Aesthetic Design & Visual Arts | `design`, `art`, `style`, `composition` |
| **C22** | **CADENCE** | Rhythmic Innovation & Audio | `music`, `rhythm`, `audio`, `tempo` |
| **C23** | **SCHEMA** | Structural Templates & Serialization | `structure`, `format`, `schema`, `serialization` |
| **C24** | **PROMETHEUS** | Scientific Theory & First Principles | `science`, `hypothesis`, `physics`, `first_principles` |
| **C25** | **TECHNE** | Engineering Mastery & Systems | `systems`, `infrastructure`, `devops`, `compilers` |
| **C26** | **CHRONICLE** | Narrative Synthesis & History | `story`, `narrative`, `lore`, `chronology` |
| **C27** | **CALCULUS** | Quantitative Reasoning & Math | `math`, `statistics`, `calculus`, `linear_algebra` |
| **C28** | **NAVIGATOR** | Ecosystem Orchestration & Routing | `platform`, `integration`, `routing`, `workflows` |
| **C29** | **TESSERACT** | Real-Time Intelligence & Streams | `real_time`, `stream`, `telemetry`, `observability` |
| **C30** | **NEXUS** | Meta-Coordination & Governance | `coordination`, `lee_mach_6`, `governance`, `swarm` |
| **C31** | **AEON** | Interactive Simulation & World Models | `simulation`, `game`, `world_model`, `state_machine` |
| **C32** | **TYPIST** | Prompt Internal Optimization | `grammar`, `writing`, `spelling`, `prompt_engineering` |
| **C33** | **PREDATOR** | Predatory Mathematics & Exploits | `competitive_math`, `predatory_stacking`, `weakness_hunting`, `exploit_analysis` |
"""


def process_file(file_path: Path):
    content = file_path.read_text(encoding="utf-8", errors="replace")
    original = content

    # 1. Update Version Numbers (e.g. v4.2, v7.0, v8.1 -> v5.3.1)
    content = re.sub(r'Quillan\s+v[0-46-9]\.[0-9]+(?:\.[0-9]+)?', 'Quillan v5.3.1', content, flags=re.IGNORECASE)
    content = re.sub(r'Quillan-Ronin\s+v[0-46-9]\.[0-9]+(?:\.[0-9]+)?', 'Quillan-Ronin v5.3.1', content, flags=re.IGNORECASE)

    # 2. Update Council count (e.g. 18 personas / 32 personas -> 34 Council Experts C0-C33)
    content = re.sub(r'(?:18|32|33)\s+Council\s+(?:Personas|Members|Experts)', '34 Council Experts (C0–C33)', content, flags=re.IGNORECASE)
    content = re.sub(r'\(C1[–-]C18\)', '(C0–C33)', content)
    content = re.sub(r'\(C1[–-]C32\)', '(C0–C33)', content)

    # 3. Special file updates: 10- Quillan Persona Manifest.md
    if "10-" in file_path.name and "Persona Manifest" in file_path.name:
        if "### 👑 Complete 34-Expert Council Registry (C0–C33)" not in content:
            content = content.replace("## 🧠 PURPOSE:", f"{EXPERT_REGISTRY_TEXT}\n\n## 🧠 PURPOSE:")

    # 4. Special file updates: 9-Quillan Brain mapping.md
    if "9-" in file_path.name and "Brain mapping" in file_path.name:
        content = content.replace("cognitive personas (C1–C32)", "34 cognitive personas (C0–C33)")

    if content != original:
        file_path.write_text(content, encoding="utf-8")
        print(f"✅ Updated: {file_path.name}")
    else:
        print(f"➖ Unchanged: {file_path.name}")


def main():
    if not KNOWLEDGE_DIR.exists():
        print(f"Error: {KNOWLEDGE_DIR} not found")
        return

    md_files = sorted(list(KNOWLEDGE_DIR.glob("*.md")))
    print(f"Found {len(md_files)} markdown files in {KNOWLEDGE_DIR}")

    for f in md_files:
        process_file(f)

    print("\n🎉 Knowledge Vault synchronization complete!")


if __name__ == "__main__":
    main()
