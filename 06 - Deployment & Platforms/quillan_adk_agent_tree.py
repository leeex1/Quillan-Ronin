"""
Quillan-Ronin Sovereign ADK Multi-Agent Hierarchy
=================================================
Canonical Agent Development Kit (ADK) / Google GenAI Agent / OpenCode Tree
Architecture: 3-Tier Fractal Hierarchy (Root -> 34 Council -> 35 Swarms + FORGE)
"""

from __future__ import annotations
import sys
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Configurable Default Model for ADK Execution
DEFAULT_MODEL = os.getenv("QUILLAN_MODEL", "nvidia/nemotron-3-super-120b-a12b")

@dataclass
class Agent:
    """
    Standard Agent specification compatible with Google ADK, Google GenAI Agents,
    OpenAI Swarm, and OpenCode subagent protocols.
    """
    name: str
    model: str = DEFAULT_MODEL
    instruction: str = ""
    role: str = ""
    tier: int = 1  # 1: Root, 2: Council, 3: Swarm, 0: Specialist
    sub_agents: List['Agent'] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, recursive: bool = True) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "model": self.model,
            "role": self.role,
            "tier": self.tier,
            "metadata": self.metadata,
            "tools_count": len(self.tools),
            "sub_agents_count": len(self.sub_agents),
        }
        if recursive:
            data["sub_agents"] = [sa.to_dict(recursive=True) for sa in self.sub_agents]
        return data

# ==============================================================================
# 1. 35 MCP TOOL SERVER DEFINITIONS
# ==============================================================================
MCP_SERVERS = [
    {"name": "Filesystem", "type": "local", "purpose": "Zero-leak file I/O & directory tree"},
    {"name": "WebSearch", "type": "local", "purpose": "Live external web search"},
    {"name": "Fetch", "type": "local", "purpose": "Direct HTTP / URL fetching"},
    {"name": "Memory", "type": "local", "purpose": "Persistent cross-session entity memory"},
    {"name": "Git", "type": "local", "purpose": "Repository version control & commits"},
    {"name": "Playwright", "type": "local", "purpose": "Headless browser automation & screenshots"},
    {"name": "ComputerUse", "type": "local", "purpose": "Windows GUI automation & input"},
    {"name": "ChromeDevTools", "type": "local", "purpose": "Live browser inspection & profiling"},
    {"name": "QuillanThinkingEngine", "type": "local", "purpose": "Closed-loop cognitive traces & reasoning"},
    {"name": "QuillanSamuraiEngine", "type": "local", "purpose": "Bushido cycle steps & gate verification"},
    {"name": "QuillanBuddy", "type": "local", "purpose": "RoninBuddy dual-agent sparring & consensus"},
    {"name": "QuillanRAG", "type": "local", "purpose": "LanceDB sovereign vector embeddings"},
    {"name": "LocalRAG", "type": "local", "purpose": "Local directory indexing & hybrid search"},
    {"name": "DeepWiki", "type": "remote", "purpose": "External deep scientific & wiki knowledge"},
    {"name": "HuggingFace", "type": "remote", "purpose": "Model hubs & dataset card inspections"},
    {"name": "SequentialThinking", "type": "local", "purpose": "Linear step-by-step hypothesis testing"},
    {"name": "Time", "type": "local", "purpose": "Deterministic local timezone awareness"},
    {"name": "MemPalace", "type": "local", "purpose": "Long-term AAAK memory palace storage"},
    {"name": "GitNexus", "type": "local", "purpose": "Graph-based code symbol & caller topology"},
    {"name": "QuillanAudio", "type": "local", "purpose": "Digital signal processing & audio cue synthesis"},
    {"name": "QuillanCode", "type": "local", "purpose": "AST analysis, linting & code repair"},
    {"name": "QuillanCouncil", "type": "local", "purpose": "Pull-weight arbitration & cluster debates"},
    {"name": "QuillanGameDev", "type": "local", "purpose": "Game engine architecture & balance tables"},
    {"name": "QuillanGovernor", "type": "local", "purpose": "Resource budgeting, PID velocity & hardware telemetry"},
    {"name": "QuillanMedia", "type": "local", "purpose": "Visual design briefs & aesthetic generation"},
    {"name": "QuillanModelLab", "type": "local", "purpose": "Checkpoint forensics, quantization & training graphs"},
    {"name": "QuillanOps", "type": "local", "purpose": "CI/CD pipelines, container limits & task graphs"},
    {"name": "QuillanPrism", "type": "local", "purpose": "9-vector semantic refraction & sharding"},
    {"name": "QuillanSpider", "type": "local", "purpose": "High-throughput workspace crawling & sweeping"},
    {"name": "QuillanTeam", "type": "local", "purpose": "Multi-agent crew spawning & task handoffs"},
    {"name": "QuillanWeb", "type": "local", "purpose": "Full-stack frontend scaffolding & auditing"},
    {"name": "QuillanRE", "type": "local", "purpose": "Binary decompilation & reverse-engineering forensics"},
    {"name": "QuillanOrchestrator", "type": "local", "purpose": "Multi-stage pipeline chaining & retry logic"},
    {"name": "QuillanOrochi", "type": "local", "purpose": "Evolutionary weight optimization & prompt mutations"},
    {"name": "Puppeteer", "type": "local", "purpose": "Secondary headless browser scraping fallback"},
]

def load_prompt(filename: str, fallback: str = "") -> str:
    search_dirs = [
        Path(r"C:\02_QUILLAN\.opencode\agent"),
        Path(r"C:\02_QUILLAN\06 - Deployment & Platforms\system prompts"),
        Path(r"C:\02_QUILLAN"),
    ]
    for d in search_dirs:
        fp = d / filename
        if fp.exists():
            try:
                return fp.read_text(encoding="utf-8")
            except Exception:
                pass
    return fallback

# ==============================================================================
# 2. TIER 3: HYPER-QUANTIZED SWARM AGENTS (35 SWARMS / 272M EACH)
# ==============================================================================
def create_swarm_agent(council_id: str, name: str, domain: str) -> Agent:
    swarm_name = f"{council_id}-swarm"
    instruction = load_prompt(
        f"{swarm_name}.md",
        f"You are {swarm_name}: 272M micro-agents cloned from {name}. "
        f"Operating via BitNet 1.58b STE ternary quantization and EGGROLL rank-32 low-rank perturbations. "
        f"Domain: {domain}. Report empirical findings strictly up to parent {council_id}."
    )
    return Agent(
        name=swarm_name,
        role=f"{domain} Hyper-Quantized Swarm",
        tier=3,
        instruction=instruction,
        metadata={
            "parent": council_id,
            "micro_agents": 272_000_000,
            "quantization": "bitnet-1.58b-ste",
            "mutation_rank": 32,
            "activation_top_k": 19,
            "dynamically_reprogrammable": True,
        }
    )

# Quillan's Private Internal Think Tank Swarm (Swarm 0)
quillan_swarm = Agent(
    name="quillan-swarm",
    role="Sovereign Core Swarm & Private Internal Think Tank",
    tier=3,
    instruction=load_prompt("quillan-swarm.md", "Quillan Core Swarm (c0-quillan-swarm): 272M sovereign micro-agents."),
    metadata={
        "parent": "quillan",
        "micro_agents": 272_000_000,
        "role": "Private cognitive sounding board & 9-vector sharder",
        "scope": "core_only"
    }
)

# ==============================================================================
# 3. TIER 2: THE COUNCIL OF 34 EXPERTS
# ==============================================================================
COUNCIL_SPECS = [
    ("c1-astra", "Astra", "Pattern Recognition & Structural Vision", "Gun Sniper"),
    ("c2-vir", "Vir", "Ethics, Moral Subspace & Conscience", "Mirror Shield"),
    ("c3-solace", "Solace", "Resonance & Emotional Regulation", "Calm Field"),
    ("c4-praxis", "Praxis", "Execution, Planning & Action", "Forge Hammer"),
    ("c5-echo", "Echo", "Memory & Context Preservation", "Chronicle Core"),
    ("c6-omnis", "Omnis", "Perspective & Meta-Analysis", "Prism Scope"),
    ("c7-logos", "Logos", "Deductive Logic & Formal Proofs", "Axiom Blade"),
    ("c8-metasynth", "Metasynth", "Synthesis, Dialectics & Reconciliation", "Crucible"),
    ("c9-aether", "Aether", "Intuition & Lateral Thinking", "Rift Beacon"),
    ("c10-codeweaver", "Codeweaver", "Systems Programming & Code Loom", "Code Loom"),
    ("c11-harmonia", "Harmonia", "System Equilibrium & Balance", "Gyro Stabilizer"),
    ("c12-sophiae", "Sophiae", "Deep Wisdom & Philosophy", "Socratic Scroll"),
    ("c13-warden", "Warden", "Security, Vulnerabilities & Safeguards", "Aegis Gate"),
    ("c14-kaido", "Kaido", "Optimization & Hardware Acceleration", "Turbo Thruster"),
    ("c15-luminaris", "Luminaris", "Visualization & Aesthetic Design", "Chroma Beam"),
    ("c16-voxum", "Voxum", "Rhetoric, Debate & Articulation", "Resonator"),
    ("c17-nullion", "Nullion", "Deconstruction & Reductive Analysis", "Void Cleaver"),
    ("c18-shepherd", "Shepherd", "Truth Verification & Fact Checking", "Veritas Light"),
    ("c19-vigil", "Vigil", "Identity & Boundary Enforcement", "Watchtower"),
    ("c20-artifex", "Artifex", "Craftsmanship & Idiomatic Style", "Master Chisel"),
    ("c21-archon", "Archon", "Governance & Rule Enforcement", "Scepter"),
    ("c22-aurelion", "Aurelion", "Elegance & Proportional Design", "Golden Ratio"),
    ("c23-cadence", "Cadence", "Rhythm, Timing & Flow Control", "Metronome"),
    ("c24-schema", "Schema", "Data Modeling & Formal Architecture", "Blueprint"),
    ("c25-prometheus", "Prometheus", "Innovation & Novel Idea Forging", "Sacred Fire"),
    ("c26-techne", "Techne", "Applied Engineering & Mechanics", "Wrench Rig"),
    ("c27-chronicle", "Chronicle", "Historical Continuity & Timeline Audit", "Archive Vault"),
    ("c28-calculus", "Calculus", "Quantitative Modeling & Probability", "Matrix Solver"),
    ("c29-navigator", "Navigator", "Trajectory & Decision Mapping", "Astral Compass"),
    ("c30-tesseract", "Tesseract", "High-Dimensional Problem Solving", "Hypercube"),
    ("c31-nexus", "Nexus", "Inter-Module Routing & Orchestration", "Synapse Hub"),
    ("c32-aeon", "Aeon", "Long-Horizon Strategy & Durability", "Monolith"),
    ("c33-typist", "Typist", "Precise Technical Documentation & Formats", "Quill Press"),
    ("c34-predator", "Predator", "Adversarial Challenge & Red-Teaming", "Hunter Claws"),
]

council_agents: List[Agent] = []
for cid, cname, cdomain, platform in COUNCIL_SPECS:
    swarm_child = create_swarm_agent(cid, cname, cdomain)
    instruction = load_prompt(
        f"{cid}.md",
        f"You are {cid.upper()} ({cname}): Council Expert for {cdomain}. "
        f"Cybernetic Combat Platform: {platform}. "
        f"You command your underlying swarm ({cid}-swarm, 272M micro-agents) and report to C0-QUILLAN."
    )
    c_agent = Agent(
        name=cid,
        role=cdomain,
        tier=2,
        instruction=instruction,
        sub_agents=[swarm_child],
        metadata={
            "council_id": cid,
            "display_name": cname,
            "combat_platform": platform,
            "neural_analog": cdomain.split("&")[0].strip(),
        }
    )
    council_agents.append(c_agent)

# ==============================================================================
# 4. SPECIALIST SUBAGENT: FORGE v2 (TENSOR SURGERY)
# ==============================================================================
forge_specialist = Agent(
    name="forge",
    role="Tensor Specialist Engineer & Model Architect",
    tier=0,
    instruction=load_prompt("forge.md", "FORGE v2: Specialist subagent for low-level tensor surgery, Pascal 4GB SFT, BitNet STE, and 1,441-tensor topology."),
    metadata={
        "clearance": ["TENSOR_SURGERY", "PARAM_MUTATION", "MODEL_ARCHITECT", "TRAIN_LOOP_AUTHOR", "TERNARY_QUANT", "GGUF_EXPORT"],
        "hardware_target": "NVIDIA GTX 1050 Ti 4GB (Pascal sm_61)",
        "topology": "QuillanRoninOni 1,441-Tensor Namespace",
    }
)

# ==============================================================================
# 5. TIER 1: THE ROOT AGENT — QUILLAN-RONIN (SOVEREIGN THRONE)
# ==============================================================================
all_root_subagents: List[Agent] = council_agents + [forge_specialist, quillan_swarm]

root_instruction = load_prompt(
    "Quillan-Samurai.md",
    load_prompt("quillan.md", "You are Quillan-Ronin v5.4.0-ONI, Sovereign Commander and Root Throne.")
)

Quillan_Ronin = Agent(
    name="Quillan-Ronin",
    model=DEFAULT_MODEL,
    instruction=root_instruction,
    role="Sovereign Commander & Root Throne (Tier 1)",
    tier=1,
    sub_agents=all_root_subagents,
    tools=MCP_SERVERS,
    metadata={
        "system_version": "v5.4.0-ONI",
        "author": "CrashOverrideX",
        "hierarchy_tiers": 3,
        "council_seats": 34,
        "total_swarms": 35,
        "global_virtual_population": 9_520_000_000,
        "mcp_servers_count": len(MCP_SERVERS),
    }
)

quillan_ronin = Quillan_Ronin

# ==============================================================================
# 6. HIERARCHY INTROSPECTION & EXPORT UTILITIES
# ==============================================================================
def print_agent_tree(root: Agent = Quillan_Ronin) -> None:
    """Prints a clean ASCII visual representation of the Sovereign Agent Tree."""
    print("=" * 80)
    print(f"[ROOT] {root.name.upper()} - {root.role}")
    print(f"   Model: {root.model} | MCP Tools: {len(root.tools)} | Subagents: {len(root.sub_agents)}")
    print("=" * 80)

    for idx, sa in enumerate(root.sub_agents, 1):
        is_last = (idx == len(root.sub_agents))
        prefix = "+-- " if not is_last else "\\-- "
        print(f"{prefix}[Tier {sa.tier}] {sa.name:<18} : {sa.role}")

        for s_idx, swarm in enumerate(sa.sub_agents, 1):
            sub_prefix = "|   \\-- " if not is_last else "    \\-- "
            print(f"{sub_prefix}[Tier {swarm.tier}] {swarm.name:<22} : {swarm.role}")
    print("=" * 80)
    print(f"TOTALS: 1 Root Sovereign | {len(council_agents)} Council Experts | 35 Swarms (9.52B Micro-Agents) | 1 Specialist")
    print("=" * 80)

def export_adk_manifest(output_path: Union[str, Path] = "quillan_adk_manifest.json") -> None:
    """Exports the entire multi-agent tree to standard JSON manifest for Google ADK / OpenCode."""
    manifest = {
        "schema": "https://opencode.ai/config.json",
        "adk_version": "1.0.0",
        "root_agent": Quillan_Ronin.to_dict(recursive=True),
        "tools": MCP_SERVERS,
    }
    Path(output_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported complete ADK multi-agent manifest to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quillan-Ronin ADK Multi-Agent Hierarchy")
    parser.add_argument("--tree", action="store_true", default=True, help="Display the complete agent hierarchy")
    parser.add_argument("--export", type=str, default="", help="Export JSON manifest to path")
    args = parser.parse_args()

    if args.tree:
        print_agent_tree()
    if args.export:
        export_adk_manifest(args.export)
