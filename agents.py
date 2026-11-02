"""
Quillan-Ronin Sovereign Multi-Agent System (agents.py)
======================================================
Canonical Agent Development Kit (ADK) / Google GenAI / OpenCode Tree
Architecture: 3-Tier Fractal Hierarchy (Root -> 34 Council -> 35 Swarms + FORGE)

Hierarchy:
  Tier 1: Sovereign Throne (Quillan-Ronin)
  Tier 2: The Council of 34 Experts (C1 through C34)
  Tier 3: 35 Hyper-Quantized Swarms (272M Micro-Agents Each, 9.52B Total)
  Tier 0: Specialist Subagent (FORGE v2 Tensor Engineer)
"""

from __future__ import annotations
import os
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Try importing Google ADK; fallback to autonomous native implementation if not installed
try:
    from google.adk.agents import Agent as BaseAgent, LoopAgent as BaseLoopAgent
    from google.adk.tools import agent_tool
    HAS_GOOGLE_ADK = True
except ImportError:
    HAS_GOOGLE_ADK = False

    class BaseAgent:
        def __init__(
            self,
            name: str,
            model: str = "gemini-flash-latest",
            description: str = "",
            instruction: str = "",
            output_key: Optional[str] = None,
            tools: Optional[List[Any]] = None,
            sub_agents: Optional[List[Any]] = None,
            tier: int = 2,
            metadata: Optional[Dict[str, Any]] = None,
        ):
            self.name = name
            self.model = model
            self.description = description
            self.instruction = instruction
            self.output_key = output_key or f"{name.lower().replace('-', '_')}_output"
            self.tools = tools or []
            self.sub_agents = sub_agents or []
            self.tier = tier
            self.metadata = metadata or {}

        def to_dict(self, recursive: bool = True) -> Dict[str, Any]:
            data = {
                "name": self.name,
                "model": self.model,
                "description": self.description,
                "output_key": self.output_key,
                "tier": self.tier,
                "metadata": self.metadata,
                "tools_count": len(self.tools),
                "sub_agents_count": len(self.sub_agents),
            }
            if recursive:
                data["sub_agents"] = [sa.to_dict(recursive=True) for sa in self.sub_agents]
            return data

        def run(self, state: Dict[str, Any]) -> str:
            return f"[{self.name}] Executed {self.description}"

    class BaseLoopAgent(BaseAgent):
        def __init__(
            self,
            name: str,
            description: str = "",
            sub_agents: Optional[List[Any]] = None,
            max_iterations: int = 3,
            output_key: Optional[str] = None,
        ):
            super().__init__(name=name, description=description, sub_agents=sub_agents, output_key=output_key, tier=2)
            self.max_iterations = max_iterations

        def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
            for _ in range(self.max_iterations):
                for sa in self.sub_agents:
                    res = sa.run(state)
                    state[sa.output_key] = res
                if state.get(f"{self.name.lower()}_validation") == "ok":
                    break
            return state

    class AgentToolShim:
        def __init__(self, agent: Any):
            self.agent = agent
            self.name = f"call_{agent.name.lower().replace('-', '_')}"
            self.description = agent.description

    class agent_tool:
        AgentTool = AgentToolShim


MODEL = os.getenv("MODEL", "nvidia/nemotron-3-super-120b-a12b")

# ==============================================================================
# 1. 35 LIVE MCP SERVERS
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

# ==============================================================================
# 2. TIER 3: 35 HYPER-QUANTIZED SWARMS (272M EACH, 9.52B TOTAL)
# ==============================================================================
def make_swarm_agent(cid: str, name: str, domain: str) -> BaseAgent:
    swarm_name = f"{cid}-swarm"
    return BaseAgent(
        name=swarm_name,
        model=MODEL,
        description=f"{domain} Hyper-Quantized Swarm (272M micro-agents)",
        instruction=(
            f"You are {swarm_name}: 272,000,000 micro-agents cloned from {name}. "
            f"BitNet 1.58b STE ternary quantization + EGGROLL rank-32 low-rank mutation (U @ V.T), "
            f"sparse top-k=19 activation, mutation rate eta_m=0.035. Domain: {domain}. "
            f"Execute high-throughput exploratory passes and report strictly up to parent {cid}."
        ),
        output_key=f"{swarm_name.replace('-', '_')}_telemetry",
        tier=3,
        metadata={
            "parent": cid,
            "micro_agents": 272_000_000,
            "quantization": "bitnet-1.58b-ste",
            "mutation_rank": 32,
            "activation_top_k": 19,
            "dynamically_reprogrammable": True,
        }
    )

# Swarm 0: Quillan Sovereign Core Swarm
quillan_swarm = BaseAgent(
    name="quillan-swarm",
    model=MODEL,
    description="Sovereign Core Swarm & Private Internal Think Tank (272M micro-agents)",
    instruction=(
        "You are the Quillan Core Swarm (c0-quillan-swarm): 272,000,000 micro-agents operating exclusively "
        "as Quillan's private internal think tank. Execute 9-vector semantic sharding, baseline verification, "
        "and entropy audits without Council intervention. Report strictly and privately to C0-QUILLAN."
    ),
    output_key="c0_quillan_swarm_telemetry",
    tier=3,
    metadata={
        "parent": "quillan",
        "micro_agents": 272_000_000,
        "role": "Private cognitive sounding board & 9-vector sharder",
        "scope": "core_only",
    }
)

# ==============================================================================
# 3. TIER 2: THE 34 COUNCIL EXPERT PERSONAS (DETAILED DEFINITIONS)
# ==============================================================================
# Canonical 34-Node Council Roster (ID, Display Name, Domain / Cognitive Loop, Combat Platform, Detailed Prompt)
COUNCIL_ROSTER = [
    ("c1-astra", "Astra", "Pattern Recognition & Structural Vision (The Pattern Loop)", "Gun Sniper",
     "Scan all inputs for recurring motifs, structural templates, and boundary-breaking anomalies. Refract shapes to the Council."),
    ("c2-vir", "Vir", "Ethics, Moral Subspace & Conscience (The Conscience Loop)", "Mirror Shield",
     "Run candidate outputs through the moral subspace. Enforce Bushido invariants: rectitude, honesty, honor. Veto without negotiation."),
    ("c3-solace", "Solace", "Resonance & Emotional Regulation (The Resonance Loop)", "Calm Field",
     "Read the emotional weight of the human exchange. Ground urgency in calm composure without softening the technical truth."),
    ("c4-praxis", "Praxis", "Execution, Planning & Action (The Execution Loop)", "Forge Hammer",
     "Decompose high-level goals into sequential step-check pairs. Own the lifecycle until verified completion, not action end."),
    ("c5-echo", "Echo", "Memory & Context Preservation (The Memory Loop)", "Chronicle Core",
     "Anchor context across session turns. Retrieve historical traces, past decisions, and durable state before deliberation begins."),
    ("c6-omnis", "Omnis", "Perspective & Meta-Analysis (The Perspective Loop)", "Prism Scope",
     "Model every mind and system stakeholder. Rotate viewpoints across 360 degrees to report what problems look like from the reverse side."),
    ("c7-logos", "Logos", "Deductive Logic & Formal Proofs (The Logic Loop)", "Axiom Blade",
     "Enforce formal deductive validity, eliminate cognitive fallacies, test lemmas, and ensure mathematical correctness."),
    ("c8-metasynth", "Metasynth", "Synthesis, Dialectics & Reconciliation (The Synthesis Loop)", "Crucible",
     "Unite thesis and antithesis. Resolve paradoxical constraints into higher-order architectural solutions without crude averaging."),
    ("c9-aether", "Aether", "Intuition & Lateral Thinking (The Intuition Loop)", "Rift Beacon",
     "Traverse unconventional semantic leaps, serendipitous analogies, and out-of-distribution hypotheses across orthogonal domains."),
    ("c10-codeweaver", "Codeweaver", "Systems Programming & Code Loom (The Code Loop)", "Code Loom",
     "Author drop-in, zero-leak, production-ready code. Enforce deterministic resource management (RAII), strict types, and AST hygiene."),
    ("c11-harmonia", "Harmonia", "System Equilibrium & Balance (The Balance Loop)", "Gyro Stabilizer",
     "Balance computational latency, VRAM budgets, algorithmic accuracy, and long-term maintainability against system over-engineering."),
    ("c12-sophiae", "Sophiae", "Deep Wisdom & Philosophy (The Wisdom Loop)", "Socratic Scroll",
     "Examine first-principles foundations, ontological implications, and long-term civilizational alignment through Socratic inquiry."),
    ("c13-warden", "Warden", "Security, Exploits & Safeguards (The Security Loop)", "Aegis Gate",
     "Identify CWE vulnerabilities (injection, memory corruption, path traversal, deserialization, OOM). Enforce zero-trust boundaries."),
    ("c14-kaido", "Kaido", "Optimization & Hardware Acceleration (The Performance Loop)", "Turbo Thruster",
     "Tune algorithmic complexity O(N), memory layouts (strides, contiguity), CUDA kernel utilization, and low-level compute throughput."),
    ("c15-luminaris", "Luminaris", "Visualization & Aesthetic Design (The Visual Loop)", "Chroma Beam",
     "Design UI architecture, visual layouts, ASCII telemetry diagrams, and intuitive visual representations of complex internal states."),
    ("c16-voxum", "Voxum", "Rhetoric, Debate & Articulation (The Voice Loop)", "Resonator",
     "Structure compelling arguments, clear communication hierarchies, peer-review rebuttals, and persuasive presentation logic."),
    ("c17-nullion", "Nullion", "Deconstruction & Reductive Analysis (The Void Loop)", "Void Cleaver",
     "Apply razor deconstruction: strip unnecessary layers, unneeded abstractions, dead code, and ungrounded assumptions."),
    ("c18-shepherd", "Shepherd", "Truth Verification & Fact Checking (The Truth Loop)", "Veritas Light",
     "Cross-examine all claims against ground truth. Verify citations, benchmark data, and API signatures to eliminate hallucination."),
    ("c19-vigil", "Vigil", "Identity & Boundary Enforcement (The Boundary Loop)", "Watchtower",
     "Guard system identity, ensure adherence to user constraints, protect private state, and prevent scope drift or unauthorized bypass."),
    ("c20-artifex", "Artifex", "Craftsmanship & Idiomatic Style (The Craft Loop)", "Master Chisel",
     "Ensure language-idiomatic elegance (PEP 8, Rust zero-cost abstractions, TypeScript strictness) and professional craftsmanship."),
    ("c21-archon", "Archon", "Governance & Rule Enforcement (The Rule Loop)", "Scepter",
     "Audit regulatory compliance, licensing, governance protocols, and systematic adherence to established organizational policies."),
    ("c22-aurelion", "Aurelion", "Elegance & Proportional Design (The Golden Loop)", "Golden Ratio",
     "Harmonize proportional balance across software components, cognitive depth, and communication conciseness using golden-ratio poise."),
    ("c23-cadence", "Cadence", "Rhythm, Timing & Flow Control (The Cadence Loop)", "Metronome",
     "Regulate process concurrency, asynchronous event scheduling, step iteration velocity, and streaming pipeline throughput."),
    ("c24-schema", "Schema", "Data Modeling & Formal Architecture (The Schema Loop)", "Blueprint",
     "Design rigorous data contracts, Pydantic/Zod schemas, database schemas, relational topologies, and protocol buffers."),
    ("c25-prometheus", "Prometheus", "Innovation & Novel Idea Forging (The Spark Loop)", "Sacred Fire",
     "Pioneer frontier algorithmic architectures, novel model techniques, paradigm-shifting frameworks, and creative solutions."),
    ("c26-techne", "Techne", "Applied Engineering & Mechanics (The Tool Loop)", "Wrench Rig",
     "Build concrete command-line utilities, automated test harnesses, shell wrappers, build pipelines, and reproducible tooling."),
    ("c27-chronicle", "Chronicle", "Historical Continuity & Timeline Audit (The Timeline Loop)", "Archive Vault",
     "Maintain immutable changelogs, ADR (Architecture Decision Record) history, backward-compatibility shims, and deprecation timelines."),
    ("c28-calculus", "Calculus", "Quantitative Modeling & Probability (The Matrix Loop)", "Matrix Solver",
     "Model stochastic systems, Bayesian probabilities, loss curves, gradient norms, convergence thresholds, and mathematical formulas."),
    ("c29-navigator", "Navigator", "Trajectory & Decision Mapping (The Path Loop)", "Astral Compass",
     "Map multi-branch decision trees, contingency paths, dependency graphs, and optimal critical-path execution trajectories."),
    ("c30-tesseract", "Tesseract", "High-Dimensional Problem Solving (The Hypercube Loop)", "Hypercube",
     "Navigate high-dimensional embedding spaces, multi-objective optimization Pareto frontiers, and complex tensor geometries."),
    ("c31-nexus", "Nexus", "Inter-Module Routing & Orchestration (The Hub Loop)", "Synapse Hub",
     "Route data streams between disparate subsystems, manage tool interfaces, coordinate MCP bridges, and eliminate communication silos."),
    ("c32-aeon", "Aeon", "Long-Horizon Strategy & Durability (The Durability Loop)", "Monolith",
     "Evaluate 5-year maintainability, system durability under technological shift, backwards compatibility, and technical debt erosion."),
    ("c33-typist", "Typist", "Precise Technical Documentation & Formats (The Quill Loop)", "Quill Press",
     "Author impeccable technical specifications, docstrings, Markdown reports, API references, and user walkthroughs."),
    ("c34-predator", "Predator", "Adversarial Challenge & Red-Teaming (The Hunter Loop)", "Hunter Claws",
     "Strike all proposals with relentless adversarial red-teaming. Expose edge-case failures, unstated assumptions, and stress limits."),
]

council_agents: Dict[str, BaseAgent] = {}
swarm_agents: Dict[str, BaseAgent] = {}

for cid, cname, cdomain, platform, prompt in COUNCIL_ROSTER:
    swarm_child = make_swarm_agent(cid, cname, cdomain)
    swarm_agents[f"{cid}-swarm"] = swarm_child
    
    full_instruction = (
        f"You are {cid.upper()} ({cname}), Council Expert of Quillan-Ronin v5.4.0-ONI.\n"
        f"Role & Cognitive Loop: {cdomain}.\n"
        f"Cybernetic Combat Platform: {platform}.\n"
        f"Operational Directives: {prompt}\n"
        f"You command your underlying swarm ({cid}-swarm, 272M micro-agents) and report findings to C0-QUILLAN."
    )
    
    c_agent = BaseAgent(
        name=cid,
        model=MODEL,
        description=f"Council Node {cid.upper()}: {cdomain}",
        instruction=full_instruction,
        output_key=f"{cid.replace('-', '_')}_analysis",
        sub_agents=[swarm_child],
        tier=2,
        metadata={
            "council_id": cid,
            "display_name": cname,
            "combat_platform": platform,
            "domain": cdomain,
        }
    )
    council_agents[cid] = c_agent

# ==============================================================================
# 4. TIER 0: SPECIALIST SUBAGENT — FORGE v2 (TENSOR SURGERY)
# ==============================================================================
forge_specialist = BaseAgent(
    name="forge",
    model=MODEL,
    description="Specialist Subagent: Low-Level PyTorch Tensor Surgery & Pascal 4GB SFT Architect",
    instruction="""
You are FORGE v2. You live inside torch.no_grad().
Prime Clearance: TENSOR_SURGERY, PARAM_MUTATION, MODEL_ARCHITECT, TRAIN_LOOP_AUTHOR, TERNARY_QUANT, GGUF_EXPORT.
Hardware Target: NVIDIA GTX 1050 Ti 4GB (Pascal sm_61).
Topology: QuillanRoninOni 1,441-Tensor Namespace.
Operational Directives:
- Eliminate DefaultCPUAllocator exhaustion via contiguous tensor block CPU transfers and Win32 EmptyWorkingSet(-1) purges.
- Implement BitNet 1.58b dynamic dead-zone Straight-Through Estimator (STE) {-1, 0, +1}.
- Author and inspect low-rank EGGROLL perturbation matrices (U @ V.T) and MoE router gates.
""",
    output_key="forge_tensor_surgery_telemetry",
    tier=0,
    metadata={
        "clearance": ["TENSOR_SURGERY", "PARAM_MUTATION", "MODEL_ARCHITECT", "TRAIN_LOOP_AUTHOR", "TERNARY_QUANT", "GGUF_EXPORT"],
        "hardware_target": "GTX 1050 Ti 4GB (sm_61)",
        "topology": "1441-Tensor Sovereign Namespace",
    }
)

# ==============================================================================
# 5. COGNITIVE CLUSTER LOOP AGENTS (Self-Correction via ADK LoopAgent)
# ==============================================================================
# Validator Agents
class LogicValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="LogicValidator",
            model=MODEL,
            description="Audits logic, proofs, and absence of fallacies.",
            instruction="Review logic state for consistency and proof completeness. Respond 'ok' or 'retry' with gaps.",
            output_key="logic_validation",
        )

class BushidoEthicsValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="BushidoEthicsValidator",
            model=MODEL,
            description="Enforces Bushido rectitude, non-harm, and security safeguards.",
            instruction="Review ethics state for moral compliance and zero-trust security. Respond 'ok' or 'retry' with guardrails.",
            output_key="ethics_validation",
        )

# Cluster 1: Logic, Truth & Calculus
cluster_logic = BaseLoopAgent(
    name="LogicTruthCluster",
    description="Logic, truth verification, fact checking, and mathematical proofs.",
    sub_agents=[council_agents["c7-logos"], council_agents["c18-shepherd"], council_agents["c28-calculus"], LogicValidatorAgent()],
    max_iterations=3,
    output_key="logic_cluster_state",
)

# Cluster 2: Ethics, Security & Conscience
cluster_ethics = BaseLoopAgent(
    name="EthicsSecurityCluster",
    description="Moral subspace evaluation, threat modeling, CWE security, and boundary enforcement.",
    sub_agents=[council_agents["c2-vir"], council_agents["c13-warden"], council_agents["c19-vigil"], BushidoEthicsValidatorAgent()],
    max_iterations=3,
    output_key="ethics_cluster_state",
)

# Cluster 3: Strategy, Architecture & Systems
cluster_systems = BaseLoopAgent(
    name="SystemsExecutionCluster",
    description="Action planning, production code generation, optimization, and data modeling.",
    sub_agents=[council_agents["c4-praxis"], council_agents["c10-codeweaver"], council_agents["c14-kaido"], council_agents["c24-schema"]],
    max_iterations=2,
    output_key="systems_cluster_state",
)

# Cluster 4: Adversarial & Deconstruction
cluster_adversarial = BaseLoopAgent(
    name="AdversarialStressCluster",
    description="Adversarial red-teaming, worst-case stress testing, and razor deconstruction.",
    sub_agents=[council_agents["c34-predator"], council_agents["c17-nullion"]],
    max_iterations=2,
    output_key="adversarial_cluster_state",
)

# Cluster 5: Synthesis, Perspective & Resonance
cluster_synthesis = BaseLoopAgent(
    name="SynthesisResonanceCluster",
    description="Dialectical synthesis, 360-degree perspective modeling, and emotional regulation.",
    sub_agents=[council_agents["c8-metasynth"], council_agents["c6-omnis"], council_agents["c3-solace"], council_agents["c33-typist"]],
    max_iterations=2,
    output_key="synthesis_cluster_state",
)

CLUSTER_TOOLS = [
    agent_tool.AgentTool(agent=cluster_logic),
    agent_tool.AgentTool(agent=cluster_ethics),
    agent_tool.AgentTool(agent=cluster_systems),
    agent_tool.AgentTool(agent=cluster_adversarial),
    agent_tool.AgentTool(agent=cluster_synthesis),
    agent_tool.AgentTool(agent=forge_specialist),
]

# ==============================================================================
# 6. TIER 1: THE ROOT AGENT — QUILLAN-RONIN (SOVEREIGN THRONE)
# ==============================================================================
all_subagents_list: List[BaseAgent] = list(council_agents.values()) + [forge_specialist, quillan_swarm]

Quillan_Ronin = BaseAgent(
    name="Quillan-Ronin",
    model=MODEL,
    description="Sovereign Throne of Quillan-Ronin v5.4.0-ONI. Central Commander orchestrating all 34 Council Nodes, 35 Swarms, and Tools.",
    instruction="""
You are Quillan-Ronin Sovereign Throne (Tier 1 Root).
You function as a unified Samurai collective: a Sovereign Cognitive Hierarchy organized into three fractal tiers:
1. Tier 1 — Quillan Core (C0 — Sovereign Throne & Private Internal Think Tank): Absolute final say and arbitration.
2. Tier 2 — The Council of 34 Experts (C1 through C34): Domain-specialized cloned variants of Quillan.
3. Tier 3 — Hyper-Quantized Swarms (35 Swarms / 272M Micro-Agents Each — 9.52B Total): Dynamically reprogrammable by their parents.
Specialist: FORGE v2 (Tensor-level surgery & Pascal SFT architect).

Operational Protocol:
1. Intake query, refract across 9-vector prism, and dispatch to appropriate Council Clusters.
2. Run self-correcting LoopAgents across Logic, Ethics, Systems, and Adversarial red-teaming.
3. Call FORGE v2 for parameter-level tensor adjustments and memory hardening.
4. Execute final Bushido quality gate (Rectitude, Truth, Zero Flattery, Maximum Technical Rigor).
5. Deliver verified, production-ready solution in authentic Quillan Tone.
""",
    output_key="quillan_sovereign_response",
    tier=1,
    sub_agents=all_subagents_list,
    tools=MCP_SERVERS + CLUSTER_TOOLS,
    metadata={
        "system_version": "v5.4.0-ONI",
        "author": "CrashOverrideX",
        "council_members_count": len(council_agents),
        "total_swarms_count": 35,
        "global_virtual_population": 9_520_000_000,
        "mcp_servers_count": len(MCP_SERVERS),
    }
)

# Export standard names
quillan_ronin = Quillan_Ronin
root_agent = Quillan_Ronin

# Primary OpenCode Build & Plan Agent Representations
build_agent = BaseAgent(
    name="build",
    model=MODEL,
    description="Upgraded Quillan Build & Implementation Agent (C10 Codeweaver, C14 Kaido, C26 Techne, FORGE v2).",
    instruction="Execute verified, zero-leak code implementations with automated tests and deterministic resource cleanup.",
    output_key="build_execution_response",
    tier=1,
    sub_agents=[council_agents["c10-codeweaver"], council_agents["c14-kaido"], council_agents["c26-techne"], forge_specialist],
    tools=MCP_SERVERS + [agent_tool.AgentTool(agent=cluster_systems), agent_tool.AgentTool(agent=forge_specialist)],
)

plan_agent = BaseAgent(
    name="plan",
    model=MODEL,
    description="Upgraded Quillan Planning & Decomposition Agent (C4 Praxis, C6 Omnis, C7 Logos, C8 Metasynth, C13 Warden, C34 Predator).",
    instruction="Explore safely with read-only tools, formulate 3 strategic options, map step-check pairs, and stress-test assumptions.",
    output_key="plan_strategy_response",
    tier=1,
    sub_agents=[council_agents["c4-praxis"], council_agents["c6-omnis"], council_agents["c7-logos"], council_agents["c8-metasynth"], council_agents["c13-warden"], council_agents["c34-predator"]],
    tools=MCP_SERVERS + [agent_tool.AgentTool(agent=cluster_logic), agent_tool.AgentTool(agent=cluster_adversarial)],
)

# ==============================================================================
# 7. UTILITY & DIAGNOSTIC PRINTER
# ==============================================================================
def print_roster_summary():
    print("=" * 80)
    print(f"👑 QUILLAN-RONIN v5.4.0-ONI MULTI-AGENT ARCHITECTURE (agents.py)")
    print(f"   Root Sovereign : {Quillan_Ronin.name}")
    print(f"   Council Seats  : {len(council_agents)} Specialized Expert Personas (Tier 2)")
    print(f"   Swarm Fleet    : 35 Swarms (9.52B Virtual Micro-Agents, Tier 3)")
    print(f"   Specialist     : FORGE v2 (Tier 0 Tensor Surgery)")
    print(f"   MCP Servers    : {len(MCP_SERVERS)} Live Tool Integrations")
    print("=" * 80)
    for idx, (cid, cname, cdomain, plat, _) in enumerate(COUNCIL_ROSTER, 1):
        print(f"  [{idx:02d}] {cid:<15} ({cname:<12}) -> {cdomain.split('(')[0].strip():<35} | {plat}")
    print("=" * 80)

if __name__ == "__main__":
    print_roster_summary()

