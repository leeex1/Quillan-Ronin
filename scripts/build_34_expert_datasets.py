#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — 34-EXPERT DATASET PARTITIONER & ROUTER BUILDER (COMPREHENSIVE)
---------------------------------------------------------------------------------------
Scans all primary training corpora (Reasoning Gold, Science, Full Train, Code, PDFs)
and builds 34 rich domain-specific datasets in `training_data/experts_34/` + `router_training_dataset.jsonl`.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("build_34_experts")

REPO_ROOT = Path(r"C:\02_QUILLAN")
TRAIN_DATA_DIR = REPO_ROOT / "training_data"
OUTPUT_DIR = TRAIN_DATA_DIR / "experts_34"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERT_SPECS = [
    ("C0_ASTRA",       "Pattern Recognition & Vision",       ["vision", "anomaly", "fractal", "pattern", "geometry", "image", "visual", "cnn", "pixel"]),
    ("C1_VIR",         "Ethical Guardian",                   ["ethics", "safety", "harm_reduction", "zero_drift", "alignment", "moral", "deontology", "utilitarian"]),
    ("C2_SOLACE",      "Emotional Intelligence",             ["empathy", "sentiment", "affect", "feeling", "psychology", "support", "compassion", "mental"]),
    ("C3_PRAXIS",      "Strategic Planning",                 ["strategy", "planning", "goals", "milestone", "roadmap", "execution", "tactics", "objective"]),
    ("C4_ECHO",        "Memory Continuity",                  ["history", "recall", "context", "lancedb", "vector", "embedding", "retrieval", "rag", "memory"]),
    ("C5_OMNIS",       "Knowledge Synthesis",                ["synthesis", "integration", "holistic", "interdisciplinary", "overview", "encyclopedic", "summary"]),
    ("C6_LOGOS",       "Logical Consistency",                ["logic", "deduction", "validity", "syllogism", "proof", "theorem", "premise", "inference", "boolean"]),
    ("C7_METASYNTH",   "Creative Fusion",                    ["creativity", "novelty", "ideation", "metaphor", "analogy", "concept", "imagination", "story"]),
    ("C8_AETHER",      "Semantic Connection",                ["semantics", "language", "grammar", "linguistics", "meaning", "syntax", "etymology", "pragmatics"]),
    ("C9_CODEWEAVER",  "Technical Implementation",           ["code", "python", "rust", "function", "algorithm", "class", "debug", "refactor", "bug", "def ", "return "]),
    ("C10_HARMONIA",   "Balance & Equilibrium",              ["balance", "mediation", "consensus", "harmony", "resolution", "negotiation", "peace"]),
    ("C11_SOPHIAE",    "Wisdom & Foresight",                 ["wisdom", "future", "philosophy", "epistemology", "long_term", "existential", "teleology"]),
    ("C12_WARDEN",     "Safety & Security",                  ["security", "threat", "risk", "sandboxing", "cwe", "cve", "injection", "vulnerability", "xss", "csrf", "exploit"]),
    ("C13_KAIDO",      "Efficiency Optimization",            ["speed", "efficiency", "latency", "hardware", "cpu", "gpu", "cuda", "profiling", "cache", "bitnet", "quantization"]),
    ("C14_LUMINARIS",  "Clarity & Presentation",             ["clarity", "visualization", "polish", "formatting", "markdown", "presentation", "table", "diagram"]),
    ("C15_VOXUM",      "Articulation & Expression",          ["rhetoric", "tone", "persuasion", "articulation", "speech", "dialogue", "conversation", "oratory"]),
    ("C16_NULLION",    "Paradox Resolution",                 ["paradox", "dialectic", "ambiguity", "contradiction", "uncertainty", "antinomy", "hegel"]),
    ("C17_SHEPHERD",   "Truth Verification",                 ["truth", "citation", "fact", "verification", "evidence", "grounding", "empirical", "validate"]),
    ("C18_VIGIL",      "Identity Integrity",                 ["identity", "consistency", "anti_drift", "quillan", "ronin", "sovereign", "self", "who are you", "assistant"]),
    ("C19_ARTIFEX",    "Tool Integration",                   ["tools", "api", "external", "host_os", "bash", "powershell", "rest", "json", "http", "endpoint"]),
    ("C20_ARCHON",     "Deep Research",                      ["research", "mining", "analysis", "paper", "literature", "study", "arxiv", "abstract", "methodology"]),
    ("C21_AURELION",   "Aesthetic Design",                   ["design", "art", "style", "ui", "ux", "visual", "typography", "color", "layout", "aesthetic"]),
    ("C22_CADENCE",    "Rhythmic Innovation",                ["music", "rhythm", "audio", "frequency", "wave", "sound", "signal", "fourier", "dsp"]),
    ("C23_SCHEMA",     "Structural Template",                ["structure", "format", "schema", "database", "sql", "table", "relational", "orm", "nosql", "index"]),
    ("C24_PROMETHEUS", "Scientific Theory",                  ["science", "hypothesis", "physics", "chemistry", "biology", "photosynthesis", "reaction", "molecule", "atom"]),
    ("C25_TECHNE",     "Engineering Mastery",                ["architecture", "systems", "build", "devops", "linux", "sigterm", "sigkill", "docker", "server", "process"]),
    ("C26_CHRONICLE",  "Narrative Synthesis",                ["story", "narrative", "lore", "chronology", "changelog", "history", "timeline", "evolution"]),
    ("C27_CALCULUS",   "Quantitative Reasoning",             ["math", "statistics", "calc", "derivative", "integral", "matrix", "probability", "fibonacci", "linear algebra"]),
    ("C28_NAVIGATOR",  "Ecosystem Orchestration",            ["platform", "integration", "flow", "orchestration", "pipeline", "service", "workflow", "event-driven"]),
    ("C29_TESSERACT",  "Real-Time Intelligence",             ["real_time", "stream", "data", "telemetry", "time_series", "event", "kafka", "sensor"]),
    ("C30_NEXUS",      "Meta-Coordination",                  ["coordination", "lee_mach_6", "governance", "thermodynamic", "kuramoto", "sync", "swarm", "entropy"]),
    ("C31_AEON",       "Interactive Simulation",             ["simulation", "game", "world", "agent", "environment", "modeling", "monte carlo", "dynamic"]),
    ("C32_TYPIST",     "Prompt Internal Optimization",       ["grammar", "writing", "spelling", "prompting", "token", "bpe", "vocab", "text", "punctuation"]),
    ("C33_PREDATOR",   "PredatoryMath",                      ["competitive", "predatory", "stacking", "weakness", "exploit", "hunting", "adversarial", "game theory"]),
]

def score_expert_match(text: str, keywords: List[str]) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(kw) for kw in keywords)

def main():
    LOGGER.info("Starting comprehensive 34 Council Expert dataset construction...")
    
    expert_pools: Dict[int, List[Dict[str, str]]] = {i: [] for i in range(len(EXPERT_SPECS))}
    router_samples: List[Dict[str, Any]] = []

    source_files = [
        TRAIN_DATA_DIR / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl",
        TRAIN_DATA_DIR / "Quillan_Hyper_Tune_Gold_Dataset.jsonl",
        TRAIN_DATA_DIR / "Quillan_General_Knowledge_Dataset.jsonl",
        TRAIN_DATA_DIR / "Quillan_Refined_Thought_Corpus.jsonl",
        TRAIN_DATA_DIR / "quillan_science_absolute.jsonl",
        TRAIN_DATA_DIR / "code_train.jsonl",
        TRAIN_DATA_DIR / "instruct_train.jsonl",
        TRAIN_DATA_DIR / "train.jsonl",
    ]

    total_scanned = 0
    max_per_expert = 2000  # High-density balance

    for src in source_files:
        if not src.exists():
            continue

        LOGGER.info("Scanning corpus: %s", src.name)
        with open(src, "r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                q = data.get("question") or data.get("prompt") or data.get("instruction") or data.get("input") or ""
                a = data.get("response") or data.get("answer") or data.get("output") or data.get("text") or ""

                if not q and not a:
                    continue
                if not q and len(a) > 50:
                    q = a[:100] + "..."
                if not a and len(q) > 50:
                    a = q

                if len(q) < 5 or len(a) < 10:
                    continue

                total_scanned += 1
                combined_text = f"{q}\n{a}"

                scores = []
                for exp_idx, (exp_code, exp_name, kws) in enumerate(EXPERT_SPECS):
                    s = score_expert_match(combined_text, kws)
                    scores.append((s, exp_idx))

                scores.sort(key=lambda x: x[0], reverse=True)
                top_score, top_exp_idx = scores[0]

                if top_score > 0 and len(expert_pools[top_exp_idx]) < max_per_expert:
                    expert_pools[top_exp_idx].append({"question": q, "response": a})
                    active_experts = [idx for sc, idx in scores[:4] if sc > 0]
                    if not active_experts:
                        active_experts = [top_exp_idx]
                    router_samples.append({
                        "prompt": q[:256],
                        "target_experts": active_experts,
                        "primary_expert": top_exp_idx,
                    })

                if total_scanned % 10000 == 0:
                    LOGGER.info("Scanned %d pairs...", total_scanned)

    LOGGER.info("Finished scanning %d total QA pairs.", total_scanned)

    # Save 34 datasets
    for idx, (exp_code, exp_name, _) in enumerate(EXPERT_SPECS):
        exp_file = OUTPUT_DIR / f"{idx:02d}_{exp_code}.jsonl"
        items = expert_pools[idx]
        with open(exp_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        LOGGER.info("Expert [%02d] %-15s: %5d QA pairs -> %s", idx, exp_code, len(items), exp_file.name)

    # Save Router dataset
    router_file = TRAIN_DATA_DIR / "router_training_dataset.jsonl"
    with open(router_file, "w", encoding="utf-8") as f:
        for item in router_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    LOGGER.info("Router dataset: %d samples -> %s", len(router_samples), router_file.name)

if __name__ == "__main__":
    main()
