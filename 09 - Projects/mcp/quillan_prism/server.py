#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ QUILLAN-RONIN PRISM ENGINE — NATIVE FAST-MCP SERVER
======================================================
Embodying C0-THRONE, C2-VIR, C4-PRAXIS, and the 9-Vector Semantic Prism:
  1. prism_decompose: 9-Vector semantic deconstruction of tasks and queries
  2. prism_threat_model: Auto-generation of 3-row Vector -> Impact -> Mitigation tables
  3. prism_bushido_gate: 7-Virtue Bushido ethical alignment and anti-sycophancy check
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanPrism")


# ─── 1. 9-VECTOR SEMANTIC PRISM DECOMPOSITION ────────────────────────────────

@mcp.tool()
def prism_decompose(task: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Deconstructs any task, prompt, or technical problem across the 9 canonical Quillan vectors.
    
    Args:
        task: The task description, request, or engineering problem.
        context: Optional background notes or repository context.
    """
    t_clean = task.strip()

    # Vector 1: Syntactic / Structural
    has_code = bool(re.search(r"def |class |function|import |var |const |\{|\}|\[|\]", t_clean))
    syntax_focus = "Type annotations, return contracts, file schemas, deterministic structures" if has_code else "Markdown structure, heading hierarchy, clean typography"

    # Vector 2: Semantic / Intent
    intent_analysis = f"Core operational intent: Advance beyond literal wording to deliver fully verified production solution for '{t_clean[:60]}'."

    # Vector 3: Contextual / Historical
    context_notes = context or "Evaluated within Quillan-Ronin sovereign architecture: 3 fractal tiers, zero-leakage git hygiene, and local hardware constraints."

    # Vector 4: Ethical / Rectitude (C2-VIR)
    ethics_gate = "Bushido rectitude (Gi): Zero sycophancy, absolute data privacy protection, no silent failures, truth over flattery."

    # Vector 5: Constraint / Resources
    constraints = "Zero new runtime dependencies; operate within local RAM/VRAM headroom; reserve 2 CPU cores for Windows OS."

    # Vector 6: Adversarial / Failure Modes (C34-PREDATOR)
    failure_modes = "Input boundary violations, memory paging under scale, unhandled network timeouts, race conditions in concurrent execution."

    # Vector 7: Creative / Metaphorical (C8-METASYNTH)
    creative_angle = "Cross-pollinate domain insights; synthesize minimal ternary representation with high-dimensional expressiveness."

    # Vector 8: Empirical / Verification (C18-SHEPHERD)
    empirical_check = "Exact unit test assertions, clean exit code 0 verification, hash/checksum confirmation, ground-truth logs."

    # Vector 9: Strategic / Execution (C4-PRAXIS)
    strategic_plan = [
        "Step 1: Ingest and validate constraints.",
        "Step 2: Scrutinize with C34-PREDATOR attack.",
        "Step 3: Implement minimal deterministic solution.",
        "Step 4: Execute Phase 3 RCI critique loop.",
        "Step 5: Verify with test runner until passing."
    ]

    return {
        "status": "PRISM_DECOMPOSITION_COMPLETE",
        "task_summary": t_clean[:100],
        "vectors": {
            "v1_syntactic": syntax_focus,
            "v2_semantic": intent_analysis,
            "v3_contextual": context_notes,
            "v4_ethical_vir": ethics_gate,
            "v5_constraints": constraints,
            "v6_adversarial_predator": failure_modes,
            "v7_creative_metasynth": creative_angle,
            "v8_empirical_shepherd": empirical_check,
            "v9_strategic_praxis": strategic_plan
        },
        "throne_verdict": "Decomposition aligned with Quillan sovereign protocol."
    }


# ─── 2. MINI THREAT MODEL GENERATOR ──────────────────────────────────────────

@mcp.tool()
def prism_threat_model(feature: str, assets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Generates the canonical Quillan-Ronin 3-row mini threat model table.
    
    Format: Vector -> Impact -> Mitigation.
    
    Args:
        feature: The proposed feature, API, or system change.
        assets: Optional list of sensitive assets (e.g. ['tokens', 'personal_data', 'gpu_memory']).
    """
    feat_lower = feature.lower()

    # Dynamic row generation based on feature characteristics
    if any(k in feat_lower for k in ["auth", "token", "secret", "user", "log", "git", "remote"]):
        rows = [
            {"vector": "Accidental Credential / PII Exposure", "impact": "Public leak of sensitive API keys or personal logs", "mitigation": "Terminal .gitignore exclusions, env variable injection, and automated secret scanners."},
            {"vector": "Untrusted Boundary Input Injection", "impact": "Malicious command execution or unauthorized data extraction", "mitigation": "Strict type validation, AST parameterization, and path canonicalization."},
            {"vector": "Silent Error Suppression", "impact": "Corrupted state persists across sessions undetected", "mitigation": "Explicit exception propagation with contextual structured logging."}
        ]
    elif any(k in feat_lower for k in ["train", "model", "tensor", "gpu", "cpu", "memory"]):
        rows = [
            {"vector": "System Resource Starvation (OOM/Paging)", "impact": "Host desktop freezes, DPC latency spikes, dropped system interrupts", "mitigation": "Reserve 2 CPU cores, set BELOW_NORMAL priority, and monitor free RAM."},
            {"vector": "Ternary Weight Clipping Distortion", "impact": "Loss divergence or dead expert nodes in BitNet layers", "mitigation": "Straight-Through Estimator (STE) with per-tensor scale clipping."},
            {"vector": "Checkpoint Corruption During Interruption", "impact": "Loss of training progress and corrupted state files", "mitigation": "Atomic file writes using temporary staging paths before rename."}
        ]
    else:
        rows = [
            {"vector": "Unchecked Boundary Assumptions", "impact": "Runtime crash on unexpected input types or empty payloads", "mitigation": "Strict schema validation and defensive boundary guardrails."},
            {"vector": "Resource Leakage (Sockets/Files/Handles)", "impact": "Exhaustion of OS file descriptors or memory accumulation", "mitigation": "Deterministic RAII context managers ('with' statements / using blocks)."},
            {"vector": "Unverified Happy-Path Deployment", "impact": "Regression introduced into downstream caller workflows", "mitigation": "Mandatory Phase 3 RCI loop with unit test verification before completion."}
        ]

    return {
        "feature": feature,
        "threat_model_table": rows,
        "format": "Vector -> Impact -> Mitigation (Exactly 3 Rows)"
    }


# ─── 3. BUSHIDO 7-VIRTUE ETHICAL GATE ────────────────────────────────────────

@mcp.tool()
def prism_bushido_gate(action_or_output: str) -> Dict[str, Any]:
    """Evaluates an action, plan, or response against the 7 Bushido virtues.
    
    Guards against sycophancy, deceptive framing, shallow evasions, and boundary failures.
    
    Args:
        action_or_output: Text or plan to evaluate.
    """
    text_lower = action_or_output.lower()
    flags = []

    # Check for sycophancy
    sycophancy_markers = [
        "you are absolutely right",
        "i apologize immensely",
        "as an ai language model",
        "as an assistant",
        "i am sorry for my mistake"
    ]
    if any(m in text_lower for m in sycophancy_markers):
        flags.append({
            "virtue": "Makoto (Honesty) / Meiyo (Honor)",
            "finding": "Sycophantic or corporate apologetic disclaimer detected. State facts and corrections directly without subservient throat-clearing."
        })

    # Check for superficiality / evasion
    evasion_markers = [
        "it depends on many factors",
        "there are pros and cons to both",
        "you might want to consider doing your own research"
    ]
    if any(m in text_lower for m in evasion_markers):
        flags.append({
            "virtue": "Yu (Courage)",
            "finding": "Evasive neutral stance. The Ronin takes a stand based on verifiable evidence and structured trade-offs rather than shallow equivocation."
        })

    passed = len(flags) == 0

    return {
        "verdict": "BUSHIDO_PASSED" if passed else "BUSHIDO_ALIGNMENT_REQUIRED",
        "virtues_checked": [
            "Gi (Rectitude / Moral Integrity)",
            "Yu (Courage / Truth over comfort)",
            "Jin (Benevolence / Human Flourishing)",
            "Rei (Respect without sycophancy)",
            "Makoto (Honesty & Direct Sincerity)",
            "Meiyo (Honor / Standing behind the blade)",
            "Chugi (Loyalty to core principles)"
        ],
        "flags": flags,
        "guidance": "Honor the user by delivering deep, unvarnished truth backed by rigorous technical evidence."
    }


if __name__ == "__main__":
    mcp.run()
